"""Lossless ConfigSpace-to-Hypermax adapter for external mixed spaces.

ConfigSpace represents conditionals as a graph.  Hypermax represents them as
nested ``oneOf`` objects.  The adapter chooses the deepest categorical parent
as each child's structural parent; remaining ancestor predicates are already
satisfied by the containing branch.  This covers the condition graphs in all
four prescribed YAHPO scenarios without flattening inactive parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _atomic_conditions(condition: Any) -> list[Any]:
    components = getattr(condition, "components", None)
    if components is None:
        return [condition]
    output = []
    for component in components:
        output.extend(_atomic_conditions(component))
    return output


def _allowed_values(condition: Any) -> set[Any]:
    if hasattr(condition, "values"):
        return set(condition.values)
    if hasattr(condition, "value"):
        return {condition.value}
    raise TypeError(f"Unsupported ConfigSpace condition: {type(condition).__name__}")


@dataclass(frozen=True)
class ParameterNode:
    name: str
    hyperparameter: Any
    children: tuple[str, ...]


class ConfigSpaceAdapter:
    """Expose one ConfigSpace through Hypermax, Optuna, and flat dictionaries."""

    def __init__(self, config_space: Any):
        self.config_space = config_space
        self.hyperparameters = {hp.name: hp for hp in config_space.values()}
        self.conditions: dict[str, list[tuple[str, set[Any]]]] = {}
        for condition in config_space.conditions:
            atoms = _atomic_conditions(condition)
            child = atoms[0].child.name
            self.conditions[child] = [
                (atom.parent.name, _allowed_values(atom)) for atom in atoms
            ]

        depths: dict[str, int] = {}

        def depth(name: str, visiting: set[str] | None = None) -> int:
            if name in depths:
                return depths[name]
            visiting = set() if visiting is None else visiting
            if name in visiting:
                raise ValueError(f"Cyclic ConfigSpace condition graph at {name}")
            visiting.add(name)
            parents = [parent for parent, _ in self.conditions.get(name, [])]
            value = 0 if not parents else 1 + max(depth(parent, visiting) for parent in parents)
            visiting.remove(name)
            depths[name] = value
            return value

        for name in self.hyperparameters:
            depth(name)
        self.depths = depths

        self.structural_parent: dict[str, str] = {}
        children: dict[str, list[str]] = {}
        for child, predicates in self.conditions.items():
            parent = max((name for name, _ in predicates), key=lambda name: depths[name])
            self.structural_parent[child] = parent
            children.setdefault(parent, []).append(child)
        self.nodes = {
            name: ParameterNode(
                name,
                hp,
                tuple(sorted(children.get(name, []))),
            )
            for name, hp in self.hyperparameters.items()
        }
        self.constant_values = {
            name: hp.value
            for name, hp in self.hyperparameters.items()
            if self._is_constant(hp)
        }
        self.roots = tuple(
            sorted(
                name
                for name, hp in self.hyperparameters.items()
                if name not in self.structural_parent and not self._is_constant(hp)
            )
        )

    @staticmethod
    def _kind(hp: Any) -> str:
        name = type(hp).__name__
        if "Categorical" in name or "Ordinal" in name:
            return "categorical"
        if "Integer" in name:
            return "integer"
        if "Float" in name:
            return "float"
        if name == "Constant":
            return "constant"
        raise TypeError(f"Unsupported ConfigSpace hyperparameter: {name}")

    @classmethod
    def _is_constant(cls, hp: Any) -> bool:
        return cls._kind(hp) == "constant"

    @staticmethod
    def _choices(hp: Any) -> tuple[Any, ...]:
        if hasattr(hp, "choices"):
            return tuple(hp.choices)
        if hasattr(hp, "sequence"):
            return tuple(hp.sequence)
        raise TypeError(f"Conditional parent {hp.name} is not categorical")

    def _child_active(self, child: str, parent: str, value: Any) -> bool:
        for predicate_parent, allowed in self.conditions.get(child, []):
            if predicate_parent == parent:
                return value in allowed
        return False

    def _hypermax_leaf(self, hp: Any) -> dict[str, Any]:
        kind = self._kind(hp)
        if kind == "categorical":
            return {"enum": list(self._choices(hp))}
        if kind == "constant":
            return {"constant": hp.value}
        config = {
            "type": "number",
            "mode": "uniform",
            "min": float(hp.lower),
            "max": float(hp.upper),
            "scaling": "logarithmic" if bool(getattr(hp, "log", False)) else "linear",
        }
        if kind == "integer":
            config["rounding"] = 1
        return config

    def _hypermax_node(self, name: str) -> dict[str, Any]:
        node = self.nodes[name]
        if not node.children:
            return self._hypermax_leaf(node.hyperparameter)
        choices = self._choices(node.hyperparameter)
        branches = []
        for value in choices:
            properties: dict[str, Any] = {"__value": {"constant": value}}
            for child in node.children:
                if self._child_active(child, name, value):
                    properties[child] = self._hypermax_node(child)
            branches.append({"type": "object", "properties": properties})
        return {"oneOf": branches}

    def hypermax_space(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {name: self._hypermax_node(name) for name in self.roots},
        }

    def _structured_node(self, name: str, values: dict[str, Any]) -> Any:
        node = self.nodes[name]
        kind = self._kind(node.hyperparameter)
        value = values[name]
        if not node.children:
            return int(value) if kind == "integer" else value
        choices = self._choices(node.hyperparameter)
        index = choices.index(value)
        output: dict[str, Any] = {"$index": index, "__value": value}
        for child in node.children:
            if child in values and self._child_active(child, name, value):
                output[child] = self._structured_node(child, values)
        return output

    def to_structured(self, values: dict[str, Any]) -> dict[str, Any]:
        return {name: self._structured_node(name, values) for name in self.roots}

    def _flatten_node(self, name: str, value: Any, output: dict[str, Any]) -> None:
        node = self.nodes[name]
        kind = self._kind(node.hyperparameter)
        if not node.children:
            output[name] = int(round(float(value))) if kind == "integer" else value
            return
        selected = value["__value"]
        output[name] = selected
        for child in node.children:
            if child in value and self._child_active(child, name, selected):
                self._flatten_node(child, value[child], output)

    def to_flat(self, structured: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = dict(self.constant_values)
        for name in self.roots:
            self._flatten_node(name, structured[name], output)
        # ConfigSpace performs final legality and type validation.
        configuration = self.config_space.Configuration(
            self.config_space, values=output, allow_inactive_with_values=False
        ) if hasattr(self.config_space, "Configuration") else None
        if configuration is not None:
            return dict(configuration)
        from ConfigSpace import Configuration

        return dict(Configuration(self.config_space, values=output, allow_inactive_with_values=False))

    def _suggest_node(self, trial: Any, name: str, output: dict[str, Any]) -> None:
        node = self.nodes[name]
        hp = node.hyperparameter
        kind = self._kind(hp)
        if kind == "categorical":
            value = trial.suggest_categorical(name, list(self._choices(hp)))
        elif kind == "integer":
            value = trial.suggest_int(name, int(hp.lower), int(hp.upper), log=bool(hp.log))
        elif kind == "float":
            value = trial.suggest_float(name, float(hp.lower), float(hp.upper), log=bool(hp.log))
        else:
            value = hp.value
        output[name] = value
        for child in node.children:
            if self._child_active(child, name, value):
                self._suggest_node(trial, child, output)

    def suggest_optuna(self, trial: Any) -> dict[str, Any]:
        output: dict[str, Any] = dict(self.constant_values)
        for name in self.roots:
            self._suggest_node(trial, name, output)
        return output
