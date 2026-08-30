"""Compatibility shim for tools that still invoke ``setup.py`` directly.

Project metadata lives in ``pyproject.toml``.
"""

from setuptools import setup


setup()
