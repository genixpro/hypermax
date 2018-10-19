import urwid
import urwid.html_fragment

from pprint import pformat
import numpy
import sys
import os.path
import csv
import yaml
import copy
import datetime
import json
from panwid import DataTable, DataTableColumn
from hypermax.hyperparameter import Hyperparameter


def makeMountedFrame(widget, header):
    content = urwid.Padding(urwid.Filler(widget, height=('relative', 100), top=1, bottom=1), left=1, right=1)
    body = urwid.Frame(urwid.AttrWrap(content, 'frame_body'), urwid.AttrWrap(urwid.Text('  ' + header), 'frame_header'))
    shadow = urwid.Columns(
        [body, ('fixed', 2, urwid.AttrWrap(urwid.Filler(urwid.Text(('background', '  ')), "top"), 'shadow'))])
    shadow = urwid.Frame(shadow, footer=urwid.AttrWrap(urwid.Text(('background', '  ')), 'shadow'))
    padding = urwid.AttrWrap(
        urwid.Padding(urwid.Filler(shadow, height=('relative', 100), top=1, bottom=1), left=1, right=1), 'background')
    return padding

class ScrollableDataTable(DataTable):
    def __init__(self, *args, **kwargs):
        if 'keepColumns' in kwargs:
            self.keepColumns = kwargs.get('keepColumns', []) + ['index']
            del kwargs['keepColumns']
        else:
            self.keepColumns = ['index']

        if 'rowClickCallback' in kwargs:
            self.rowClickCallback = kwargs['rowClickCallback']
            del kwargs['rowClickCallback']
        else:
            self.rowClickCallback = None

        self.localRows = []

        super(ScrollableDataTable, self).__init__(*args, **kwargs)

        self.columnPos = 0

    def keypress(self, widget, key):
        if key == 'right':
            if self.columnPos < len([c for c in self.columns if c.name not in self.keepColumns])-1:
                self.columnPos += 1

            self.toggle_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index >= self.columnPos], show=False)
            self.toggle_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index < self.columnPos], show=True)

        elif key == 'left':
            if self.columnPos > 0:
                self.columnPos -= 1

            self.toggle_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index >= self.columnPos], show=False)
            self.toggle_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index < self.columnPos], show=True)
        elif key =='enter' and self.rowClickCallback!=None:
            self.rowClickCallback(self.localRows[self.focus_position])
        else:
            super(ScrollableDataTable, self).keypress(widget, key)

        if key != 'up' or self.focus_position < 1:
            return key

class ExportCSVPopup(urwid.WidgetWrap):
    signals = ['close']

    """A dialog that appears with nothing but a close button """
    def __init__(self, optimizer):
        header = urwid.Text("Where would you like to save?")

        self.edit = urwid.Edit(edit_text=os.path.join(os.getcwd(), "results.csv"))

        save_button = urwid.Button("Save")
        close_button = urwid.Button("Cancel")
        urwid.connect_signal(close_button, 'click',lambda button: self._emit("close"))
        urwid.connect_signal(save_button, 'click',lambda button: self.saveResults())

        pile = urwid.Pile([header, urwid.Text('\n'), self.edit,urwid.Text(''), urwid.Columns([save_button, close_button])])
        fill = urwid.Filler(pile)
        super(ExportCSVPopup, self).__init__(makeMountedFrame(fill, 'Export File'))

        self.optimizer = optimizer

    def saveResults(self):
        self.optimizer.exportResultsCSV(self.edit.edit_text)
        self._emit('close')


class ExportParametersPopup(urwid.WidgetWrap):
    signals = ['close']

    """A dialog that appears with nothing but a close button """
    def __init__(self, optimizer):
        header = urwid.Text("Where would you like to save?")

        self.edit = urwid.Edit(edit_text=os.path.join(os.getcwd(), "parameters.json"))

        save_button = urwid.Button("Save")
        close_button = urwid.Button("Cancel")
        urwid.connect_signal(close_button, 'click',lambda button: self._emit("close"))
        urwid.connect_signal(save_button, 'click',lambda button: self.saveResults())

        pile = urwid.Pile([header, urwid.Text('\n'), self.edit,urwid.Text(''), urwid.Columns([save_button, close_button])])
        fill = urwid.Filler(pile)
        super(ExportParametersPopup, self).__init__(makeMountedFrame(fill, 'Export File'))

        self.optimizer = optimizer

    def saveResults(self):
        paramKeys = [key for key in self.optimizer.best.keys() if key not in self.optimizer.resultInformationKeys]

        with open(self.edit.edit_text, 'wt') as file:
            json.dump({key:self.optimizer.best[key] for key in paramKeys}, file, indent=4)

        self._emit('close')


class CorrelationGridPopup(urwid.WidgetWrap):
    signals = ['close']

    """A dialog that appears with nothing but a close button """
    def __init__(self, optimizer):

        matrix, labels = optimizer.resultsAnalyzer.computeCorrelations(optimizer)

        columns = [DataTableColumn('field', label='field', width=16, align="right", attr="body", padding=0)]
        for label in labels:
            column = DataTableColumn(label, label=label, width=16, align="right", attr="body", padding=0)
            columns.append(column)

        data = []
        for index, row in enumerate(matrix):
            rowData = {
                'field': labels[index]
            }

            for labelIndex, label in enumerate(labels):
                rowData[label] = row[labelIndex]

            data.append(rowData)

        self.data = data
        self.labels = labels

        table = ScrollableDataTable(columns = columns, data=data)

        close_button = urwid.Button("Cancel")
        urwid.connect_signal(close_button, 'click',lambda button: self._emit("close"))

        export_button = urwid.Button("Export")
        urwid.connect_signal(export_button, 'click',lambda button: self.exportCorrelations())

        buttons = urwid.Filler(urwid.Columns([close_button, export_button]))

        super(CorrelationGridPopup, self).__init__(makeMountedFrame(urwid.Pile([(5, buttons), table]), 'Export File'))

        self.optimizer = optimizer

    def exportCorrelations(self):
        self._emit('close')

        with open('correlations.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=['field'] + self.labels)
            writer.writerows(self.data)


class HumanGuidancePopup(urwid.WidgetWrap):
    signals = ['close']

    """A dialog shows with the human guidance options """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.guidanceOptions = copy.deepcopy(optimizer.humanGuidedATPEOptimizer.guidanceOptions)

        self.parameterLockedValueEdits = {}
        self.parameterMinEdits = {}
        self.parameterMaxEdits = {}
        self.statusLabels = {}

        self.listWalker = urwid.SimpleListWalker(self.generateGrid())
        listbox = urwid.ListBox(self.listWalker)

        close_button = urwid.Button("Close")
        urwid.connect_signal(close_button, 'click',lambda button: self.close())

        buttons = urwid.Filler(urwid.Columns([close_button]))

        super(HumanGuidancePopup, self).__init__(makeMountedFrame(urwid.Pile([(5, buttons), listbox]), 'Apply Human Guidance'))

        self.optimizer = optimizer

    def createParameterEditor(self, parameter, index):
        title = urwid.Text(parameter.name)

        shouldLock = urwid.Button("Lock")
        urwid.connect_signal(shouldLock, 'click',lambda button: self.lockParameter(parameter, index))
        shouldScramble = urwid.Button("Scramble")
        urwid.connect_signal(shouldScramble, 'click',lambda button: self.scrambleParameter(parameter, index))
        shouldRelearn = urwid.Button("Relearn")
        urwid.connect_signal(shouldRelearn, 'click',lambda button: self.refitParameter(parameter, index))

        best = None
        if self.optimizer.best and parameter.name in self.optimizer.best:
            best = urwid.Text("Best: " + str(self.optimizer.best[parameter.name]))
        else:
            best = urwid.Text("Not in best")

        minEdit = urwid.Edit()
        minLabel = urwid.Text("Min")
        maxEdit = urwid.Edit()
        maxLabel = urwid.Text("Max")
        minEdit.set_edit_text(str(parameter.config['min']))
        maxEdit.set_edit_text(str(parameter.config['max']))
        rangeArea = urwid.Columns([minLabel,minEdit,maxLabel,maxEdit])
        self.parameterMinEdits[parameter.name] = minEdit
        self.parameterMaxEdits[parameter.name] = maxEdit
        urwid.connect_signal(minEdit, 'postchange', lambda button, value: self.updateMin(parameter))
        urwid.connect_signal(maxEdit, 'postchange', lambda button, value: self.updateMax(parameter))

        edit = None
        self.parameterLockedValueEdits[parameter.name] = urwid.Edit()
        self.statusLabels[parameter.name] = urwid.Text("")
        status = self.statusLabels[parameter.name]
        found = False
        if not found:
            for lockedParam in self.guidanceOptions['lockedParameters']:
                if lockedParam['variable'] == parameter.name:
                    status.set_text("Locked to " + str(lockedParam['value']))
                    edit = self.parameterLockedValueEdits[parameter.name]
                    edit.set_edit_text (str(lockedParam['value']))
                    shouldLock = urwid.Button("Unlock")
                    urwid.connect_signal(shouldLock, 'click',lambda button: self.cancelSpecialsOnParameter(parameter, index))
        if not found:
            for refitParam in self.guidanceOptions['refitParameters']:
                if refitParam['variable'] == parameter.name:
                    status.set_text("Refitting from trial " + str(refitParam['refitStartTrial']))
                    shouldRelearn = urwid.Button("Stop Relearning")
                    urwid.connect_signal(shouldRelearn, 'click',lambda button: self.cancelSpecialsOnParameter(parameter, index))
        if not found:
            for refitParam in self.guidanceOptions['scrambleParameters']:
                if refitParam['variable'] == parameter.name:
                    status.set_text("Scrambling (random searching)")
                    shouldScramble = urwid.Button("Stop Scrambling")
                    urwid.connect_signal(shouldScramble, 'click',lambda button: self.cancelSpecialsOnParameter(parameter, index))

        if edit is None:
            edit = urwid.Text("")

        if status is None:
            status = urwid.Text("")

        urwid.connect_signal(self.parameterLockedValueEdits[parameter.name], 'postchange', lambda button, value: self.updateLockValue(parameter, index))

        return urwid.Columns([urwid.Columns([('pack', title), ('pack', urwid.Text("      ")), ('pack', best)]), rangeArea, urwid.Columns([('pack', status), ('pack', edit)]), urwid.Columns([shouldLock, shouldScramble, shouldRelearn])])

    def close(self):
        # Convert all the locked values into floats, remove ones which don't convert
        newLockedParams = []
        for param in self.guidanceOptions['lockedParameters']:
            try:
                param['value'] = float(param['value'])
                newLockedParams.append(param)
            except ValueError:
                pass

        self.optimizer.humanGuidedATPEOptimizer.guidanceOptions['lockedParameters'] = newLockedParams
        self.optimizer.humanGuidedATPEOptimizer.guidanceOptions = self.guidanceOptions
        self._emit("close")

    def generateGrid(self):
        parameters = sorted([param for param in Hyperparameter(self.optimizer.config.data['hyperparameters']).getFlatParameters() if param.config['type'] == 'number'], key=lambda param:param.name)

        content = [
            urwid.AttrWrap(self.createParameterEditor(parameter, index), 'body', focus_attr='focus')
            for index, parameter in enumerate(parameters)
        ]
        return content

    def updateMin(self, parameter):
        try:
            parameter.config['min'] = float(self.parameterMinEdits[parameter.name].edit_text)
        except ValueError:
            pass

    def updateMax(self, parameter):
        try:
            parameter.config['max'] = float(self.parameterMaxEdits[parameter.name].edit_text)
        except ValueError:
            pass

    def updateLockValue(self, parameter, index):
        for paramIndex, lockedParam in enumerate(self.guidanceOptions['lockedParameters']):
            if lockedParam['variable'] == parameter.name:
                lockedParam['value'] = self.parameterLockedValueEdits[parameter.name].edit_text
                self.statusLabels[parameter.name].set_text("Locked to " + str(self.parameterLockedValueEdits[parameter.name].edit_text))

    def lockParameter(self, parameter, index):
        self.cancelSpecialsOnParameter(parameter, index)
        self.guidanceOptions['lockedParameters'].append({
            "variable": parameter.name,
            "value": self.parameterLockedValueEdits[parameter.name].edit_text
        })
        self.listWalker.contents[index] = self.createParameterEditor(parameter, index)

    def refitParameter(self, parameter, index):
        self.cancelSpecialsOnParameter(parameter, index)
        self.guidanceOptions['refitParameters'].append({
            "variable": parameter.name,
            "refitStartTrial": len(self.optimizer.results)
        })
        self.listWalker.contents[index] = self.createParameterEditor(parameter, index)

    def scrambleParameter(self, parameter, index):
        self.cancelSpecialsOnParameter(parameter, index)
        self.guidanceOptions['scrambleParameters'].append({
            "variable": parameter.name
        })
        self.listWalker.contents[index] = self.createParameterEditor(parameter, index)

    def cancelSpecialsOnParameter(self, parameter, index):
        for paramIndex, lockedParam in enumerate(self.guidanceOptions['lockedParameters']):
            if lockedParam['variable'] == parameter.name:
                del self.guidanceOptions['lockedParameters'][paramIndex]
                break
        for paramIndex, refitParam in enumerate(self.guidanceOptions['refitParameters']):
            if refitParam['variable'] == parameter.name:
                del self.guidanceOptions['refitParameters'][paramIndex]
                break
        for paramIndex, scrambleParam in enumerate(self.guidanceOptions['scrambleParameters']):
            if scrambleParam['variable'] == parameter.name:
                del self.guidanceOptions['scrambleParameters'][paramIndex]

        self.listWalker.contents[index] = self.createParameterEditor(parameter, index)

class MessagePopup(urwid.WidgetWrap):
    signals = ['close']

    """A dialog that appears with nothing but a close button """
    def __init__(self, message):

        text = urwid.Text(message)
        close_button = urwid.Button("Cancel")
        urwid.connect_signal(close_button, 'click',lambda button: self._emit("close"))

        super(MessagePopup, self).__init__(makeMountedFrame(urwid.Filler(urwid.Pile([text, close_button])), 'Warning'))

class PopupContainer(urwid.PopUpLauncher):
    def __init__(self, widget, optimizer):
        super(PopupContainer, self).__init__(widget)

        self.optimizer = optimizer

    def open_pop_up_with_widget(self, type, size=(('relative', 50), 15)):
        self.type = type
        self.size = size
        self.open_pop_up()

    def create_pop_up(self):
        pop_up = self.type
        urwid.connect_signal(pop_up, 'close', lambda button: self.close_pop_up())
        return urwid.AttrWrap(urwid.Filler(urwid.Padding(pop_up, 'center', width=self.size[0]), height=self.size[1]), 'background')

    def get_pop_up_parameters(self):
        return {'left':0, 'top':0, 'overlay_width':('relative', 100.0), 'overlay_height':('relative', 100.0)}




class ScrollableTextArea(urwid.WidgetWrap):
    signals = ['close']

    """A text area with fixed contents that can be scrolled."""
    def __init__(self):
        self.content = []

        self.listWalker = urwid.SimpleFocusListWalker(self.content)
        self.listbox = urwid.ListBox(self.listWalker)


        super(ScrollableTextArea, self).__init__(self.listbox)

    def setText(self, text):
        pass


def launchHypermaxUI(optimizer):
    screen = urwid.raw_display.Screen()

    palette = [
        ('background', 'white', 'dark blue', 'standout'),
        ('body', 'dark gray', 'light gray', 'standout'),
        ('frame_header', 'white', 'dark gray', 'standout'),
        ('frame_shadow', 'black', 'black', 'standout'),
        ('frame_body', 'dark gray', 'light gray', 'standout'),
        ('tab_buttons', 'white', 'dark red', 'standout'),

        ('focus', 'black', 'light gray', 'underline'),
        ('reverse', 'light gray', 'black'),
        ('header', 'white', 'dark red', 'bold'),
        ('important', 'dark blue', 'light gray', ('standout', 'underline')),
        ('editfc', 'white', 'dark blue', 'bold'),
        ('editbx', 'light gray', 'dark blue'),
        ('editcp', 'black', 'light gray', 'standout'),
        ('bright', 'dark gray', 'light gray', ('bold', 'standout')),
        ('buttn', 'black', 'dark cyan'),
        ('buttnf', 'white', 'dark blue', 'bold'),

        ('graph_bg', 'black', 'light gray'),
        ('graph_bar', 'black', 'dark cyan', 'bold'),
        ('graph_label', 'dark cyan', 'light gray', 'bold'),

        ('table_row_body', 'dark gray', 'light gray', 'standout'),
        ('table_row_header', 'dark gray', 'light gray', 'underline'),
        ('table_row_footer', 'dark gray', 'light gray', 'standout'),

        ('table_row_body focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_body column_focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_body highlight', 'light gray', 'dark gray', 'standout'),
        ('table_row_body highlight focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_body highlight column_focused', 'light gray', 'dark gray', 'standout'),

        ('table_row_header focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_header column_focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_header highlight', 'light gray', 'dark gray', 'standout'),
        ('table_row_header highlight focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_header highlight column_focused', 'light gray', 'dark gray', 'standout'),

        ('table_row_footer focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_footer column_focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_footer highlight', 'light gray', 'dark gray', 'standout'),
        ('table_row_footer highlight focused', 'light gray', 'dark gray', 'standout'),
        ('table_row_footer highlight column_focused', 'light gray', 'dark gray', 'standout'),
    ]

    def onExitClicked(widget):
        raise urwid.ExitMainLoop()

    def viewHyperparameterCorrelations():
        if optimizer.results:
            popupContainer.open_pop_up_with_widget(CorrelationGridPopup(optimizer), size=(('relative', 95), ('relative', 95)))
        else:
            popupContainer.open_pop_up_with_widget(MessagePopup('No results to compute correlation on yet.'), size=(('relative', 95), ('relative', 95)))

    def viewHumanGuidance():
        popupContainer.open_pop_up_with_widget(HumanGuidancePopup(optimizer), size=(('relative', 95), ('relative', 95)))

    def exportBestParameters():
        if optimizer.best:
            popupContainer.open_pop_up_with_widget(ExportParametersPopup(optimizer))
        else:
            popupContainer.open_pop_up_with_widget(MessagePopup('There is no best model to export yes.'), size=(('relative', 95), ('relative', 95)))

    popupContainer = None
    graph = None
    graphVscale = None
    graphColumns = None
    currentTrialsLeft = None
    currentTrialsMiddle = None
    currentTrialsRight = None
    currentBestLeft = None
    currentBestRight = None
    def makeMainMenu():
        content = [
            urwid.AttrWrap(urwid.Button("Export Results to CSV", on_press=lambda button: popupContainer.open_pop_up_with_widget(ExportCSVPopup(optimizer))), 'body', focus_attr='focus'),
            urwid.AttrWrap(urwid.Button('View Hyperparameter Correlations', on_press=lambda button: viewHyperparameterCorrelations()), 'body', focus_attr='focus'),
            urwid.AttrWrap(urwid.Button('Export Best Hyperparameters to File', on_press=lambda button: exportBestParameters()), 'body', focus_attr='focus'),
            urwid.AttrWrap(urwid.Button('Apply Human Guidance', on_press=lambda button: viewHumanGuidance()), 'body', focus_attr='focus'),
            urwid.AttrWrap(urwid.Button('Exit', on_press=onExitClicked), 'body', focus_attr='focus')
        ]

        listbox = urwid.ListBox(urwid.SimpleFocusListWalker(content))

        menu = makeMountedFrame(urwid.AttrWrap(listbox, 'body'), header='Hypermax v0.1')

        return menu

    def makeGraphArea():
        nonlocal graph, graphVscale, graphColumns
        graph = urwid.BarGraph(attlist=['graph_bg', 'graph_bar'])
        graph.set_data([], top=1)

        labels = [[i, '{:.3f}'.format(i)] for i in numpy.arange(0.0, 1.0, 0.01)]
        graphVscale = urwid.AttrWrap(urwid.GraphVScale(labels=labels, top=1), 'graph_label')
        graphColumns = urwid.Columns([(7, urwid.Padding(graphVscale, left=0, right=1)), graph, (7, urwid.Padding(graphVscale, left=1, right=0))])

        graphFrame = makeMountedFrame(graphColumns, 'Rolling Loss')
        return graphFrame

    def makeCurrentTrialsArea():
        nonlocal currentTrialsLeft,currentTrialsMiddle, currentTrialsRight
        currentTrialsLeft = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        currentTrialsMiddle = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        currentTrialsRight = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        columns = urwid.Columns([currentTrialsLeft, currentTrialsMiddle, currentTrialsRight])
        return makeMountedFrame(urwid.Filler(columns), "Current Trials")

    currentOptimizationParamsLeft = None
    currentOptimizationParamsMiddle = None
    currentOptimizationParamsRight = None
    def makeOptimizationParametersArea():
        nonlocal currentOptimizationParamsLeft, currentOptimizationParamsMiddle, currentOptimizationParamsRight
        currentOptimizationParamsLeft = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        currentOptimizationParamsMiddle = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        currentOptimizationParamsRight = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        columns = urwid.Columns([currentOptimizationParamsLeft, currentOptimizationParamsMiddle, currentOptimizationParamsRight])
        return makeMountedFrame(urwid.Filler(columns), "Optimization Parameters")

    optimizationDetailsLeft = None
    optimizationDetailsRight = None
    def makeOptimizationDetailsArea():
        nonlocal optimizationDetailsLeft, optimizationDetailsRight
        optimizationDetailsLeft = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        optimizationDetailsRight = urwid.AttrWrap(urwid.Text(markup=''), 'body')
        columns = urwid.Columns([optimizationDetailsLeft, optimizationDetailsRight])
        return makeMountedFrame(urwid.Filler(columns), "Optimization Details")

    def makeCurrentBestArea():
        nonlocal currentBestLeft, currentBestRight
        currentBestLeft = urwid.Text(markup='')
        currentBestRight = urwid.Text(markup='')
        columns = urwid.Columns([currentBestLeft, (1, urwid.Text(markup=' ')), currentBestRight])
        return makeMountedFrame(urwid.AttrWrap(urwid.Filler(columns), 'frame_body'), "Current Best")

    # trialsList = urwid.SimpleFocusListWalker([])
    trialsTable = None
    tableResultsSize = 0
    def makeTrialsView():
        nonlocal trialsTable
        def displayTrialDetails(currentTrial):
            popupContainer.open_pop_up_with_widget(MessagePopup(json.dumps(currentTrial, indent=4)),
                                                   size=(('relative', 95), ('relative', 95)))

        # listbox = urwid.ListBox(trialsList)

        columns = [
            # DataTableColumn("uniqueid", width=10, align="right", padding=1),
            DataTableColumn("trial",
                            label="Trial",
                            width=6,
                            align="right",
                            attr="body",
                            padding=0
                            # footer_fn=lambda column, values: sum(v for v in values if v is not None)),
                            ),
            DataTableColumn("loss",
                            label="Loss",
                            width=10,
                            align="right",
                            attr="body",
                            padding=0,
                            # footer_fn=lambda column, values: sum(v for v in values if v is not None)),
                            ),
            DataTableColumn("time",
                            label="Time",
                            width=6,
                            align="right",
                            attr="body",
                            padding=0
                            # footer_fn=lambda column, values: sum(v for v in values if v is not None)),
                            ),
        ]

        keys = Hyperparameter(optimizer.config.data['hyperparameters']).getFlatParameterNames()

        for key in sorted(keys):
            columns.append(
            DataTableColumn(key[5:],
                            label=key[5:],
                            width=len(key[5:])+2,
                            align="right",
                            attr="body",
                            padding=0
                            # footer_fn=lambda column, values: sum(v for v in values if v is not None)),
                            ))

        trialsTable = ScrollableDataTable(columns=columns, data=[{}], keepColumns=['trial', 'loss', 'time'],rowClickCallback=displayTrialDetails)

        return makeMountedFrame(urwid.AttrWrap(trialsTable, 'body'), 'Trials')

    currentBestArea = makeCurrentBestArea()

    columns = urwid.Columns([makeMainMenu(), currentBestArea])

    currentTrialsArea = makeCurrentTrialsArea()
    graphArea = makeGraphArea()
    trialsArea = makeTrialsView()
    optimizationParametersArea = makeOptimizationParametersArea()
    optimizationDetailsArea = makeOptimizationDetailsArea()

    bottomArea = None

    def showLossGraph(widget):
        bottomArea.contents[1] = (graphArea, (urwid.WEIGHT, 1))

    def showCurrentTrials(widget):
        bottomArea.contents[1] = (currentTrialsArea, (urwid.WEIGHT, 1))

    def showTrials(widget):
        bottomArea.contents[1] = (trialsArea, (urwid.WEIGHT, 1))

    def showOptimizationParameters(widget):
        bottomArea.contents[1] = (optimizationParametersArea, (urwid.WEIGHT, 1))

    def showOptimizationDetails(widget):
        bottomArea.contents[1] = (optimizationDetailsArea, (urwid.WEIGHT, 1))

    bottomButtons = urwid.Columns([
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('Loss', on_press=showLossGraph), 'tab_buttons'), left=1, right=5)),
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('Current Trials', on_press=showCurrentTrials), 'tab_buttons'), left=5, right=5)),
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('ATPE Parameters', on_press=showOptimizationParameters), 'tab_buttons'), left=5, right=5)),
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('ATPE Details', on_press=showOptimizationDetails), 'tab_buttons'), left=5, right=5)),
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('Trials', on_press=showTrials), 'tab_buttons'), left=5, right=1)),
    ])

    bottomArea = urwid.Pile([(2, bottomButtons), graphArea])

    background = urwid.Frame(urwid.Pile([
        urwid.AttrWrap(columns, 'background'),
        urwid.AttrWrap(bottomArea, 'background')
    ]))

    background = PopupContainer(background, optimizer)
    popupContainer = background

    def unhandled(key):
        if key == 'f8':
            raise urwid.ExitMainLoop()

    loop = urwid.MainLoop(background, palette, screen, pop_ups=True, unhandled_input=unhandled)

    def formatParamVal(value):
        if isinstance(value, float):
            return float('{:.4E}'.format(value))
        else:
            return value

    def splitObjectIntoColumns(obj, num_columns):
        texts = [""] * num_columns
        if obj is None:
            return texts

        paramKeys = sorted(list(obj.keys()))

        cutoffs = []
        for cutoff in range(num_columns+1):
            cutoffs.append(int((len(paramKeys) + 1) * (cutoff) / 3))

        for column in range(num_columns):
            columnKeys = paramKeys[cutoffs[column]:cutoffs[column+1]]
            texts[column] += yaml.dump({key: formatParamVal(obj[key]) for key in columnKeys}, default_flow_style=False)

        lines = max(*[text.count("\n") for text in texts])

        for index in range(len(texts)):
            texts[index] += "\n" * (lines - texts[index].count("\n"))

        return tuple(texts)

    try:
        loop.start()
        while True:
            loop.draw_screen()

            loop.screen.set_input_timeouts(0.1)
            keys, raw = loop.screen.get_input(True)
            keys = loop.input_filter(keys, raw)
            if keys:
                loop.process_input(keys)
            if 'window resize' in keys:
                loop.screen_size = None


            currentTrialsLeftText = ""
            currentTrialsMiddleText = ""
            currentTrialsRightText = ""
            for trial in optimizer.currentTrials:
                trial = trial

                leftText, middleText, rightText = splitObjectIntoColumns(trial['params'], 3)

                runningTime = (datetime.datetime.now() - trial['start']).total_seconds()

                leftText = "Time: " + str(formatParamVal(runningTime)) + " seconds\n\n" + leftText
                middleText = "Trial: #" + str(trial['trial']) + " \n\n" + middleText
                rightText = "\n\n" + rightText

                currentTrialsLeftText += leftText
                currentTrialsMiddleText += middleText
                currentTrialsRightText += rightText

            currentTrialsLeft.set_text(currentTrialsLeftText)
            currentTrialsMiddle.set_text(currentTrialsMiddleText)
            currentTrialsRight.set_text(currentTrialsRightText)

            optimizationParamsLeftText, optimizationParamsMiddleText, optimizationParamsRightText = splitObjectIntoColumns(optimizer.lastATPEParameters, 3)
            currentOptimizationParamsLeft.set_text(optimizationParamsLeftText)
            currentOptimizationParamsMiddle.set_text(optimizationParamsMiddleText)
            currentOptimizationParamsRight.set_text(optimizationParamsRightText)

            optimizationDetailsLeftText, optimizationDetailsRightText = splitObjectIntoColumns(optimizer.atpeParamDetails, 2)
            optimizationDetailsLeft.set_text(optimizationDetailsLeftText)
            optimizationDetailsRight.set_text(optimizationDetailsRightText)

            if optimizer.best:
                paramKeys = [key for key in optimizer.best.keys() if key not in optimizer.resultInformationKeys]

                cutoff = int((len(paramKeys)+1)/2)
                leftParamKeys = paramKeys[:cutoff]
                rightParamKeys = paramKeys[cutoff:]

                bestLeftText = yaml.dump({key:formatParamVal(optimizer.best[key]) for key in leftParamKeys}, default_flow_style=False)
                bestRightText = yaml.dump({key:formatParamVal(optimizer.best[key]) for key in rightParamKeys}, default_flow_style=False)

                bestLeftText += "\n\nLoss: " + str(optimizer.bestLoss)
                bestRightText += "\n\nTime: " + str(optimizer.best['time']) + " (s)"
                bestLeftText += "\nTrials: " + str(optimizer.completed()) + "/" + str(optimizer.totalTrials)

                if optimizer.resultsAnalyzer.totalCharts > 0 and optimizer.resultsAnalyzer.completedCharts < optimizer.resultsAnalyzer.totalCharts:
                    bestRightText += "\nCharts: " + str(optimizer.resultsAnalyzer.completedCharts) + "/" + str(optimizer.resultsAnalyzer.totalCharts)

                currentBestLeft.set_text(bestLeftText)
                currentBestRight.set_text(bestRightText)

            if len(optimizer.results) > 0:
                numResultsToAdd = max(0, len(optimizer.results) - tableResultsSize)
                if numResultsToAdd > 0:
                    resultsToAdd = optimizer.results[-numResultsToAdd:]

                    newResults = []
                    for result in resultsToAdd:
                        newResult = {}
                        for key in result.keys():
                            if isinstance(result[key], float):
                                if result[key] > 1e-3:
                                    newResult[key] = '{:.3F}'.format(result[key])
                                else:
                                    newResult[key] = '{:.3E}'.format(result[key])
                            else:
                                newResult[key] = str(result[key])
                        newResults.append(newResult)

                    trialsTable.append_rows(newResults)
                    trialsTable.apply_filters()
                    tableResultsSize += len(resultsToAdd)
                trialsTable.localRows = optimizer.results

            if len(optimizer.results) > 1:
                allResults = numpy.array([result['loss'] for result in optimizer.results if isinstance(result['loss'], float)])

                windowSize = max(0, min(10, len(allResults)-10))

                allResults = [numpy.median(allResults[max(0, index-windowSize):index+1]) for index in range(0, len(allResults), 1)]

                top = None
                bottom = None
                data = []
                for result in allResults[-min(len(allResults), 50):]:
                    data.append([result])
                    if top is None or result > top:
                        top = result
                    if bottom is None or result < bottom:
                        bottom = result

                if top is None:
                    top = 1

                if bottom is None:
                    bottom = 0

                if '{:.3E}'.format(bottom) == '{:.3E}'.format(top):
                    top = bottom + 1

                graph_range = top - bottom

                graph.set_data([[d[0] - bottom] for d in data], graph_range)

                labels = [[i - bottom, '{:.3f}'.format(i)] for i in numpy.arange(bottom, top, graph_range/100.0)]
                graphVscale = urwid.AttrWrap(urwid.GraphVScale(labels=labels, top=graph_range), 'graph_label')
                graphColumns.contents[0] = (urwid.Padding(graphVscale, left=0, right=1), (urwid.GIVEN, 7, False))
                graphColumns.contents[2] = (urwid.Padding(graphVscale, left=1, right=0), (urwid.GIVEN, 7, False))

            # if len(optimizer.results) > 0:
            #     if optimizer.results[-1]['status'] != 'ok':
            #         statusText += optimizer.results[-1]['log']
            #         status.set_text(statusText)


    except urwid.ExitMainLoop:
        pass
    finally:
        loop.stop()

