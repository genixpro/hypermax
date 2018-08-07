import urwid

from pprint import pformat
import numpy
import sys
import os.path
import csv
import yaml
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
        return makeMountedFrame(urwid.Filler(columns), "Status")

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

        trialsTable = ScrollableDataTable(columns=columns, data=[{}], keepColumns=['trial', 'loss', 'time'])

        return makeMountedFrame(urwid.AttrWrap(trialsTable, 'body'), 'Trials')

    currentBestArea = makeCurrentBestArea()

    columns = urwid.Columns([makeMainMenu(), currentBestArea])

    currentTrialsArea = makeCurrentTrialsArea()
    graphArea = makeGraphArea()
    trialsArea = makeTrialsView()

    bottomArea = None

    def showLossGraph(widget):
        bottomArea.contents[1] = (graphArea, (urwid.WEIGHT, 1))

    def showCurrentTrials(widget):
        bottomArea.contents[1] = (currentTrialsArea, (urwid.WEIGHT, 1))

    def showTrials(widget):
        bottomArea.contents[1] = (trialsArea, (urwid.WEIGHT, 1))

    bottomButtons = urwid.Columns([
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('Loss', on_press=showLossGraph), 'tab_buttons'), left=1, right=5)),
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('Current Trials', on_press=showCurrentTrials), 'tab_buttons'), left=5, right=5)),
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

    try:
        loop.start()
        while True:
            loop.draw_screen()

            loop.screen.set_input_timeouts(0.1)
            keys, raw = loop.screen.get_input(True)
            keys = loop.input_filter(keys, raw)
            if keys:
                loop.process_input(keys)

            def formatParamVal(value):
                if isinstance(value, float):
                    return float('{:.4E}'.format(value))
                else:
                    return value

            currentTrialsLeftText = ""
            currentTrialsMiddleText = ""
            currentTrialsRightText = ""
            for trial in optimizer.currentTrials:
                trial = trial

                paramKeys = sorted(list(trial['params'].keys()))

                leftCutoff = int((len(paramKeys)+1)/3)
                middleCutoff = int((len(paramKeys)+1)*2/3)

                leftParamKeys = paramKeys[:leftCutoff]
                middleParamKeys = paramKeys[leftCutoff:middleCutoff]
                rightParamKeys = paramKeys[middleCutoff:]

                runningTime = (datetime.datetime.now() - trial['start']).total_seconds()

                leftText = "Time: " + str(formatParamVal(runningTime)) + " seconds\n\n"
                middleText = "Trial: #" + str(trial['trial']) + " \n\n"
                rightText = "\n"

                leftText += yaml.dump({key:formatParamVal(trial['params'][key]) for key in leftParamKeys}, default_flow_style=False)
                middleText += yaml.dump({key:formatParamVal(trial['params'][key]) for key in middleParamKeys}, default_flow_style=False)
                rightText += yaml.dump({key:formatParamVal(trial['params'][key]) for key in rightParamKeys}, default_flow_style=False)

                lines = max(leftText.count("\n"), middleText.count("\n"), rightText.count("\n"))
                leftText += "\n" * (lines - leftText.count("\n"))
                middleText += "\n" * (lines - middleText.count("\n"))
                rightText += "\n" * (lines - rightText.count("\n"))

                currentTrialsLeftText += leftText
                currentTrialsMiddleText += middleText
                currentTrialsRightText += rightText

            currentTrialsLeft.set_text(currentTrialsLeftText)
            currentTrialsMiddle.set_text(currentTrialsMiddleText)
            currentTrialsRight.set_text(currentTrialsRightText)

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

