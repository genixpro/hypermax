import urwid

from pprint import pformat
import numpy
import sys
import os.path
from panwid import DataTable, DataTableColumn
from hypermax.hyperparameter import Hyperparameter


def makeMountedFrame(widget, header):
    body = urwid.Frame(urwid.AttrWrap(widget, 'frame_body'), urwid.AttrWrap(urwid.Text('  ' + header), 'frame_header'))
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
            if self.columnPos < len(self.columns):
                self.columnPos += 1

            self.hide_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index > self.columnPos])
            self.show_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index <= self.columnPos])

        elif key == 'left':
            if self.columnPos > 0:
                self.columnPos -= 1

            self.hide_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index > self.columnPos])
            self.show_columns([column.name for index, column in enumerate(c for c in self.columns if c.name not in self.keepColumns) if index <= self.columnPos])
        else:
            super(ScrollableDataTable, self).keypress(widget, key)

        if key != 'up' or self.focus_position < 1:
            return key

class ExportFilePopup(urwid.WidgetWrap):
    signals = ['close']

    """A dialog that appears with nothing but a close button """
    def __init__(self, optimizer):
        header = urwid.Text("Where would you like to save?")

        self.edit = urwid.Edit(edit_text=os.path.join(os.getcwd(), "results.csv"))

        save_button = urwid.Button("save")
        close_button = urwid.Button("Cancel")
        urwid.connect_signal(close_button, 'click',lambda button: self._emit("close"))
        urwid.connect_signal(save_button, 'click',lambda button: self.saveResults())

        pile = urwid.Pile([header, urwid.Text('\n'), self.edit,urwid.Text(''), urwid.Columns([save_button, close_button])])
        fill = urwid.Filler(pile)
        super(ExportFilePopup, self).__init__(makeMountedFrame(fill, 'Export File'))

        self.optimizer = optimizer

    def saveResults(self):
        self.optimizer.exportCSV(self.edit.edit_text)
        self._emit('close')

class PopupContainer(urwid.PopUpLauncher):
    def __init__(self, widget, optimizer):
        super(PopupContainer, self).__init__(widget)

        self.optimizer = optimizer

    def create_pop_up(self):
        pop_up = ExportFilePopup(self.optimizer)
        urwid.connect_signal(pop_up, 'close', lambda button: self.close_pop_up())
        return urwid.AttrWrap(urwid.Filler(urwid.Padding(pop_up, 'center', width=('relative', 50)), height=15), 'background')

    def get_pop_up_parameters(self):
        return {'left':0, 'top':0, 'overlay_width':('relative', 100.0), 'overlay_height':('relative', 100.0)}


def launchHypermaxUI(optimizer):
    screen = urwid.raw_display.Screen()

    palette = [
        ('background', 'white', 'dark blue', 'standout'),
        ('body', 'dark gray', 'light gray', 'standout'),
        ('frame_header', 'white', 'dark gray', 'standout'),
        ('frame_shadow', 'black', 'black', 'standout'),
        ('frame_body', 'black', 'light gray', 'standout'),
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

    def onExportResultsClicked(widget):
        pass

    popupContainer = None
    graph = None
    graphVscale = None
    graphColumns = None
    status = None
    def makeMainMenu():
        content = [
            urwid.AttrWrap(urwid.Button("Export Results to CSV", on_press=lambda button: popupContainer.open_pop_up()), 'body', focus_attr='focus'),
            urwid.AttrWrap(urwid.Button('View Hyper Parameters'), 'body', focus_attr='focus'),
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
        graphColumns = urwid.Columns([(10, urwid.Padding(graphVscale, left=3, right=0)), graph, (10, urwid.Padding(graphVscale, left=1, right=2))])

        graphFrame = makeMountedFrame(graphColumns, 'Rolling Loss')
        return graphFrame

    def makeStatusArea():
        nonlocal status
        status = urwid.Text(markup='')
        return makeMountedFrame(urwid.Filler(status), "Status")

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
                            width=8,
                            align="right",
                            attr="body",
                            padding=0
                            # footer_fn=lambda column, values: sum(v for v in values if v is not None)),
                            ),
            DataTableColumn("loss",
                            label="Loss",
                            width=8,
                            align="right",
                            attr="body",
                            padding=0,
                            # footer_fn=lambda column, values: sum(v for v in values if v is not None)),
                            )
        ]

        keys = Hyperparameter(optimizer.config.data['hyperparameters']).getFlatParameterNames()

        for key in keys:
            columns.append(
            DataTableColumn(key[5:],
                            label=key[5:],
                            width=20,
                            align="right",
                            attr="body",
                            padding=0
                            # footer_fn=lambda column, values: sum(v for v in values if v is not None)),
                            ))

        trialsTable = ScrollableDataTable(columns=columns, data=[{}], keepColumns=['trial', 'loss'])

        return makeMountedFrame(urwid.AttrWrap(trialsTable, 'body'), 'Trials')

    columns = urwid.Columns([makeMainMenu(), urwid.Filler(urwid.Text(''))])


    statusArea = makeStatusArea()
    graphArea = makeGraphArea()
    trialsArea = makeTrialsView()

    bottomArea = None

    def showLossGraph(widget):
        bottomArea.contents[1] = (graphArea, (urwid.WEIGHT, 1))

    def showStatus(widget):
        bottomArea.contents[1] = (statusArea, (urwid.WEIGHT, 1))

    def showTrials(widget):
        bottomArea.contents[1] = (trialsArea, (urwid.WEIGHT, 1))

    bottomButtons = urwid.Columns([
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('Loss', on_press=showLossGraph), 'tab_buttons'), left=1, right=5)),
        urwid.Filler(urwid.Padding(urwid.AttrWrap(urwid.Button('Status', on_press=showStatus), 'tab_buttons'), left=5, right=5)),
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

            statusText = "Completed: " + str(optimizer.completed()) + "/" + str(optimizer.totalTrials)
            status.set_text(statusText)

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

            allResults = numpy.array([result['loss'] for result in optimizer.results])

            allResults = [numpy.mean(allResults[max(0, index-10):min(len(allResults)-1, index+10)]) for index in range(0, len(allResults), 1)]

            top = 0
            data = []
            for result in allResults[-min(len(allResults), 50):]:
                data.append([result])
                top = max(top, result*1.1)

            graph.set_data(data, top)

            if top == 0:
                top = 1
            labels = [[i, '{:.3f}'.format(i)] for i in numpy.arange(0.0, top, top/100.0)]
            graphVscale = urwid.AttrWrap(urwid.GraphVScale(labels=labels, top=top), 'graph_label')
            graphColumns.contents[0] = (urwid.Padding(graphVscale, left=3, right=0), (urwid.GIVEN, 10, False))
            graphColumns.contents[2] = (urwid.Padding(graphVscale, left=1, right=2), (urwid.GIVEN, 10, False))


    except urwid.ExitMainLoop:
        pass
    finally:
        loop.stop()

