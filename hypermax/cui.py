import urwid

from pprint import pformat
import numpy
import sys

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
    ]

    def onRunAlgorithmClicked(widget):
        pass

    def onExitClicked(widget):
        raise urwid.ExitMainLoop()

    def makeMountedFrame(widget, header):
        body = urwid.Frame(urwid.AttrWrap(widget, 'frame_body'), urwid.AttrWrap(urwid.Text('  ' + header), 'frame_header'))
        shadow = urwid.Columns([body, ('fixed', 2, urwid.AttrWrap(urwid.Filler(urwid.Text(('background', '  ')), "top"), 'shadow'))])
        shadow = urwid.Frame(shadow, footer=urwid.AttrWrap(urwid.Text(('background', '  ')), 'shadow'))
        padding = urwid.AttrWrap(urwid.Padding(urwid.Filler(shadow, height=('relative', 100), top=1, bottom=1), left=1, right=1), 'background')
        return padding

    graph = None
    graphVscale = None
    graphColumns = None
    status = None
    def makeMainMenu():
        content = [
            urwid.AttrWrap(urwid.Button('Run Algorithm', on_press=onRunAlgorithmClicked), 'body', focus_attr='focus'),
            urwid.AttrWrap(urwid.Button('Dump Results'), 'body', focus_attr='focus'),
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

    trialsList = urwid.SimpleFocusListWalker([])
    def makeTrialsView():
        listbox = urwid.ListBox(trialsList)
        return makeMountedFrame(listbox, 'Trials')

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

    def unhandled(key):
        if key == 'f8':
            raise urwid.ExitMainLoop()

    loop = urwid.MainLoop(background, palette, screen, unhandled_input=unhandled)

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

            trialsToAdd = len(optimizer.results) - len(trialsList)
            for result in optimizer.results[-trialsToAdd:]:
                column = []
                for key in result.keys():
                    column.append(urwid.AttrWrap(urwid.Padding(urwid.Text(str(result[key]))), 'body'))
                trialsList.append(urwid.Columns(column))

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

