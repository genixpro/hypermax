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
        nonlocal graph
        graph = urwid.BarGraph(attlist=['graph_bg', 'graph_bar'])
        graph.set_data([], top=1)

        labels = [[i, '{:.3f}'.format(i)] for i in numpy.arange(0.0, 1.1, 0.1)]
        graphVscale = urwid.AttrWrap(urwid.GraphVScale(labels=labels, top=1), 'graph_label')
        graphArea = urwid.Columns([(10, urwid.Padding(graphVscale, left=3, right=0)), graph])
        graphFrame = makeMountedFrame(graphArea, 'Rolling Loss')
        return graphFrame

    def makeStatusArea():
        nonlocal status
        status = urwid.Text(markup='')
        return makeMountedFrame(urwid.Filler(status), "Status")

    otherFrame = urwid.Pile([makeGraphArea(), makeStatusArea()])
    columns = urwid.Columns([makeMainMenu(), otherFrame])

    background = urwid.Frame(urwid.Pile([
        urwid.AttrWrap(columns, 'background')
    ]))

    def unhandled(key):
        if key == 'f8':
            raise urwid.ExitMainLoop()

    loop = urwid.MainLoop(background, palette, screen, unhandled_input=unhandled)

    def updateStatus(a,b):
        statusText = "Completed: " + str(optimizer.completed()) + "/" + str(optimizer.totalTrials)
        status.set_text(statusText)

        top = 1

        allResults = numpy.array([result['loss'] for result in optimizer.results])


        allResults = [numpy.mean(allResults[max(0, index-10):min(len(allResults)-1, index+10)]) for index in range(0, len(allResults), 5)]

        data = []
        for result in allResults[-min(len(allResults), 20):]:
            data.append([result])

        graph.set_data(data, 1.0)

        loop.set_alarm_in(1.0, updateStatus)

    loop.set_alarm_in(1.0, updateStatus)
    loop.run()

