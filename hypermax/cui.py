import urwid





def launchHypermaxUI():
    screen = urwid.raw_display.Screen()

    palette = [
        ('background', 'white', 'dark gray', 'standout'),
        ('body', 'dark gray', 'light gray', 'standout'),
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
    ]

    def onRunAlgorithmClicked(widget):
        pass

    def onExitClicked(widget):
        raise urwid.ExitMainLoop()

    content = [
        urwid.AttrWrap(urwid.Button('Run Algorithm', on_press=onRunAlgorithmClicked), 'body', focus_attr='focus'),
        urwid.AttrWrap(urwid.Button('Dump Results'), 'body', focus_attr='focus'),
        urwid.AttrWrap(urwid.Button('View Hyper Parameters'), 'body', focus_attr='focus'),
        urwid.AttrWrap(urwid.Button('Exit', on_press=onExitClicked), 'body', focus_attr='focus')
    ]

    listbox = urwid.ListBox(urwid.SimpleFocusListWalker(content))

    frame = urwid.Frame(urwid.AttrWrap(listbox, 'body'), header=urwid.Text('Hypermax v0.1'))

    background = urwid.Frame(urwid.Pile([
        urwid.AttrWrap(urwid.Padding(urwid.Filler(frame, height=('relative', 90)), left=10, right=10), 'background')
    ]))

    def unhandled(key):
        if key == 'f8':
            raise urwid.ExitMainLoop()

    loop = urwid.MainLoop(background, palette, screen, unhandled_input=unhandled)
    loop.run()

