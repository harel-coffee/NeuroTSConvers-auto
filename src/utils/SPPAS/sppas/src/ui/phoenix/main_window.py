"""
    ..
        ---------------------------------------------------------------------
         ___   __    __    __    ___
        /     |  \  |  \  |  \  /              the automatic
        \__   |__/  |__/  |___| \__             annotation and
           \  |     |     |   |    \             analysis
        ___/  |     |     |   | ___/              of speech

        http://www.sppas.org/

        Use of this software is governed by the GNU Public License, version 3.

        SPPAS is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        SPPAS is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with SPPAS. If not, see <http://www.gnu.org/licenses/>.

        This banner notice must not be removed.

        ---------------------------------------------------------------------

    ui.phoenix.main_window.py
    ~~~~~~~~~~~~~~~~~~~~~~~~

"""
import wx
import webbrowser

from sppas.src.config import sg
from sppas.src.config import ui_translation

from .windows import sppasBitmapTextButton
from .windows import sppasTextButton
from .windows import sppasSimplebook
from .windows import sppasBitmapButton
from .windows import sppasPanel
from .windows import sppasDialog

from .pages import sppasHomePanel
from .pages import sppasFilesPanel
from .pages import sppasAnnotatePanel
from .pages import sppasAnalyzePanel
from .pages import sppasPluginsPanel

from .dialogs import YesNoQuestion
from .dialogs import About
from .dialogs import Settings
from .main_log import sppasLogWindow

# ---------------------------------------------------------------------------

MSG_ACTION_HOME = ui_translation.gettext('Home')
MSG_ACTION_FILES = ui_translation.gettext('Files')
MSG_ACTION_ANNOTATE = ui_translation.gettext('Annotate')
MSG_ACTION_ANALYZE = ui_translation.gettext('Analyze')
MSG_ACTION_PLUGINS = ui_translation.gettext('Plugins')
MSG_ACTION_EXIT = ui_translation.gettext('Exit')
MSG_ACTION_ABOUT = ui_translation.gettext('About')
MSG_ACTION_SETTINGS = ui_translation.gettext('Settings')
MSG_ACTION_VIEWLOGS = ui_translation.gettext('View logs')

MSG_TOOLTIP_WEBSITE = ui_translation.gettext('Visit the website at ')
MSG_CONFIRM = ui_translation.gettext("Confirm exit of {:s}").format(sg.__name__)

# -----------------------------------------------------------------------


class sppasMainWindow(sppasDialog):
    """Create the main frame of SPPAS.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    This class:

        - does not inherit of wx.TopLevelWindow because we need EVT_CLOSE
        - does not inherit of wx.Frame because we don't need neither a
        status bar, nor a toolbar, nor a menu.

    """

    def __init__(self):
        super(sppasMainWindow, self).__init__(
            parent=None,
            title=wx.GetApp().GetAppDisplayName(),
            style=wx.DEFAULT_FRAME_STYLE | wx.DIALOG_NO_PARENT)

        # Members
        self._init_infos()

        # Create the log window of the application and show it.
        self.log_window = sppasLogWindow(self, wx.GetApp().GetAppLogLevel())

        # Fix this frame content
        self._create_content()
        self._setup_events()

        # Fix this frame properties
        self.Enable()
        self.SetFocus()
        self.CenterOnScreen(wx.BOTH)
        self.FadeIn(deltaN=-4)
        self.Show(True)

    # ------------------------------------------------------------------------
    # Private methods to create the GUI and initialize members
    # ------------------------------------------------------------------------

    def _init_infos(self):
        """Initialize the main frame.

        Set the title, the icon and the properties of the frame.

        """
        sppasDialog._init_infos(self)

        # Fix other frame properties
        self.SetMinSize(wx.Size(640, 480))
        self.SetSize(wx.GetApp().settings.frame_size)
        self.SetName('{:s}'.format(sg.__name__))

    # -----------------------------------------------------------------------

    def _create_content(self):
        """Create the content of the frame.

        Content is made of a menu, an area for panels and action buttons.

        """
        # add a customized menu (instead of an header+toolbar)
        menus = sppasMenuPanel(self)
        self.SetHeader(menus)

        # The content of this main frame is organized in a book
        book = self._create_book()
        self.SetContent(book)

        # add some action buttons
        actions = sppasActionsPanel(self)
        self.SetActions(actions)

        # organize the content and lays out.
        self.LayoutComponents()

    # -----------------------------------------------------------------------

    def _create_book(self):
        """Create the simple book to manage the several pages of the frame.

        Names of the pages are:
        page_welcome, page_files, page_annotate, page_analyze, page_plugins

        """
        book = sppasSimplebook(
            parent=self,
            style=wx.BORDER_NONE | wx.TAB_TRAVERSAL | wx.WANTS_CHARS,
            name="content"
        )
        book.SetEffectsTimeouts(200, 200)

        # 1st page: a panel with a welcome message
        book.ShowNewPage(sppasHomePanel(book))

        # 2nd: file browser
        book.AddPage(sppasFilesPanel(book), text="")

        # 3rd: annotate automatically selected files
        book.AddPage(sppasAnnotatePanel(book), text="")

        # 4th: analyze selected files
        book.AddPage(sppasAnalyzePanel(book), text="")

        # 5th: plugins
        book.AddPage(sppasPluginsPanel(book), text="")

        return book

    # -----------------------------------------------------------------------
    # Events management
    # -----------------------------------------------------------------------

    def _setup_events(self):
        """Associate a handler function with the events.

        It means that when an event occurs then the process handler function
        will be called.

        """
        # Bind close event from the close dialog 'x' on the frame
        self.Bind(wx.EVT_CLOSE, self.on_exit)

        # Bind all events from our buttons (including 'exit')
        self.Bind(wx.EVT_BUTTON, self._process_event)

        # Capture keys
        self.Bind(wx.EVT_CHAR_HOOK, self._process_key_event)

    # -----------------------------------------------------------------------

    def _process_event(self, event):
        """Process any kind of events.

        :param event: (wx.Event)

        """
        event_obj = event.GetEventObject()
        event_name = event_obj.GetName()
        event_id = event_obj.GetId()

        wx.LogMessage("Received event id {:d} of {:s}"
                      "".format(event_id, event_name))

        if event_name == "exit":
            self.exit()

        elif event_name == "view_log":
            self.log_window.focus()

        elif event_name == "about":
            About(self)

        elif event_name == "settings":
            self.on_settings()

        elif event_name in ("home", "files", "annotate", "analyze", "plugins"):
            self.show_page("page_" + event_name)

        else:
            event.Skip()

    # -----------------------------------------------------------------------

    def _process_key_event(self, event):
        """Process a key event.

        :param event: (wx.Event)

        """
        key_code = event.GetKeyCode()

        if key_code == wx.WXK_F4 and event.AltDown():
            self.on_exit(event)
        else:
            event.Skip()

    # -----------------------------------------------------------------------
    # Callbaks to events
    # -----------------------------------------------------------------------

    def on_exit(self, event):
        """Makes sure the user was intending to exit the application.

        :param event: (wx.Event) Un-used.

        """
        response = YesNoQuestion(MSG_CONFIRM)
        if response == wx.ID_YES:
            self.exit()

    # -----------------------------------------------------------------------

    def on_settings(self):
        """Open settings dialog and apply changes."""
        response = Settings(self)
        if response == wx.ID_CANCEL:
            return

        self.UpdateUI()
        self.log_window.UpdateUI()

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def exit(self):
        """Destroy the frame, terminating the application."""
        # Stop redirecting logging to this application
        self.log_window.redirect_logging(False)
        # Terminate all frames
        self.DestroyChildren()
        self.DestroyFadeOut(deltaN=-6)

    # -----------------------------------------------------------------------

    def show_page(self, page_name):
        """Show a page of the content panel.

        If the page can't be found, the default home page is shown.

        :param page_name: (str) one of 'page_home', 'page_files', ...

        """
        # Find the page number to switch on
        book = self.FindWindow("content")
        w = book.FindWindow(page_name)
        if w is None:
            w = book.FindWindow("page_home")
        p = book.FindPage(w)
        if p == -1:
            p = 0

        # current page number
        c = book.FindPage(book.GetCurrentPage())

        # assign the effect
        if c < p:
            book.SetEffects(showEffect=wx.SHOW_EFFECT_SLIDE_TO_LEFT,
                            hideEffect=wx.SHOW_EFFECT_SLIDE_TO_LEFT)
        elif c > p:
            book.SetEffects(showEffect=wx.SHOW_EFFECT_SLIDE_TO_RIGHT,
                            hideEffect=wx.SHOW_EFFECT_SLIDE_TO_RIGHT)
        else:
            book.SetEffects(showEffect=wx.SHOW_EFFECT_NONE,
                            hideEffect=wx.SHOW_EFFECT_NONE)

        # then change to the page
        book.ChangeSelection(p)

# ---------------------------------------------------------------------------


class sppasMenuPanel(sppasPanel):
    """Create my own menu panel with several action buttons.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    sppasMenuPanel() aims to replace the commons menus+toolbar.

    """
    def __init__(self, parent):
        super(sppasMenuPanel, self).__init__(
            parent=parent,
            name="header")

        self.SetMinSize(wx.Size(-1, wx.GetApp().settings.title_height))

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        home = sppasTextButton(self, MSG_ACTION_HOME, name="home")
        sizer.Add(home, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        files = sppasTextButton(self, MSG_ACTION_FILES, name="files")
        sizer.Add(files, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        annotate = sppasTextButton(self, MSG_ACTION_ANNOTATE, name="annotate")
        sizer.Add(annotate, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        analyze = sppasTextButton(self, MSG_ACTION_ANALYZE, name="analyze")
        sizer.Add(analyze, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        plugins = sppasTextButton(self, MSG_ACTION_PLUGINS, name="plugins")
        sizer.Add(plugins, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        sizer.AddStretchSpacer(2)

        sppas_logo = sppasBitmapButton(
            parent=self,
            name="sppas_64",
            height=int(wx.GetApp().settings.title_height * 0.8)
        )
        sppas_logo.SetToolTip(MSG_TOOLTIP_WEBSITE + sg.__url__)
        sizer.Add(sppas_logo, 0,
                  wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, border=10)

        self.Bind(wx.EVT_BUTTON, self._process_event)
        self.SetSizer(sizer)

    # ------------------------------------------------------------------------
    # Callback to events
    # ------------------------------------------------------------------------

    def _process_event(self, event):
        """Process any kind of events.

        :param event: (wx.Event)

        """
        event_obj = event.GetEventObject()
        event_name = event_obj.GetName()

        if event_name == "sppas_64":
            webbrowser.open(sg.__url__)
        else:
            event.Skip()

# ---------------------------------------------------------------------------


class sppasActionsPanel(sppasPanel):
    """Create my own panel with some action buttons.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    """
    def __init__(self, parent):

        super(sppasActionsPanel, self).__init__(
            parent=parent,
            name="actions")

        settings = wx.GetApp().settings

        # Create the action panel and sizer
        self.SetMinSize(wx.Size(-1, settings.action_height))
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        exit_btn = sppasBitmapTextButton(
            self, MSG_ACTION_EXIT, name="exit")
        about_btn = sppasBitmapTextButton(
            self, MSG_ACTION_ABOUT, name="about")
        settings_btn = sppasBitmapTextButton(
            self, MSG_ACTION_SETTINGS, name="settings")
        log_btn = sppasBitmapTextButton(
            self, MSG_ACTION_VIEWLOGS, name="view_log")

        vertical_line_1 = wx.StaticLine(self, style=wx.LI_VERTICAL)
        vertical_line_2 = wx.StaticLine(self, style=wx.LI_VERTICAL)
        vertical_line_3 = wx.StaticLine(self, style=wx.LI_VERTICAL)

        sizer.Add(log_btn, 1, wx.ALL | wx.EXPAND, 0)
        sizer.Add(vertical_line_1, 0, wx.ALL | wx.EXPAND, 0)
        sizer.Add(settings_btn, 1, wx.ALL | wx.EXPAND, 0)
        sizer.Add(vertical_line_2, 0, wx.ALL | wx.EXPAND, 0)
        sizer.Add(about_btn, 1, wx.ALL | wx.EXPAND, 0)
        sizer.Add(vertical_line_3, 0, wx.ALL | wx.EXPAND, 0)
        sizer.Add(exit_btn, 4, wx.ALL | wx.EXPAND, 0)

        self.SetSizer(sizer)
