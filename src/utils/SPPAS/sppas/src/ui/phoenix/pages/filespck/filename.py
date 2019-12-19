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

    ui.phoenix.pages.files.filenamepy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import wx

from ...windows import sppasPanel

# ---------------------------------------------------------------------------


class sppasFilePanel(sppasPanel):
    """Create a panel to manage a file and its annotations.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2019  Brigitte Bigi

    """

    def __init__(self, parent):
        super(sppasFilePanel, self).__init__(
            parent=parent,
            name="single_file",
            style=wx.BORDER_NONE
        )
        self.SetBackgroundColour(wx.GetApp().settings.bg_color)
        self.SetForegroundColour(wx.GetApp().settings.fg_color)
        self.SetFont(wx.GetApp().settings.text_font)
        self._filename = ""

    # ------------------------------------------------------------------------

    def set_filename(self, filename):
        """Set the file to represent."""
        self._filename = filename

    # ------------------------------------------------------------------------

    def _create_content(self):
        """"""
        pass

# ---------------------------------------------------------------------------


class sppasFileToggleButton(wx.Window):
    """

    """
    def __init__(self, parent, name, height, filename):
        """

        :param parent:
        :param name:
        :param height:
        :param filename:
        """
        wx.Window.__init__(self, parent, style=wx.NO_BORDER)

        self.Bind(wx.EVT_LEFT_UP, self.process_event)
        self.Bind(wx.EVT_ENTER_WINDOW, self.process_event)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.process_event)
        self.SetMinSize((48, 48))
        self._state = False

    def GetValue(self):
        """Gets the state of the toggle button."""
        return self._state

    def process_event(self, evt):
        pass

