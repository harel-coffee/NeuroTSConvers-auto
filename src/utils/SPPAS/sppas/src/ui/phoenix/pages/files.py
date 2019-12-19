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

    ui.phoenix.pages.files.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import wx

from ..windows import sppasTitleText
from ..windows import sppasMessageText
from ..windows import sppasPanel

# ---------------------------------------------------------------------------


class sppasFilesPanel(sppasPanel):
    """Create a panel to browse and select files.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    """

    def __init__(self, parent):
        super(sppasFilesPanel, self).__init__(
            parent=parent,
            name="page_files",
            style=wx.BORDER_NONE
        )
        self._create_content()

        self.SetBackgroundColour(wx.GetApp().settings.bg_color)
        self.SetForegroundColour(wx.GetApp().settings.fg_color)
        self.SetFont(wx.GetApp().settings.text_font)

    # ------------------------------------------------------------------------

    def _create_content(self):
        """"""
        # Create a title
        st = sppasTitleText(
            parent=self,
            label="Not implemented...")
        st.SetName("title")

        # Create the welcome message
        txt = sppasMessageText(
            self,
            "In future versions, this will be a file browser."
        )

        # Organize the title and message
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(st, 0, wx.TOP | wx.BOTTOM | wx.ALIGN_CENTER_HORIZONTAL, 15)
        sizer.Add(txt, 6, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)

        self.SetSizer(sizer)

    # -----------------------------------------------------------------------

    def SetFont(self, font):
        sppasPanel.SetFont(self, font)
        self.FindWindow("title").SetFont(wx.GetApp().settings.header_text_font)
        self.Layout()
