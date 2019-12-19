# -*- coding: UTF-8 -*-
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

    src.ui.phoenix.dialogs.messages.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import wx

from sppas.src.config import ui_translation

from ..windows import sppasMessageText
from ..windows import sppasPanel
from ..windows import sppasDialog

# ----------------------------------------------------------------------------

MSG_HEADER_ERROR = ui_translation.gettext("Error")
MSG_HEADER_WARNING = ui_translation.gettext("Warning")
MSG_HEADER_QUESTION = ui_translation.gettext("Question")
MSG_HEADER_INFO = ui_translation.gettext("Information")

# ----------------------------------------------------------------------------


class sppasBaseMessageDialog(sppasDialog):
    """Base class to create message dialogs.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    """
    def __init__(self, parent, message, style=wx.ICON_INFORMATION):
        """Create a dialog with a message.

        :param parent: (wx.Window)
        :param message: (str) the file to display in this frame.
        :param style: ONE of wx.ICON_INFORMATION, wx.ICON_ERROR, wx.ICON_EXCLAMATION, wx.YES_NO

        """
        super(sppasBaseMessageDialog, self).__init__(
            parent=parent,
            title="Message",
            style=wx.DEFAULT_FRAME_STYLE | wx.DIALOG_NO_PARENT)

        self._create_content(style, message)
        self._create_buttons()

        # Fix frame properties
        self.SetMinSize((320, 200))
        w = int(wx.GetApp().settings.frame_size[0] * 0.3)
        h = int(wx.GetApp().settings.frame_size[1] * 0.3)
        self.SetSize(wx.Size(w, h))

        self.LayoutComponents()
        self.CenterOnParent()
        self.FadeIn(deltaN=-10)

    # -----------------------------------------------------------------------

    def _create_content(self, style, message):
        """Create the content of the message dialog."""
        # Create the header
        if style == wx.ICON_ERROR:
            self.CreateHeader(MSG_HEADER_ERROR, icon_name="error")
        elif style == wx.ICON_WARNING:
            self.CreateHeader(MSG_HEADER_WARNING, icon_name="warning")
        elif style == wx.YES_NO:
            self.CreateHeader(MSG_HEADER_QUESTION, icon_name="question")
        else:
            self.CreateHeader(MSG_HEADER_INFO, icon_name="information")

        # Create the message content
        p = sppasPanel(self)
        s = wx.BoxSizer(wx.HORIZONTAL)
        txt = sppasMessageText(p, message)
        s.Add(txt, 1, wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 10)
        p.SetSizer(s)
        p.SetName("content")

    # -----------------------------------------------------------------------

    def _create_buttons(self):
        """Override to create the buttons and bind events."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Message dialogs
# ---------------------------------------------------------------------------


class sppasYesNoDialog(sppasBaseMessageDialog):
    """Create a message in a wx.Dialog with a yes-no question.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    wx.ID_YES or wx.ID_NO is returned if a button is clicked.
    wx.ID_CANCEL is returned if the dialog is destroyed.

    >>> dialog = sppasYesNoDialog("Confirm exit...")
    >>> response = dialog.ShowModal()
    >>> dialog.Destroy()
    >>> if response == wx.ID_YES:
    >>>     # do something here

    """

    def __init__(self, message):
        super(sppasYesNoDialog, self).__init__(
            parent=None,
            message=message,
            style=wx.YES_NO)

    # -----------------------------------------------------------------------

    def _create_buttons(self):
        self.CreateActions([wx.ID_NO, wx.ID_YES])
        self.Bind(wx.EVT_BUTTON, self._process_event)
        self.SetAffirmativeId(wx.ID_YES)

    # -----------------------------------------------------------------------

    def _process_event(self, event):
        """Process any kind of events.

        :param event: (wx.Event)

        """
        event_obj = event.GetEventObject()
        event_id = event_obj.GetId()
        if event_id == wx.ID_NO:
            self.SetReturnCode(wx.ID_NO)
            self.Close()
        else:
            event.Skip()

# ---------------------------------------------------------------------------


class sppasInformationDialog(sppasBaseMessageDialog):
    """Create a message in a wx.Dialog with an information.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    wx.ID_OK is returned if the button is clicked.
    wx.ID_CANCEL is returned if the dialog is destroyed.

    >>> dialog = sppasInformationDialog("you are here")
    >>> dialog.ShowModal()
    >>> dialog.Destroy()

    """

    def __init__(self, message):
        super(sppasInformationDialog, self).__init__(
            parent=None,
            message=message,
            style=wx.ICON_INFORMATION)

    # -----------------------------------------------------------------------

    def _create_buttons(self):
        self.CreateActions([wx.ID_OK])
        self.SetAffirmativeId(wx.ID_OK)

# ---------------------------------------------------------------------------
# Ready-to-use functions to display messages
# ---------------------------------------------------------------------------


def YesNoQuestion(message):
    """Display a yes-no question.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    :param message: (str) The question to ask
    :returns: the response

    wx.ID_YES or wx.ID_NO is returned if a button is clicked.
    wx.ID_CANCEL is returned if the dialog is destroyed.

    """
    dialog = sppasYesNoDialog(message)
    response = dialog.ShowModal()
    dialog.Destroy()
    return response


def Information(message):
    """Display an information.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    :param message: (str) The question to ask
    :returns: the response

    wx.ID_OK is returned if a button is clicked.
    wx.ID_CANCEL is returned if the dialog is destroyed.

    """
    dialog = sppasInformationDialog(message)
    response = dialog.ShowModal()
    dialog.Destroy()
    return response
