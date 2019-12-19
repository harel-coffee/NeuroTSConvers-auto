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

    src.ui.phoenix.windows.button.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import wx
import logging
from wx.lib.buttons import GenBitmapTextButton, GenButton, GenBitmapButton

from ..tools import sppasSwissKnife
from .image import ColorizeImage

# ---------------------------------------------------------------------------

DEFAULT_STYLE = wx.BORDER_NONE | wx.TAB_TRAVERSAL | wx.WANTS_CHARS

# ---------------------------------------------------------------------------


class sppasTextButton(GenButton):
    """Create a simple text button. Inherited from the wx.Button.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    """
    def __init__(self, parent, label, name):
        super(sppasTextButton, self).__init__(
           parent,
           wx.ID_ANY,
           label,
           style=DEFAULT_STYLE,
           name=name)

        self.SetInitialSize()
        self.Enable(True)
        self.SetBezelWidth(0)
        self.SetUseFocusIndicator(False)

# ---------------------------------------------------------------------------


class sppasBitmapTextButton(GenBitmapTextButton):
    """Create a simple text button.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    Create a button with bitmap and text. A tooltip can optionally be added.

    >>> button = sppasBitmapTextButton(None, "Exit", "exit")
    >>> button.SetToolTipString("Quit the application")

    """

    def __init__(self, parent, label, name, style=DEFAULT_STYLE):
        btn_height = int(parent.GetSize()[1])
        super(sppasBitmapTextButton, self).__init__(
            parent,
            id=wx.NewId(),
            bitmap=sppasSwissKnife.get_bmp_icon(name, height=btn_height),
            label=" "+label+" ",
            style=style,
            name=name
        )

        self.SetInitialSize()
        self.Enable(True)
        self.SetBezelWidth(0)
        self.SetUseFocusIndicator(False)

    # -----------------------------------------------------------------------

    def SetForegroundColour(self, colour):
        """Override. Apply fg colour to both the image and the text.

        :param colour: (wx.Colour)

        """
        current = self.GetForegroundColour()
        try:
            bmp = self.GetBitmapLabel()
            img = bmp.ConvertToImage()
            ColorizeImage(img, current, colour)
            self.SetBitmapLabel(wx.Bitmap(img))
        except:
            logging.debug('SetForegroundColour not applied to image'
                          'for button {:s}'.format(self.GetName()))

        GenBitmapTextButton.SetForegroundColour(self, colour)

# ---------------------------------------------------------------------------


class sppasBitmapButton(GenBitmapButton):
    """Create a simple bitmap button.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    Create a button with bitmap. A tooltip can optionally be added.

    >>> button = sppasBitmapButton(None, "exit")
    >>> button.SetToolTipString("Quit the application")

    """

    def __init__(self, parent, name, style=DEFAULT_STYLE, height=None):

        if height is None:
            height = int(parent.GetSize()[1])
        super(sppasBitmapButton, self).__init__(
            parent,
            id=wx.NewId(),
            bitmap=sppasSwissKnife.get_bmp_icon(name, height=height),
            style=style,
            name=name
        )

        self.SetInitialSize()
        self.Enable(True)
        self.SetBezelWidth(0)
        self.SetUseFocusIndicator(False)

    # -----------------------------------------------------------------------

    def SetForegroundColour(self, colour):
        """Override. Apply fg colour to the image.

        :param colour: (wx.Colour)

        """
        try:
            bmp = self.GetBitmapLabel()
            img = bmp.ConvertToImage()
            current = self.GetForegroundColour()
            ColorizeImage(img, current, colour)
            self.SetBitmapLabel(wx.Bitmap(img))
        except:
            logging.debug('SetForegroundColour not applied to image'
                          'for button {:s}'.format(self.GetName()))

        GenBitmapButton.SetForegroundColour(self, colour)
