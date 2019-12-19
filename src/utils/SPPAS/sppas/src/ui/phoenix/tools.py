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

    ui.phoenix.tools.py
    ~~~~~~~~~~~~~~~~~~~

"""

import os
import wx

from sppas.src.config import paths

# -----------------------------------------------------------------------


class sppasSwissKnife:
    """Create some useful tools.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    """

    @staticmethod
    def get_bmp_icon(name, height=None):
        """Return the bitmap corresponding to the name of an icon.

        :param name: (str) Name of an icon.
        :param height: (int) Height of the bitmap. Width=height.
        :return: (wx.Bitmap)

        """
        # fix the icon file name with the current theme
        icon_name = os.path.join(
            paths.etc, "icons",
            wx.GetApp().settings.icons_theme,
            name + ".png")

        # instead, find the icon in the default set
        if os.path.exists(icon_name) is False:
            icon_name = os.path.join(
                paths.etc, "icons",
                "Refine",
                name + ".png")

        # instead, use the default icon
        if os.path.exists(icon_name) is False:
            icon_name = os.path.join(
                paths.etc, "icons",
                "Refine",
                "default.png")

        # create an image from the png file
        img = wx.Image(icon_name, wx.BITMAP_TYPE_ANY)
        if height is not None:
            img.Rescale(height, height, wx.IMAGE_QUALITY_HIGH)

        return wx.Bitmap(img)

    # ------------------------------------------------------------------------

    @staticmethod
    def get_bmp_image(name, height=None):
        """Return the bitmap corresponding to the name of an image.

        :param name: (str) Name of an image.
        :param height: (int) Height of the bitmap, Width is proportional.
        :return: (wx.Bitmap)

        """
        # fix the image file name
        img_name = os.path.join(paths.etc, "images", name + ".png")

        # instead, use the logo of SPPAS!
        if os.path.exists(img_name) is False:
            img_name = os.path.join(paths.etc, "images", "sppas.png")

        if height is not None:
            img = wx.Image(img_name, wx.BITMAP_TYPE_ANY)
            proportion = height / img.GetHeight()
            w = int(img.GetWidth() * proportion)
            img.Rescale(w, height, wx.IMAGE_QUALITY_HIGH)
            return wx.Bitmap(img)

        return wx.Bitmap(img_name, wx.BITMAP_TYPE_PNG)
