import wx
import sys
from os import path
PROGRAM = path.abspath(__file__)
SPPAS = path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(PROGRAM)))))
sys.path.append(SPPAS)

from sppas.src.ui.phoenix import sppasApp
from sppas.src.ui.phoenix.windows.button import sppasTextButton
from sppas.src.ui.phoenix.windows.button import sppasBitmapTextButton


class testButtonsFrame(wx.Frame):
    def __init__(self):
        super(testButtonsFrame, self).__init__(
            parent=None,
            title="Frame to test buttons",
            style=wx.DEFAULT_FRAME_STYLE | wx.CLOSE_BOX)
        self.Show()

        # create a sizer to add and organize objects
        btn1 = sppasTextButton(self, "Simple text button", "textbtn")
        btn1.SetPosition((25, 25))

        btn2 = sppasBitmapTextButton(self, "Bitmap button", "exit")
        btn2.SetPosition((25, 125))


if __name__ == '__main__':

    app = sppasApp()
    app.background_initialization()
    frame = testButtonsFrame()
    app.SetTopWindow(frame)
    app.MainLoop()
