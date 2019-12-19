#!/usr/bin/env python

import sys
import os

sppasDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, sppasDir)
from sppas.src.ui.phoenix import sppasApp

# Create and run the wx application
app = sppasApp()
app.run()
sys.exit()
