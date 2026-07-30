# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Marla Metternich
"""

# Packages importation
from os import listdir, path
from os.path import isfile, join
import numpy as np

script_tessellate = '1-Tessellation/main_tessellate.py'
script_optimize   = '2-Persistence-analysis/optimize.py'    
# wouldn't work yet because they both require input changes