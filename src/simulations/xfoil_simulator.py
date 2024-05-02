"""
code for generating airfoil designs using online airfoil datasets
code is inspired by:
The Department of Energy [https://catalog.data.gov/dataset/airfoil-computational-fluid-dynamics-2k-shapes-25-aoas-3-re-numbers]
"""

# standard library imports
import math
import os
import pickle
import platform
import subprocess
import sys

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

# local imports


class XfoilSimulator:
    MAC_PLATFORM = 1
    LINUX_PLATFORM = 2
    UNKNOWN_PLATFORM = 3
    PLATFORM = UNKNOWN_PLATFORM

    VERBOSE = False

    def __init__(
        self,
        reynolds_number=1e6,
        mach_number=0.3,
        s_aot=2.0,
        t_aot=6.0,
        aot_increments=1.0,
    ):
        self.reynolds_number = reynolds_number
        self.mach_number = mach_number
        self.s_aot = s_aot
        self.t_aot = t_aot
        self.aot_increments = aot_increments

        os_type = platform.system()
        if os_type == "Darwin":
            self.PLATFORM = self.MAC_PLATFORM
        elif os_type == "Linux":
            self.PLATFORM = self.LINUX_PLATFORM
        else:
            self.PLATFORM = self.UNKNOWN_PLATFORM


"""
x/c: The x-coordinate normalized by the chord length.
y/c: The y-coordinate normalized by the chord length.
Ue/Vinf: Edge velocity divided by the free stream velocity.
Dstar: Displacement thickness.
Theta: Momentum thickness.
Cf: Skin friction coefficient.
H: Shape factor.
"""
