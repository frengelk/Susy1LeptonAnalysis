# coding: utf-8
"""
Physics constants.
"""
import scinum as sn
import numpy as np

BR_W_HAD = sn.Number(0.6741, {"br_whad": 0.0027})
BR_W_LEP = 1 - BR_W_HAD
BR_WW_SL = 2 * BR_W_HAD.mul(BR_W_LEP, rho=-1, inplace=False)
