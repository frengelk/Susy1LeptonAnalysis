# coding: utf-8
"""
Defining Signal Regions
"""

import os

import luigi
import law
import order as od
import importlib
import math


signal_regions_0b = {
    "G0": [
        "n_jets ==5",
        "(LT > 250) & (LT < 350)",
        "(HT > 500) & (HT < 750)",
        "Dphi > 1",
    ],
    "G1": ["n_jets ==5", "(LT > 250) & (LT < 350)", "(HT > 750)", "Dphi > 1"],
    "G2": [
        "n_jets ==5",
        "(LT > 350) & (LT < 450)",
        "(HT > 500) & (HT < 750)",
        "Dphi > 1",
    ],
    "G3": ["n_jets ==5", "(LT > 350) & (LT < 450)", "(HT > 750)", "Dphi > 1"],
    "G4": [
        "n_jets ==5",
        "(LT > 450) & (LT < 650)",
        "(HT > 500) & (HT < 750)",
        "Dphi > 0.75",
    ],
    "G5": [
        "n_jets ==5",
        "(LT > 450) & (LT < 650)",
        "(HT > 750) & (HT < 1250)",
        "Dphi > 0.75",
    ],
    "G6": ["n_jets ==5", "(LT > 450) & (LT < 650)", "(HT > 1250)", "Dphi > 0.75"],
    "G7": ["n_jets ==5", "(LT > 650)", "(HT > 500) & (HT < 1250)", "Dphi > 0.5"],
    "G8": ["n_jets ==5", "(LT > 650)", "(HT > 1250)", "Dphi > 0.5"],
    "H1": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 250) & (LT < 350)",
        "(HT > 500) & (HT < 1000)",
        "Dphi > 1",
    ],
    "H2": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 250) & (LT < 350)",
        "(HT > 1000)",
        "Dphi > 1",
    ],
    "H3": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 350) & (LT < 450)",
        "(HT > 500) & (HT < 1000)",
        "Dphi > 1",
    ],
    "H4": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 350) & (LT < 450)",
        "(HT > 1000)",
        "Dphi > 1",
    ],
    "H5": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 450) & (LT < 650)",
        "(HT > 500) & (HT < 750)",
        "Dphi > 0.75",
    ],
    "H6": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 450) & (LT < 650)",
        "(HT > 750) & (HT < 1250)",
        "Dphi > 0.75",
    ],
    "H7": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 450) & (LT < 650)",
        "(HT > 1250)",
        "Dphi > 0.75",
    ],
    "H8": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 650)",
        "(HT > 500) & (HT < 1250)",
        "Dphi > 0.5",
    ],
    "H9": [
        "(n_jets >= 6) & (n_jets <= 7)",
        "(LT > 650)",
        "(HT > 1250)",
        "Dphi > 0.5",
    ],
    "I1": [
        "n_jets >=8",
        "(LT > 250) & (LT < 350)",
        "(HT > 500) & (HT < 1000)",
        "Dphi > 1",
    ],
    "I2": ["n_jets >=8", "(LT > 250) & (LT < 350)", "(HT > 1000)", "Dphi > 1"],
    "I3": [
        "n_jets >=8",
        "(LT > 350) & (LT < 450)",
        "(HT > 500) & (HT < 1000)",
        "Dphi > 1",
    ],
    "I4": ["n_jets >=8", "(LT > 350) & (LT < 450)", "(HT > 1000)", "Dphi > 1"],
    "I5": [
        "n_jets >=8",
        "(LT > 450) & (LT < 650)",
        "(HT > 500) & (HT < 1250)",
        "Dphi > 0.75",
    ],
    "I6": ["n_jets >=8", "(LT > 450) & (LT < 650)", "(HT > 1250)", "Dphi > 0.75"],
    "I7": ["n_jets >=8", "(LT > 650)", "(HT > 500) & (HT < 1250)", "Dphi > 0.5"],
    "I8": ["n_jets >=8", "(LT > 650)", "(HT > 1250)", "Dphi > 0.5"],
}
