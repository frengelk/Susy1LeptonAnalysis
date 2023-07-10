"""
Defintion of the campaign and datasets for CMS.
"""

import order as od
import scinum as sn
from config.processes import setup_processes
from config.datasets import setup_datasets

# FIX ALL NUMBERS HERE
# campaign
campaign = od.Campaign("Run2_pp_13TeV_2017", 11, ecm=13, bx=25)
# base config
base_config = cfg = od.Config(campaign)
ch_e = cfg.add_channel("e", 1)  # , context=campaign.name)
ch_mu = cfg.add_channel("mu", 2)  # , context=campaign.name)
# FIXME BTAG WP VALUES CHECK
# store b-tagger working points
cfg.set_aux(
    "working_points",
    {
        "deepjet": {
            "loose": 0.0614,
            "medium": 0.3093,
            "tight": 0.7221,
        }
    },
)
# FIXME LUMI VALUES CHECK
cfg.set_aux("lumi", 41296.082)
