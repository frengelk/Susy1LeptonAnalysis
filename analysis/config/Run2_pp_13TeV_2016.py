"""
Defintion of the campaign and datasets for CMS.
"""
import order as od
import scinum as sn
from config.processes import setup_processes
from config.datasets import setup_datasets

# campaign
campaign = od.Campaign("Run2_pp_13TeV_2016", 1, ecm=13, bx=25)
# base config
base_config = cfg = od.Config(campaign)
ch_e = cfg.add_channel("e", 1)
ch_mu = cfg.add_channel("mu", 2)
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
cfg.set_aux("lumi", 35922.0)
# add dataset and processes
