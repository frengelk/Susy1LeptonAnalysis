########################################################################################
# Initializes which processes, datasets and general information update the data        #
########################################################################################
import copy
import os

# base config
import config.Run2_pp_13TeV_2016 as run_2016
import config.Run2_pp_13TeV_2017 as run_2017
import order as od
from config.datasets_2016 import setup_datasets
from config.datasets_2017 import setup_datasets
import scinum as sn
import six
from config.processes import setup_processes

# categories and variables, more setup
from .categories import setup_categories
from .variables import setup_variables

# create the analysis
analysis = od.Analysis("mj", 1)
config_2016 = analysis.add_config(run_2016.base_config.copy())
config_2017 = analysis.add_config(run_2017.base_config.copy())
for year, cfg in ("2016", config_2016), ("2017", config_2017):
    # with od.uniqueness_context(cfg.campaign.name):
    campaign = cfg.campaign
    if year == "2016":
        setup_datasets(cfg, campaign=campaign)
    else:
        setup_datasets(cfg, campaign=campaign)
    setup_processes(cfg)
    # maybe do that later again
    # for dat in cfg.datasets:
    #    dat.add_process(cfg.get_process(dat.name))
    setup_variables(cfg)
    setup_categories(cfg)
    cfg.set_aux(
        "signal_masspoints",
        [(2200, 800), (2200, 100), (1500, 1000), (1800, 1300)],
    )
    cfg.set_aux(
        "signal_process",
        "T5qqqqVV",  # found like this in sample names
    )

    cfg.set_aux("job_dict", os.path.expandvars("$ANALYSIS_BASE/config/datasets_2017.json"))
    cfg.set_aux("DNN_model", os.path.expandvars("$ANALYSIS_BASE/config/DNN_model.pt"))

    cfg.set_aux("data", ["MET", "SingleMuon", "SingleElectron"])
    cfg.set_aux("channels", ["Muon", "Electron"])

    cfg.set_aux(
        "DNN_process_template",
        {
            "N1ib": {
                "tt_1l": ["SingleTop", "TTbar"],
                "tt_2l": ["TTbar"],
                "W+jets": ["WJets", "DY", "rare"],
            },
            "N0b": {
                # "tt+jets": ["TTbar"],
                # "SingleTop": ["SingleTop"],
                # "W+jets": ["WJets", "Rare"],
                # "DY": ["DY"],
                # "T5qqqqWW": ["T5qqqqWW_2200_100", "T5qqqqWW_1500_1000"],
                "tt+jets": ["SingleTop", "TTbar"],
                "W+jets": ["WJets", "Rare", "DY"],
                "T5qqqqWW": ["T5qqqqWW"],
            },
        },
    )
