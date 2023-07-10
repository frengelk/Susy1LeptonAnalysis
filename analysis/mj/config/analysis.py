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
        for dat in cfg.datasets:
            dat.add_process(cfg.get_process(dat.name))
        setup_variables(cfg)
        setup_categories(cfg)
        cfg.set_aux(
            "signal_process",
            "T5qqqqWW",
        )

        cfg.set_aux("job_dict", os.path.expandvars("$ANALYSIS_BASE/config/datasets_2017.json"))
        cfg.set_aux(
            "DNN_process_template",
            {
                "tt_1l": ["st", "TTJets_sl_fromt", "TTJets_sl_fromtbar"],
                "tt_2l": ["TTJets_dilep"],
                "W+jets": ["WJets", "DY", "rare"],
            },
        )


"""

ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM
ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM

"""
