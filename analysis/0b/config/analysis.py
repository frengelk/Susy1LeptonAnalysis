########################################################################################
# Initializes which processes, datasets and general information update the data        #
########################################################################################
import copy
import os

# base config
import config.Run2_pp_13TeV_2016 as run_2016
import config.Run2_pp_13TeV_2017 as run_2017
import config.Run2_pp_13TeV_2018 as run_2018
import order as od
from config.datasets_2016 import setup_datasets
from config.datasets_2017 import setup_datasets
import scinum as sn
import numpy as np
import six
from config.processes import setup_processes

# categories and variables, more setup
from .categories import setup_categories
from .variables import setup_variables

# create the analysis
analysis = od.Analysis("mj", 1)
config_2016 = analysis.add_config(run_2016.base_config.copy())
config_2017 = analysis.add_config(run_2017.base_config.copy())
config_2018 = analysis.add_config(run_2018.base_config.copy())
for year, cfg in ("2016", config_2016), ("2017", config_2017), ("2018", config_2018):
    # with od.uniqueness_context(cfg.campaign.name):
    # campaign = cfg.campaign
    # if year == "2016":
    #     setup_datasets(cfg, campaign=campaign)
    # else:
    #     setup_datasets(cfg, campaign=campaign)
    setup_processes(cfg)
    # maybe do that later again
    # for dat in cfg.datasets:
    #    dat.add_process(cfg.get_process(dat.name))
    setup_variables(cfg)
    setup_categories(cfg)
    #  mge [TeV] 1.5 1.5 1.6 1.7 1.8 1.9 1.9 1.9 2.2 2.2
    # mχe0 [TeV] 1.0 1.2 1.1 1.2 1.3 0.1 0.8 1.0 0.1 0.8
    cfg.set_aux(
        "signal_masspoints",
        [(1500, 1000), (1500, 1200), (1600, 1100), (1700, 1200), (1800, 1300), (1900, 100), (1900, 800), (1900, 1000), (2200, 100), (2200, 800)],
    )
    cfg.set_aux(
        "signal_process",
        "T5qqqqVV",  # found like this in sample names
    )

    cfg.set_aux("job_dict", os.path.expandvars("$ANALYSIS_BASE/config/datasets_2017.json"))
    cfg.set_aux("masspoints", os.path.expandvars("$ANALYSIS_BASE/config/masspoints_2017.json"))
    cfg.set_aux("all_btag_SF", os.path.expandvars("/all_btagSF_T5qqqqWW.npy"))
    cfg.set_aux("DNN_model", os.path.expandvars("$ANALYSIS_BASE/config/DNN_model.pt"))

    cfg.set_aux("channels", {"N0b": ["Muon", "Electron"], "SR0b": ["LeptonIncl"], "SR0b_backup": ["LeptonIncl"], "Anti_cuts": ["LeptonIncl"], "SB_cuts": ["LeptonIncl"], "SR_Anti": ["LeptonIncl"], "All_Lep": ["LeptonIncl"], "All_Signal": ["LeptonIncl"]})
    cfg.set_aux("DNNId", [-1, 1])  # "SR0b": ["Muon", "Electron"],

    # for these, it's enough if we redo the event weights, don't need to process everything again
    # change PileUp to UP so that we don't mess up shifts while using string matching
    cfg.set_aux("systematic_shifts", ["MuonMediumIsoSfDown", "MuonMediumIsoSfUp", "MuonMediumSfDown", "MuonMediumSfUp", "MuonTriggerSfDown", "MuonTriggerSfUp", "PileUPWeightDown", "PileUPWeightUp", "PreFireWeightDown", "PreFireWeightUp", "ElectronTightSfDown", "ElectronTightSfUp", "ElectronRecoSfDown", "ElectronRecoSfUp", "JetDeepJetMediumSfUp", "JetDeepJetMediumSfDown"])
    # since signal is fastsim, there are additional SF to make the events look like fullsim signal
    cfg.set_aux("systematic_shifts_signal", ["MuonMediumFastSfDown", "MuonMediumFastSfUp", "ElectronTightFastSfDown", "ElectronTightFastSfUp", "ElectronTightMVAFastSfDown", "ElectronTightMVAFastSfUp"])  # 'MuonLooseFastSf', 'ElectronVetoFastSf', 'ElectronVetoMVAFastSf'
    cfg.set_aux("systematic_variable_shifts", ["TotalUp", "TotalDown", "JERUp", "JERDown"])
    cfg.set_aux("JEC_correlations", {"fully": ["FlavorQCD", "SubTotalPile", "SubTotalScale"], "uncorrelated": ["TimePtEta", "SubTotalPt", "SubTotalRelative"]})

    cfg.set_aux("lumi_unc_per_year", {"2016": 1.012, "2017": 1.023, "2018": 1.025})

    # signal always last category!
    # we allow QCD in processes for All_Lep to train, and in SR0b to predict bg QCD
    # not in SR_Anti to now muddle EWk
    if year == "2016":
        cfg.set_aux("signal_binning", [0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.939, 0.978, 1])  # how to bin signal region DNN output  # 0.977 :  1.0879750804559463 0.978 :  1.0775330484184822 0.979 :  0.8320608318715793
        cfg.set_aux("data", ["MET", "SingleMuon", "SingleElectron"])
        cfg.set_aux(
            "DNN_process_template",
            {
                "SR0b": {
                    "ttjets": ["SingleTop", "TTbar"],  # FIXME eject QCD!
                    # "ttbar": ["TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],  # FIXME , "QCD"
                    "QCD": ["QCD"],
                    # "BG": ["SingleTop", "TTbar", "WJets", "Rare", "DY"],
                    "T5qqqqWW": ["T5qqqqWW"],
                },
                "All_Lep": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "T5qqqqWW": ["T5qqqqWW"],
                },
                "SR_Anti": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],  # we don't want this ending up in EWk
                    "T5qqqqWW": ["T5qqqqWW_2200_100"],  # for signal plotting
                    "data": ["MET", "SingleMuon", "SingleElectron"],
                },
                "Anti_cuts": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "data": ["MET", "SingleMuon", "SingleElectron"],
                },
                "SB_cuts": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "data": ["MET", "SingleMuon", "SingleElectron"],
                },
            },
        )
    if year == "2017":
        cfg.set_aux(
            "signal_binning",
            [0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.939, 0.978, 1.0],  # newest training
            # [0, 0.5, 0.7, 0.9],
            # [0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.935, 0.97, 1]  # how to bin signal region DNN output
            # [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.915, 0.93, 0.945, 0.96, 0.975, 0.990, 1],  # how to bin signal region DNN output
            # [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.919, 0.938, 0.9570, 0.9760, 0.995, 1.0],  #   Ashrafs binning
        )
        cfg.set_aux("data", ["MET", "SingleMuon", "SingleElectron"])
        cfg.set_aux(
            "DNN_process_template",
            {
                "N1ib": {
                    "tt_1l": ["SingleTop", "TTbar"],
                    "tt_2l": ["TTbar"],
                    "W+jets": ["WJets", "DY", "rare"],
                },
                "N0b": {
                    "ttjets": ["SingleTop", "TTbar"],  # FIXME , "QCD" eject QCD!
                    # "ttbar": ["TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    # "BG": ["SingleTop", "TTbar", "WJets", "Rare", "DY"],
                    "T5qqqqWW": ["T5qqqqWW"],
                    # "T5qqqqWW": ["T5qqqqWW_1500_1000", "T5qqqqWW_1500_1200", "T5qqqqWW_1600_1100", "T5qqqqWW_1700_1200", "T5qqqqWW_1800_1300", "T5qqqqWW_1900_100", "T5qqqqWW_1900_800", "T5qqqqWW_1900_1000", "T5qqqqWW_2200_100", "T5qqqqWW_2200_800"],
                },
                "SR0b": {
                    "ttjets": ["SingleTop", "TTbar"],  # FIXME eject QCD!
                    # "ttbar": ["TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],  # FIXME , "QCD"
                    "QCD": ["QCD"],
                    # "BG": ["SingleTop", "TTbar", "WJets", "Rare", "DY"],
                    "T5qqqqWW": ["T5qqqqWW_1500_1000", "T5qqqqWW_1500_1200", "T5qqqqWW_1600_1100", "T5qqqqWW_1700_1200", "T5qqqqWW_1800_1300", "T5qqqqWW_1900_100", "T5qqqqWW_1900_800", "T5qqqqWW_1900_1000", "T5qqqqWW_2200_100", "T5qqqqWW_2200_800"],
                },
                "SR_Anti": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],  # we don't want this ending up in EWk
                    "T5qqqqWW": ["T5qqqqWW_2200_100"],  # for signal plotting
                    "data": ["MET", "SingleMuon", "SingleElectron"],
                },
                "Anti_cuts": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "data": ["MET", "SingleMuon", "SingleElectron"],
                },
                "SB_cuts": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "data": ["MET", "SingleMuon", "SingleElectron"],
                },
                "All_Lep": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "T5qqqqWW": ["T5qqqqWW_1500_1000", "T5qqqqWW_1500_1200", "T5qqqqWW_1600_1100", "T5qqqqWW_1700_1200", "T5qqqqWW_1800_1300", "T5qqqqWW_1900_100", "T5qqqqWW_1900_800", "T5qqqqWW_1900_1000", "T5qqqqWW_2200_100", "T5qqqqWW_2200_800"],
                },
                "All_Signal": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "T5qqqqWW": ["T5qqqqWW"],
                },
            },
        )
    elif year == "2018":
        cfg.set_aux("signal_binning", [0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.9335, 0.967, 1])  # how to bin signal region DNN output  # 0.966 :  1.0760171477869869 0.967 :  1.0496283811006415 0.968 :  0.9868771300518243
        cfg.set_aux("data", ["MET", "SingleMuon", "EGamma"])
        cfg.set_aux(
            "DNN_process_template",
            {
                "SR0b": {
                    "ttjets": ["SingleTop", "TTbar"],  # FIXME eject QCD!
                    # "ttbar": ["TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],  # FIXME , "QCD"
                    "QCD": ["QCD"],
                    # "BG": ["SingleTop", "TTbar", "WJets", "Rare", "DY"],
                    "T5qqqqWW": ["T5qqqqWW_1500_1000", "T5qqqqWW_2200_100", "T5qqqqWW_2200_800"],
                },
                "All_Lep": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "T5qqqqWW": ["T5qqqqWW"],
                },
                "SR_Anti": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],  # we don't want this ending up in EWk
                    "T5qqqqWW": ["T5qqqqWW_2200_100"],  # for signal plotting
                    "data": ["MET", "SingleMuon", "EGamma"],
                },
                "Anti_cuts": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "data": ["MET", "SingleMuon", "EGamma"],
                },
                "SB_cuts": {
                    "ttjets": ["SingleTop", "TTbar"],
                    "Wjets": ["WJets", "Rare", "DY"],
                    "QCD": ["QCD"],
                    "data": ["MET", "SingleMuon", "EGamma"],
                },
            },
        )


# if (era == 2016) {
# 	if (preVFP) {
# 		luminosity = 19.5; //fb
# 	} else {
# 		luminosity = 16.5; //fb
# 	}
# } else if (era == 2017) {
# 	luminosity = 41.48; //fb
# } else if (era == 2018) {
# 	luminosity = 59.83; //fb
