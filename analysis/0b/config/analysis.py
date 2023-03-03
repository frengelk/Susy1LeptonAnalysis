########################################################################################
# Initializes which processes, datasets and general information update the data        #
########################################################################################
import copy

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

    cfg.set_aux("job_dict", "/nfs/dust/cms/user/frengelk/Code/SusyLeptonAnalysis/analysis/config/datasets_2017.json")

    cfg.set_aux(
        "proc_data_template",
        {
            "TTJets_sl_fromt": [
                "TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "TTJets_sl_fromtbar": [
                "TTJets_SingleLeptFromTbar_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "TTJets_SingleLeptFromTbar_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "TTJets_dilep": [
                "TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "TTJets_HT600to800": ["TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM"],
            "TTJets_HT800to1200": ["TTJets_HT-800to1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM"],
            "TTJets_HT1200to2500": ["TTJets_HT-1200to2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM"],
            "TTJets_HT2500toInf": [
                "TTJets_HT-1200to2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "TTJets_HT-2500toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "QCD_HT100to200": ["QCD_HT100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "QCD_HT200to300": [
                "QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "QCD_HT300to500": [
                "QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "QCD_HT500to700": [
                "QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "QCD_HT700to1000": [
                "QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "QCD_HT1000to1500": [
                "QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "QCD_HT1500to2000": [
                "QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "QCD_HT2000toInf": [
                "QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "WJets_HT70to100": ["WJetsToLNu_HT-70To100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "WJets_HT100to200": [
                "WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext2-v1_NANOAODSIM",
            ],
            "WJets_HT200to400": [
                "WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext2-v1_NANOAODSIM",
                "WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "WJets_HT400to600": [
                "WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "WJets_HT600to800": [
                "WJetsToLNu_HT-600To800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "WJetsToLNu_HT-600To800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "WJets_HT800to1200": ["WJetsToLNu_HT-800To1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM"],
            "WJets_HT1200to2500": [
                "WJetsToLNu_HT-1200To2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "WJetsToLNu_HT-1200To2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "WJets_HT2500toInf": [
                "WJetsToLNu_HT-2500ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "WJetsToLNu_HT-2500ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "DY_HT70to100": ["DYJetsToLL_M-50_HT-70to100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "DY_HT100to200": [
                "DYJetsToLL_M-50_HT-100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "DYJetsToLL_M-50_HT-100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "DY_HT200to400": [
                "DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "DY_HT400to600": [
                "DYJetsToLL_M-50_HT-400to600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "DYJetsToLL_M-50_HT-400to600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
            ],
            "DY_HT600to800": ["DYJetsToLL_M-50_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "DY_HT800to1200": [
                "DYJetsToLL_M-50_HT-800to1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
                "WJetsToLNu_HT-800To1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "DY_HT1200to2500": ["DYJetsToLL_M-50_HT-1200to2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "DY_HT2500toInf": ["DYJetsToLL_M-50_HT-2500toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            # "st_tW_top" :  ,
            # "st_tW_antitop" :  ,
            # "st_tW_antitop_no_fh" :  ,
            # "st_tW_top_no_fh" :  ,
            "st_schannel_4f": ["ST_s-channel_4f_leptonDecays_13TeV-amcatnlo-pythia8_TuneCUETP8M1_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "st_tchannel_4f_incl": ["ST_t-channel_top_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "st_tchannel_antitop_4f_incl": ["ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "TTZ_llnunu": [
                "TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext3-v1_NANOAODSIM",
                "TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext2-v1_NANOAODSIM",
            ],
            "TTZ_qq": ["TTZToQQ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "TTWjets_lnu": [
                "TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext2-v1_NANOAODSIM",
            ],
            "TTWjets_qq": ["TTWJetsToQQ_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "WW_llnunu": ["WWTo2L2Nu_13TeV-powheg_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "WW_lnuqq": [
                "WWToLNuQQ_13TeV-powheg_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7_ext1-v1_NANOAODSIM",
                "WWToLNuQQ_13TeV-powheg_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM",
            ],
            "WZ_lnuqq": ["WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "WZ_lnununu": ["WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "WZ_llqq": ["WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "ZZ_qqnunu": ["ZZTo2Q2Nu_13TeV_amcatnloFXFX_madspin_pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "ZZ_llnunu": ["ZZTo2L2Nu_13TeV_powheg_pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            "ZZ_ll_qq": ["ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM"],
            # "tZq_ll4f": [],
            "data_e_B_v1": [
                "SingleElectron_Run2016B_ver1-Nano25Oct2019_ver1-v1_NANOAOD",
                "SingleElectron_Run2016B_ver2-Nano25Oct2019_ver2-v1_NANOAOD",
            ],
            "data_e_C": ["SingleElectron_Run2016C-Nano25Oct2019-v1_NANOAOD"],
            "data_e_D": ["SingleElectron_Run2016D-Nano25Oct2019-v1_NANOAOD"],
            "data_e_E": ["SingleElectron_Run2016E-Nano25Oct2019-v1_NANOAOD"],
            "data_e_F": ["SingleElectron_Run2016F-Nano25Oct2019-v1_NANOAOD"],
            "data_e_G": ["SingleElectron_Run2016G-Nano25Oct2019-v1_NANOAOD"],
            "data_e_H": ["SingleElectron_Run2016H-Nano25Oct2019-v1_NANOAOD"],
            "data_mu_B": [
                "SingleMuon_Run2016B_ver1-Nano25Oct2019_ver1-v1_NANOAOD",
                "SingleMuon_Run2016B_ver2-Nano25Oct2019_ver2-v1_NANOAOD",
            ],
            "data_mu_C": ["SingleMuon_Run2016C-Nano25Oct2019-v1_NANOAOD"],
            "data_mu_D": ["SingleMuon_Run2016D-Nano25Oct2019-v1_NANOAOD"],
            "data_mu_E": ["SingleMuon_Run2016E-Nano25Oct2019-v1_NANOAOD"],
            "data_mu_F": ["SingleMuon_Run2016F-Nano25Oct2019-v1_NANOAOD"],
            "data_mu_G": ["SingleMuon_Run2016G-Nano25Oct2019-v1_NANOAOD"],
            "data_mu_H": ["SingleMuon_Run2016H-Nano25Oct2019-v1_NANOAOD"],
        },
    )

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
