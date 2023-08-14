# coding: utf-8
"""
Physics processes.
If not stated otherwise, cross sections are given in pb.
The cross sections are rather old, so we need to check them before use
However, we use the xsec from metadata right now

declare processes by overall process
 --binned processes according to files

each process needs an unique number, (label) and a xsec (defined on the fileset)
"""
import order as od
import scinum as sn
from math import inf
from config.constants import *


def setup_processes(cfg):
    ### signal ###
    cfg.add_process(
        "T5qqqqVV",
        1,
        label=r"T5qqqqVV",
        label_short="T5",
        color=(100, 100, 100),
        xsecs={
            13: sn.Number(0.1),  # FIXME
        },
    )
    cfg.add_process("T1tttt", 2, label=r"T1tttt", label_short="T1", color=(150, 150, 150), xsecs={13: sn.Number(0.1)})  # FIXME,

    #### MC ####
    cfg.add_process(
        "TTbar",
        100,
        label=r"$t \bar{t}$",
        label_short="TT",
        color=(0, 0, 255),
        aux={"isData": False, "histtype": "fill"},
        processes=[
            od.Process(
                "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
                102,
                label=r"TTJets sl tbar",
                xsecs={
                    13: sn.Number(365.34),
                },
            ),
            od.Process(
                "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                103,
                label=r"TTJets dl",
                xsecs={
                    13: sn.Number(88.29),
                },
            ),
            od.Process(
                "TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8",
                104,
                label=r"TTJets HT 600-800",
                xsecs={
                    13: sn.Number(1.402),
                },
            ),
            od.Process(
                "TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8",
                105,
                label=r"TTJets HT 800-1200",
                xsecs={
                    13: sn.Number(0.00501),
                },
            ),
            od.Process(
                "TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8",
                106,
                label=r"TTJets HT 1200-2500",
                xsecs={
                    13: sn.Number(0.09876),
                },
            ),
            od.Process(
                "TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                107,
                label=r"TTJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.001124),
                },
            ),
        ],
    )

    cfg.add_process(
        "QCD",
        200,
        label=r"QCD Multijet",
        label_short="QCD",
        color=(139, 28, 98),
        aux={"isData": False, "histtype": "fill"},
        processes=[
            od.Process(
                "QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8",
                201,
                label=r"QCD HT 100-200",
                xsecs={
                    13: sn.Number(23500000.0),
                },
            ),
            od.Process(
                "QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8",
                202,
                label=r"QCD HT 200-300",
                xsecs={
                    13: sn.Number(1552000.0),
                },
            ),
            od.Process(
                "QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8",
                203,
                label=r"QCD HT 300-500",
                xsecs={
                    13: sn.Number(321100.0),
                },
            ),
            od.Process(
                "QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8",
                204,
                label=r"QCD HT 500-700",
                xsecs={
                    13: sn.Number(30250.0),
                },
            ),
            od.Process(
                "QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8",
                205,
                label=r"QCD HT 700-1000",
                xsecs={13: sn.Number(6398.0)},
            ),
            od.Process(
                "QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8",
                206,
                label=r"QCD HT 1000-1500",
                xsecs={
                    13: sn.Number(1122.0),
                },
            ),
            od.Process(
                "QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8",
                207,
                label=r"QCD HT 1500-2000",
                xsecs={13: sn.Number(109.4)},
            ),
            od.Process(
                "QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                208,
                label=r"QCD HT 2000-Inf",
                xsecs={
                    13: sn.Number(21.74),
                },
            ),
        ],
    )

    cfg.add_process(
        "WJets",
        300,
        label=r"$W+Jets \rightarrow l \nu$",
        label_short="W+JEts",
        color=(255, 165, 0),
        aux={"isData": False, "histtype": "fill"},
        processes=[
            od.Process(
                "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
                301,
                label=r"WJets HT 70-100",
                xsecs={
                    13: sn.Number(1283.0),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
                302,
                label=r"WJets HT 100-200",
                xsecs={
                    13: sn.Number(1244.0),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
                303,
                label=r"WJets HT 200-400",
                xsecs={
                    13: sn.Number(337.8),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
                304,
                label=r"WJets HT 400-600",
                xsecs={
                    13: sn.Number(44.93),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
                305,
                label=r"WJets HT 600-800",
                xsecs={13: sn.Number(11.19)},
            ),
            od.Process(
                "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
                306,
                label=r"WJets HT 800-1200",
                xsecs={
                    13: sn.Number(4.926),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
                307,
                label=r"WJets HT 1200-2500",
                xsecs={13: sn.Number(1.152)},
            ),
            od.Process(
                "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                308,
                label=r"WJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.02646),
                },
            ),
        ],
    )

    cfg.add_process(
        "DY",
        400,
        label=r"$DY \rightarrow l l$",
        label_short="DY",
        # color=(100, 100, 100),
        color=(210, 105, 30),
        aux={"isData": False, "histtype": "fill"},
        processes=[
            od.Process(
                "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                401,
                label=r"DY HT 70-100",
                xsecs={
                    13: sn.Number(140.0),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                402,
                label=r"DY HT 100-200",
                xsecs={
                    13: sn.Number(139.2),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                403,
                label=r"DY HT 200-400",
                xsecs={
                    13: sn.Number(38.4),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                404,
                label=r"DY HT 400-600",
                xsecs={
                    13: sn.Number(5.174),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                405,
                label=r"DY HT 600-800",
                xsecs={13: sn.Number(1.258)},
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                406,
                label=r"DY HT 800-1200",
                xsecs={
                    13: sn.Number(0.5598),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                407,
                label=r"DY HT 1200-2500",
                xsecs={13: sn.Number(0.1305)},
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                408,
                label=r"DY HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.002997),
                },
            ),
        ],
    )
    cfg.add_process(
        "SingleTop",
        500,
        label=r"Single top",
        label_short="st",
        color=(255, 0, 0),
        aux={"isData": False, "histtype": "fill"},
        processes=[
            od.Process(
                "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
                501,
                label=r"st tW top",
                xsecs={
                    13: sn.Number(32.45),
                },
            ),
            od.Process(
                "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
                502,
                label=r"st tW antitop",
                xsecs={
                    13: sn.Number(32.51),
                },
            ),
            od.Process(
                "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
                503,
                label=r"s antitop no fh",
                xsecs={
                    13: sn.Number(32.51),
                },
            ),
            # FIXME
            # od.Process(
            # "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia",
            # 504,
            # label=r"s top no fh",
            # xsecs={
            # 13: sn.Number(32.45),
            # },
            # ),
            od.Process(
                "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
                505,
                label=r"st s 4f",
                xsecs={13: sn.Number(3.549)},
            ),
            od.Process(
                "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                506,
                label=r"st t_ch incl",
                xsecs={
                    13: sn.Number(113.4),
                },
            ),
            od.Process(
                "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                507,
                label=r"santit t_ch incl",
                xsecs={13: sn.Number(67.93)},
            ),
        ],
    )

    cfg.add_process(
        "Rare",
        600,
        label=r"Rare Processes",
        label_short="rare",
        color=(0, 255, 0),
        aux={"isData": False, "histtype": "fill"},
        processes=[
            od.Process(
                # TTZ -> ll missing?
                "TTZToNuNu_TuneCP5_13TeV-amcatnlo-pythia8",
                601,
                label=r"TTZ ll nu nu",
                xsecs={
                    13: sn.Number(0.1476),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8",
                602,
                label=r"TTZ qq",
                xsecs={
                    13: sn.Number(0.5113),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
                603,
                label=r"TTW+jets l nu",
                xsecs={
                    13: sn.Number(0.2163),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
                604,
                label=r"TTW+jets qq",
                xsecs={
                    13: sn.Number(0.4432),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                605,
                label=r"WW ll nu nu",
                xsecs={13: sn.Number(11.09)},
                aux={"isData": False, "histtype": "fill"},
            ),
            # od.Process(
            # "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # 606,
            # label=r"WW l nu qq",
            # xsecs={
            # 13: sn.Number(51.65),
            # },
            # aux={"isData": False, "histtype": "fill"},
            # ),
            od.Process(
                "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                607,
                label=r"WZ l nu qq",
                xsecs={13: sn.Number(9.119)},
                aux={"isData": False, "histtype": "fill"},
            ),
            # od.Process(
            # 'WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8',
            # 608,
            # label=r"WZ l nununu",
            # xsecs={
            # 13: sn.Number(3.414),
            # },
            # aux={"isData": False, "histtype": "fill"},
            # ),
            # od.Process(
            # "WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8",
            # 609,
            # label=r"WZ ll qq",
            # xsecs={
            # 13: sn.Number(6.331),
            # },
            # aux={"isData": False, "histtype": "fill"},
            # ),
            # od.Process(
            # "ZZ_qqnunu",
            # 610,
            # label=r"ZZ qq nunu",
            # xsecs={
            # 13: sn.Number(4.033),
            # },
            # aux={"isData": False, "histtype": "fill"},
            # ),
            od.Process(
                "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8",
                611,
                label=r"ZZ ll nunu",
                xsecs={
                    13: sn.Number(0.9738),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                612,
                label=r"ZZ ll qq",
                xsecs={13: sn.Number(3.676)},
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8",
                613,
                label=r"tZq ll 4f",
                xsecs={
                    13: sn.Number(0.07561),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
        ],
    )

    # write datasets, no crosssection, is_data flag instead

    cfg.add_process(
        "SingleElectron",  # "data_electron",
        700,
        label=r"data electron",
        label_short="dat ele",
        color=(0, 0, 0),
        aux={"isData": True, "histtype": "errorbar"},
        processes=[
            od.Process("data_e_B", 702, label=r"data", is_data=True),
            # od.Process(
            # "data_e_B_v2",
            # 702,
            # label=r"data",
            # is_data=True
            # ),
            od.Process("data_e_C", 703, label=r"data", is_data=True),
            od.Process("data_e_D", 704, label=r"data", is_data=True),
            od.Process("data_e_E", 705, label=r"data", is_data=True),
            od.Process("data_e_F", 706, label=r"data", is_data=True),
            od.Process("data_e_G", 707, label=r"data", is_data=True),
            od.Process("data_e_H", 708, label=r"data", is_data=True),
        ],
    )

    cfg.add_process(
        "SingleMuon",  # "data_muon",
        800,
        label=r"data muon",
        label_short="dat mu",
        color=(0, 0, 0),
        aux={"isData": True, "histtype": "errorbar"},
        processes=[
            od.Process("data_mu_B", 802, label=r"data", is_data=True),
            # od.Process(
            # "data_e_B_v2",
            # 702,
            # label=r"data",
            # is_data=True
            # ),
            od.Process("data_mu_C", 803, label=r"data", is_data=True),
            od.Process("data_mu_D", 804, label=r"data", is_data=True),
            od.Process("data_mu_E", 805, label=r"data", is_data=True),
            od.Process("data_mu_F", 806, label=r"data", is_data=True),
            od.Process("data_mu_G", 807, label=r"data", is_data=True),
            od.Process("data_mu_H", 808, label=r"data", is_data=True),
        ],
    )
    # blank MET data
    cfg.add_process(
        "MET",
        900,
        label=r"data MET",
        label_short="dat met",
        color=(0, 0, 0),
        aux={"isData": True, "histtype": "errorbar"},
        processes=[
            od.Process("data_MET_B", 902, label=r"data", is_data=True),
            od.Process("data_MET_C", 903, label=r"data", is_data=True),
            od.Process("data_MET_D", 904, label=r"data", is_data=True),
            od.Process("data_MET_E", 905, label=r"data", is_data=True),
            od.Process("data_MET_F", 906, label=r"data", is_data=True),
            od.Process("data_MET_G", 907, label=r"data", is_data=True),
            od.Process("data_MET_H", 908, label=r"data", is_data=True),
        ],
    )
    cfg.add_process(
        "data",
        789,
        label=r"data all",
        label_short="data",
        color=(0, 0, 0),
        aux={"isData": True, "histtype": "errorbar"},
    )
