# coding: utf-8
"""
Physics processes.
If not stated otherwise, cross sections are given in pb.
The cross sections are rather old, so we need to check them before use
However, we use the xsec from metadata right now

declare processes by overall process
 --binned processes according to files

each process needs an unique number, (label) and a xsec (defined on the fileset)
xsec +- uncert is done via scinum.Number(xsec, unc)
"""
import order as od
import scinum as sn
from math import inf
from config.constants import *


def setup_processes(cfg):
    ### signal ###
    #  mge [TeV] 1.5 1.5 1.6 1.7 1.8 1.9 1.9 1.9 2.2 2.2
    # mχe0 [TeV] 1.0 1.2 1.1 1.2 1.3 0.1 0.8 1.0 0.1 0.8
    cfg.add_process(
        "T5qqqqWW_1500_1000",
        15001000,
        label=r"T5qqqqWW (1500, 1000)",
        label_short="T5",
        color="#FFD700",  # (Gold)
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1500_1000",
                1510,
                label=r"T5qqqqWW (1500, 1000)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1500, 1000), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_1500_1200",
        15001200,
        label=r"T5qqqqWW (1500, 1200)",
        label_short="T5",
        color="#9370DB",  # (Medium Purple)
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1500_1200",
                1512,
                label=r"T5qqqqWW (1500, 1200)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1500, 1200), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_1600_1100",
        16001100,
        label=r"T5qqqqWW (1600, 1100)",
        label_short="T5",
        color="#00FF7F",  # (Spring Green)
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1600_1100",
                1611,
                label=r"T5qqqqWW (1600, 1100)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1600, 1100), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_1700_1200",
        17001200,
        label=r"T5qqqqWW (1700, 1200)",
        label_short="T5",
        color="#FF4500",  # (Orange Red)
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1700_1200",
                1712,
                label=r"T5qqqqWW (1700, 1200)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1700, 1200), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_1800_1300",
        18001300,
        label=r"T5qqqqWW (1800, 1300)",
        label_short="T5",
        color="#FF1493",  # Deep Pink
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1800_1300",
                221,
                label=r"T5qqqqWW (1800, 1300)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1800, 1300), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_1900_100",
        1900100,
        label=r"T5qqqqWW (1900, 100)",
        label_short="T5",
        color="#6A5ACD",  # (Slate Blue)
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1900_100",
                191,
                label=r"T5qqqqWW (1900, 100)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1900, 100), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_1900_800",
        1900800,
        label=r"T5qqqqWW (1900, 800)",
        label_short="T5",
        color="#8A2BE2",  # (Blue Violet)
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1900_800",
                1510,
                label=r"T5qqqqWW (1900, 800)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1900, 800), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_1900_1000",
        19001000,
        label=r"T5qqqqWW (1900, 1000)",
        label_short="T5",
        color="#FFA07A",  # (Light Salmon)
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_1900_1000",
                1910,
                label=r"T5qqqqWW (1900, 1000)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                aux={"isData": False, "isSignal": True, "masspoint": (1900, 1000), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_2200_100",
        2200100,
        label=r"T5qqqqWW (2200, 100)",
        label_short="T5",
        color="#FF1493",  # Deep Pink
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_2200_100",
                221,
                label=r"T5qqqqWW (2200, 100)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                color="#FF1493",  # Deep Pink
                aux={"isData": False, "isSignal": True, "masspoint": (2200, 100), "histtype": "step", "isSignal": True},
            )
        ],
    )

    cfg.add_process(
        "T5qqqqWW_2200_800",
        2200800,
        label=r"T5qqqqWW (2200, 800)",
        label_short="T5",
        color=(100, 0, 0),  # Deep Pink
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8_2200_800",
                221,
                label=r"T5qqqqWW (2200, 800)",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                color="#FF1493",  # Deep Pink
                aux={"isData": False, "isSignal": True, "masspoint": (2200, 800), "histtype": "step", "isSignal": True},
            )
        ],
    )

    # all signal placeholder
    cfg.add_process(
        "T5qqqqWW",
        1,
        label=r"T5qqqqWW",
        label_short="T5",
        color=(0, 0, 0),
        xsecs={
            13: sn.Number(0.385, 0.385 * 0.116),  # placeholder
        },
        aux={"isData": False, "histtype": "step", "isSignal": True},
        processes=[
            od.Process(
                "SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8",
                103,
                label=r"T5qqqqWW",
                xsecs={
                    13: sn.Number(1.0),  # placeholder
                },
                label_short="T5",
                color=(0, 0, 0),
                aux={"isData": False, "isSignal": True, "histtype": "fill"},
            )
        ],
    )

    cfg.add_process("T1tttt", 2, label=r"T1tttt", label_short="T1", color=(150, 150, 150), xsecs={13: sn.Number(0.1)})  # FIXME,

    #### MC ####
    cfg.add_process(
        "TTbar",
        100,
        label=r"$t \bar{t}$",
        label_short="TT",
        color="#00BFFF",  # Deep Sky Blue"#ADD8E6",  #(0, 0, 255),
        aux={"isData": False, "histtype": "fill", "isSignal": False},
        processes=[
            # the not HT binned are not exactly included in XSDB as of now
            od.Process(
                "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
                102,
                label=r"TTJets sl tbar",
                xsecs={
                    13: sn.Number(687.1, 0.5174),  # old was 365.34 ?
                },
            ),
            od.Process(
                "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                103,
                label=r"TTJets dl",
                xsecs={
                    13: sn.Number(687.1, 0.5174),  # 88.29
                },
            ),
            od.Process(
                "TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8",
                104,
                label=r"TTJets HT 600-800",
                xsecs={
                    13: sn.Number(1.402, 0.01244),
                },
            ),
            od.Process(
                "TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8",
                105,
                label=r"TTJets HT 800-1200",
                xsecs={
                    13: sn.Number(0.5581, 0.00501),
                },
            ),
            od.Process(
                "TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8",
                106,
                label=r"TTJets HT 1200-2500",
                xsecs={
                    13: sn.Number(0.09876, 0.0008709),
                },
            ),
            od.Process(
                "TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                107,
                label=r"TTJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.001124, 9.277e-06),
                },
            ),
        ],
    )

    cfg.add_process(
        "QCD",
        200,
        label=r"QCD Multijet",
        label_short="QCD",
        color="#9400D3",  # Dark Violet"#E0FFFF",  #(139, 28, 98),
        aux={"isData": False, "histtype": "fill", "isSignal": False},
        processes=[
            od.Process(
                "QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8",
                209,
                label=r"QCD HT 50-100",
                xsecs={
                    13: sn.Number(187700000.0, 1639000.0),
                },
            ),
            od.Process(
                "QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8",
                201,
                label=r"QCD HT 100-200",
                xsecs={
                    13: sn.Number(23500000.0, 207400.0),
                },
            ),
            od.Process(
                "QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8",
                202,
                label=r"QCD HT 200-300",
                xsecs={
                    13: sn.Number(1552000.0, 14450.0),
                },
            ),
            od.Process(
                "QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8",
                203,
                label=r"QCD HT 300-500",
                xsecs={
                    13: sn.Number(321100.0, 2968.0),
                },
            ),
            od.Process(
                "QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8",
                204,
                label=r"QCD HT 500-700",
                xsecs={
                    13: sn.Number(30250.0, 284.0),
                },
            ),
            od.Process(
                "QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8",
                205,
                label=r"QCD HT 700-1000",
                xsecs={13: sn.Number(6398.0, 59.32)},
            ),
            od.Process(
                "QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8",
                206,
                label=r"QCD HT 1000-1500",
                xsecs={
                    13: sn.Number(1122.0, 10.41),
                },
            ),
            od.Process(
                "QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8",
                207,
                label=r"QCD HT 1500-2000",
                xsecs={13: sn.Number(109.4, 1.006)},
            ),
            od.Process(
                "QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                208,
                label=r"QCD HT 2000-Inf",
                xsecs={
                    13: sn.Number(21.74, 0.2019),
                },
            ),
        ],
    )

    cfg.add_process(
        "WJets",
        300,
        label=r"$W+Jets \rightarrow l \nu$",
        label_short="W+JEts",
        color="#FFD700",  # Gold"#B0C4DE",  #(255, 165, 0),
        aux={"isData": False, "histtype": "fill", "isSignal": False},
        processes=[
            od.Process(
                "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
                301,
                label=r"WJets HT 70-100",
                xsecs={
                    13: sn.Number(1283.0, 11.53),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
                302,
                label=r"WJets HT 100-200",
                xsecs={
                    13: sn.Number(1244.0, 11.66),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
                303,
                label=r"WJets HT 200-400",
                xsecs={
                    13: sn.Number(337.8, 3.128),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
                304,
                label=r"WJets HT 400-600",
                xsecs={
                    13: sn.Number(44.93, 0.4146),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
                305,
                label=r"WJets HT 600-800",
                xsecs={13: sn.Number(11.19, 0.105)},
            ),
            od.Process(
                "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
                306,
                label=r"WJets HT 800-1200",
                xsecs={
                    13: sn.Number(4.926, 0.04705),
                },
            ),
            od.Process(
                "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
                307,
                label=r"WJets HT 1200-2500",
                xsecs={13: sn.Number(1.152, 0.01076)},
            ),
            od.Process(
                "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                308,
                label=r"WJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.02646, 0.0002507),
                },
            ),
        ],
    )

    cfg.add_process(
        "DY",
        400,
        label=r"$DY \rightarrow l l$",
        label_short="DY",
        color="#FF8C00",  # Dark Orange"#D8BFD8",  #(210, 105, 30),
        aux={"isData": False, "histtype": "fill", "isSignal": False},
        processes=[
            od.Process(
                "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                401,
                label=r"DY HT 70-100",
                xsecs={
                    13: sn.Number(140.0, 1.255),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                402,
                label=r"DY HT 100-200",
                xsecs={
                    13: sn.Number(139.2, 1.249),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                403,
                label=r"DY HT 200-400",
                xsecs={
                    13: sn.Number(38.4, 0.3494),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                404,
                label=r"DY HT 400-600",
                xsecs={
                    13: sn.Number(5.174, 0.04871),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                405,
                label=r"DY HT 600-800",
                xsecs={13: sn.Number(1.258, 0.01194)},
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                406,
                label=r"DY HT 800-1200",
                xsecs={
                    13: sn.Number(0.5598, 0.005237),
                },
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                407,
                label=r"DY HT 1200-2500",
                xsecs={13: sn.Number(0.1305, 0.001241)},
            ),
            od.Process(
                "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
                408,
                label=r"DY HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.002997, 2.837e-05),
                },
            ),
        ],
    )
    cfg.add_process(
        "SingleTop",
        500,
        label=r"Single top",
        label_short="st",
        color="#B22222",  # Firebrick"#E6E6FA",  #(255, 0, 0),
        aux={"isData": False, "histtype": "fill", "isSignal": False},
        processes=[
            od.Process(
                "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
                501,
                label=r"st tW top",
                xsecs={
                    13: sn.Number(32.45, 0.02338),
                },
            ),
            od.Process(
                "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
                502,
                label=r"st tW antitop",
                xsecs={
                    13: sn.Number(32.51, 0.02346),
                },
            ),
            od.Process(
                "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
                503,
                label=r"s antitop no fh",
                xsecs={
                    13: sn.Number(32.51, 0.02346),
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
                xsecs={13: sn.Number(3.549, 0.003412)},
            ),
            od.Process(
                "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                506,
                label=r"st t_ch incl",
                xsecs={
                    13: sn.Number(113.4, 0.8831),
                },
            ),
            od.Process(
                "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                507,
                label=r"santit t_ch incl",
                xsecs={13: sn.Number(67.93, 0.4479)},
            ),
        ],
    )

    cfg.add_process(
        "Rare",
        600,
        label=r"Rare Processes",
        label_short="rare",
        color="#3CB371",  # Medium Sea Green"#FFFFE0",  #(0, 255, 0),
        aux={"isData": False, "histtype": "fill", "isSignal": False},
        processes=[
            od.Process(
                # TTZ -> ll missing?
                "TTZToNuNu_TuneCP5_13TeV-amcatnlo-pythia8",
                601,
                label=r"TTZ ll nu nu",
                xsecs={
                    13: sn.Number(0.1476, 0.0001971),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            # deleted?
            od.Process(
                "TTZToQQ_TuneCP5_13TeV_amcatnlo-pythia8",
                602,
                label=r"TTZ qq",
                xsecs={
                    13: sn.Number(0.5113, 0.001),  # placeholder unc
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
                603,
                label=r"TTW+jets l nu",
                xsecs={
                    13: sn.Number(0.2163, 0.002284),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
                604,
                label=r"TTW+jets qq",
                xsecs={
                    13: sn.Number(0.4432, 0.004706),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                605,
                label=r"WW ll nu nu",
                xsecs={13: sn.Number(11.09, 0.00704)},
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
                xsecs={13: sn.Number(9.119, 0.09682)},
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
                    13: sn.Number(0.9738, 0.0009971),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
            od.Process(
                "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                612,
                label=r"ZZ ll qq",
                xsecs={13: sn.Number(3.676, 0.03147)},
                aux={"isData": False, "histtype": "fill"},
            ),
            # only UL16 xsec given
            od.Process(
                "tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8",
                613,
                label=r"tZq ll 4f",
                xsecs={
                    13: sn.Number(0.07561, 0.0002088),
                },
                aux={"isData": False, "histtype": "fill"},
            ),
        ],
    )

    # extra colors
    # "#FFE4B5",  # Moccasin
    # "#FFFACD"   # Lemon Chiffon
    # "#FF1493",  # Deep Pink
    # "#000080"   # Navy

    # write data, no crosssection, is_data flag instead

    cfg.add_process(
        "SingleElectron",  # "data_electron",
        700,
        label=r"data electron",
        label_short="dat ele",
        color=(0, 0, 0),
        aux={"isData": True, "histtype": "errorbar", "isSignal": False},
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
        aux={"isData": True, "histtype": "errorbar", "isSignal": False},
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
        aux={"isData": True, "histtype": "errorbar", "isSignal": False},
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
        aux={"isData": True, "histtype": "errorbar", "isSignal": False},
    )
