# coding: utf-8
"""
Physics processes.
If not stated otherwise, cross sections are given in pb.

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
        "TTJets",
        100,
        label=r"TT+Jets",
        label_short="TTJ",
        color=(0, 0, 255),
        processes=[
            od.Process(
                "TTJets_sl_fromt",
                101,
                label=r"TTJets sl t",
                xsecs={
                    13: sn.Number(182.18),
                },
            ),
            od.Process(
                "TTJets_sl_fromtbar",
                102,
                label=r"TTJets sl tbar",
                xsecs={
                    13: sn.Number(182.18),
                },
            ),
            od.Process(
                "TTJets_dilep",
                103,
                label=r"TTJets dl",
                xsecs={
                    13: sn.Number(87.315),
                },
            ),
            od.Process(
                "TTJets_HT600to800",
                104,
                label=r"TTJets HT 600-800",
                xsecs={
                    13: sn.Number(2.76),
                },
            ),
            od.Process(
                "TTJets_HT800to1200",
                105,
                label=r"TTJets HT 800-1200",
                xsecs={
                    13: sn.Number(1.116),
                },
            ),
            od.Process(
                "TTJets_HT1200to2500",
                106,
                label=r"TTJets HT 1200-2500",
                xsecs={
                    13: sn.Number(0.198),
                },
            ),
            od.Process(
                "TTJets_HT2500toInf",
                107,
                label=r"TTJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.002),
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
        processes=[
            od.Process(
                "QCD_HT100to200",
                201,
                label=r"QCD HT 100-200",
                xsecs={
                    13: sn.Number(27990000),
                },
            ),
            od.Process(
                "QCD_HT200to300",
                202,
                label=r"QCD HT 200-300",
                xsecs={
                    13: sn.Number(1712000),
                },
            ),
            od.Process(
                "QCD_HT300to500",
                203,
                label=r"QCD HT 300-500",
                xsecs={
                    13: sn.Number(347700),
                },
            ),
            od.Process(
                "QCD_HT500to700",
                204,
                label=r"QCD HT 500-700",
                xsecs={
                    13: sn.Number(32100),
                },
            ),
            od.Process(
                "QCD_HT700to1000",
                205,
                label=r"QCD HT 700-1000",
                xsecs={13: sn.Number(6831)},
            ),
            od.Process(
                "QCD_HT1000to1500",
                206,
                label=r"QCD HT 1000-1500",
                xsecs={
                    13: sn.Number(1207),
                },
            ),
            od.Process(
                "QCD_HT1500to2000",
                207,
                label=r"QCD HT 1500-2000",
                xsecs={13: sn.Number(119.9)},
            ),
            od.Process(
                "QCD_HT2000toInf",
                208,
                label=r"QCD HT 2000-Inf",
                xsecs={
                    13: sn.Number(25.24),
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
        processes=[
            od.Process(
                "WJets_HT70to100",
                301,
                label=r"WJets HT 70-100",
                xsecs={
                    13: sn.Number(1353),
                },
            ),
            od.Process(
                "WJets_HT100to200",
                302,
                label=r"WJets HT 100-200",
                xsecs={
                    13: sn.Number(1627.45),
                },
            ),
            od.Process(
                "WJets_HT200to400",
                303,
                label=r"WJets HT 200-400",
                xsecs={
                    13: sn.Number(435.237),
                },
            ),
            od.Process(
                "WJets_HT400to600",
                304,
                label=r"WJets HT 400-600",
                xsecs={
                    13: sn.Number(59.181),
                },
            ),
            od.Process(
                "WJets_HT600to800",
                305,
                label=r"WJets HT 600-800",
                xsecs={13: sn.Number(14.581)},
            ),
            od.Process(
                "WJets_HT800to1200",
                306,
                label=r"WJets HT 800-1200",
                xsecs={
                    13: sn.Number(6.656),
                },
            ),
            od.Process(
                "WJets_HT1200to2500",
                307,
                label=r"WJets HT 1200-2500",
                xsecs={13: sn.Number(1.608)},
            ),
            od.Process(
                "WJets_HT2500toInf",
                308,
                label=r"WJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.039),
                },
            ),
        ],
    )

    cfg.add_process(
        "DY",
        400,
        label=r"$DY \rightarrow l l$",
        label_short="DY",
        color=(100, 100, 100),
        processes=[
            od.Process(
                "DY_HT70to100",
                401,
                label=r"DY HT 70-100",
                xsecs={
                    13: sn.Number(208.977),
                },
            ),
            od.Process(
                "DY_HT100to200",
                402,
                label=r"DY HT 100-200",
                xsecs={
                    13: sn.Number(181.305),
                },
            ),
            od.Process(
                "DY_HT200to400",
                403,
                label=r"DY HT 200-400",
                xsecs={
                    13: sn.Number(50.148),
                },
            ),
            od.Process(
                "DY_HT400to600",
                404,
                label=r"DY HT 400-600",
                xsecs={
                    13: sn.Number(6.984),
                },
            ),
            od.Process(
                "DY_HT600to800",
                405,
                label=r"DY HT 600-800",
                xsecs={13: sn.Number(1.681)},
            ),
            od.Process(
                "DY_HT800to1200",
                406,
                label=r"DY HT 800-1200",
                xsecs={
                    13: sn.Number(0.775),
                },
            ),
            od.Process(
                "DY_HT1200to2500",
                407,
                label=r"DY HT 1200-2500",
                xsecs={13: sn.Number(0.186)},
            ),
            od.Process(
                "DY_HT2500toInf",
                408,
                label=r"DY HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.004),
                },
            ),
        ],
    )

    cfg.add_process(
        "st",
        500,
        label=r"Single t SM",
        label_short="st",
        color=(255, 0, 0),
        processes=[
            od.Process(
                "st_tW_top",
                501,
                label=r"st tW top",
                xsecs={
                    13: sn.Number(35.85),
                },
            ),
            od.Process(
                "st_tW_antitop",
                502,
                label=r"st tW antitop",
                xsecs={
                    13: sn.Number(35.85),
                },
            ),
            od.Process(
                "st_tW_antitop_no_fh",
                503,
                label=r"s antitop no fh",
                xsecs={
                    13: sn.Number(16.295),
                },
            ),
            od.Process(
                "st_tW_top_no_fh",
                504,
                label=r"s top no fh",
                xsecs={
                    13: sn.Number(16.295),
                },
            ),
            od.Process(
                "st_schannel_4f",
                505,
                label=r"st s 4f",
                xsecs={13: sn.Number(3.360)},
            ),
            od.Process(
                "st_tchannel_4f_incl",
                506,
                label=r"st t_ch incl",
                xsecs={
                    13: sn.Number(136.02),
                },
            ),
            od.Process(
                "st_tchannel_antitop_4f_incl",
                507,
                label=r"santit t_ch incl",
                xsecs={13: sn.Number(80.97)},
            ),
        ],
    )

    cfg.add_process(
        "rare",
        600,
        label=r"Rare Processes",
        label_short="rare",
        color=(0, 255, 0),
        processes=[
            od.Process(
                "TTZ_llnunu",
                601,
                label=r"TTZ ll nu nu",
                xsecs={
                    13: sn.Number(0.253),
                },
            ),
            od.Process(
                "TTZ_qq",
                602,
                label=r"TTZ qq",
                xsecs={
                    13: sn.Number(0.530),
                },
            ),
            od.Process(
                "TTWjets_lnu",
                603,
                label=r"TTW+jets l nu",
                xsecs={
                    13: sn.Number(0.204),
                },
            ),
            od.Process(
                "TTWjets_qq",
                604,
                label=r"TTW+jets qq",
                xsecs={
                    13: sn.Number(0.406),
                },
            ),
            od.Process(
                "WW_llnunu",
                605,
                label=r"WW ll nu nu",
                xsecs={13: sn.Number(12.178)},
            ),
            od.Process(
                "WW_lnuqq",
                606,
                label=r"WW l nu qq",
                xsecs={
                    13: sn.Number(49.997),
                },
            ),
            od.Process(
                "WZ_lnuqq",
                607,
                label=r"WZ l nu qq",
                xsecs={13: sn.Number(10.71)},
            ),
            od.Process(
                "WZ_lnununu",
                608,
                label=r"WZ l nununu",
                xsecs={
                    13: sn.Number(3.033),
                },
            ),
            od.Process(
                "WZ_llqq",
                609,
                label=r"WZ ll qq",
                xsecs={
                    13: sn.Number(5.595),
                },
            ),
            od.Process(
                "ZZ_qqnunu",
                610,
                label=r"ZZ qq nunu",
                xsecs={
                    13: sn.Number(4.033),
                },
            ),
            od.Process(
                "ZZ_llnunu",
                611,
                label=r"ZZ ll nunu",
                xsecs={
                    13: sn.Number(0.564),
                },
            ),
            od.Process(
                "ZZ_ll_qq",
                612,
                label=r"ZZ ll qq",
                xsecs={13: sn.Number(3.22)},
            ),
            od.Process(
                "tZq_ll4f",
                613,
                label=r"tZq ll 4f",
                xsecs={
                    13: sn.Number(0.0758),
                },
            ),
        ],
    )

    # write datasets, no crosssection, is_data flag instead

    cfg.add_process(
        "data_electron",
        700,
        label=r"data electron",
        label_short="dat ele",
        color=(0, 0, 0),
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
        "data_muon",
        800,
        label=r"data muon",
        label_short="dat mu",
        color=(0, 0, 0),
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
        "data_MET",
        900,
        label=r"data MET",
        label_short="dat met",
        color=(0, 0, 0),
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
