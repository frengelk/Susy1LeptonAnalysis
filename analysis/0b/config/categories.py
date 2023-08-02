# coding: utf-8
########################################################################################
# Setup Signal bins for the analysis                                                   #
########################################################################################

triggers = [
    "HLT_Ele115_CaloIdVT_GsfTrkIdT",
    "HLT_Ele15_IsoVVVL_PFHT450",
    "HLT_Ele35_WPTight_Gsf",
    "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
    "HLT_EleOr",
    "HLT_IsoMu27",
    "HLT_MetOr",
    "HLT_Mu15_IsoVVVL_PFHT450",
    "HLT_Mu50",
    "HLT_MuonOr",
    "HLT_PFMET100_PFMHT100_IDTight",
    "HLT_PFMET110_PFMHT110_IDTight",
    "HLT_PFMET120_PFMHT120_IDTight",
    "HLT_PFMETNoMu100_PFMHTNoMu100_IDTight",
    "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight",
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
]


def setup_categories(cfg):
    # write "cut" for boolean cuts
    # Otherwise declare variable from skimmed root file with respective cut
    # ("hard_lep", "cut"), ("selected", "cut"), ("no_veto_lepton", "cut"), ("nJet", ">=3"), ("HLT_Or", "cut"),
    N0b = cfg.add_category("N0b", label="0 btagged jets", label_short="0 btag", aux={"cuts": [("LT", ">350"), ("HT", ">500"), ("ghost_muon_filter", "cut"), ("JetPt_2", ">80"), ("nDeepJetMediumBTag", "==0"), ("iso_cut", "cut"), ("doubleCounting_XOR", "cut")]})
    N1ib = cfg.add_category("N1ib", label=">=1 btagged jets", label_short="1i btag", aux={"cuts": [("LT", ">350"), ("HT", ">500"), ("ghost_muon_filter", "cut"), ("JetPt_2", ">80"), ("nDeepJetMediumBTag", ">=1"), ("iso_cut", "cut"), ("doubleCounting_XOR", "cut")]})
    # Min_cuts = cfg.add_category("Min_cuts", label="Min_cuts", label_short="min cuts", aux={"cuts": [("LT", "> 350"), ("HT", ">500")]})
    # No_cuts = cfg.add_category("No_cuts", label="No_cuts", label_short="No cuts", aux={"cuts": []})
