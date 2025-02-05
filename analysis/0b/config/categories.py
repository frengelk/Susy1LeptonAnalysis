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
    N0b = cfg.add_category("N0b", label="0 btagged jets", label_short="0 btag", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("nJet", ">=3"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("zerob", "cut"), ("iso_cut", "cut"), ("doubleCounting_XOR", "cut")]})
    # ("HEM_cut", "cut"),
    SR0b = cfg.add_category("SR0b", label="0 btagged jets, njet5", label_short="0 btag", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("nJet5", "cut"), ("lepton_sel", "cut"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("zerob", "cut"), ("iso_cut", "cut"), ("doubleCounting_XOR", "cut")]})
    # N1ib = cfg.add_category("N1ib", label=">=1 btagged jets", label_short=">=1 btag", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("nDeepJetMediumBTag", ">=1"), ("iso_cut", "cut"), ("doubleCounting_XOR", "cut")]})
    Anti_cuts = cfg.add_category("Anti_cuts", label="Anti_cuts", label_short="min cuts", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("nJet34", "cut"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("zerob", "cut"), ("iso_cut", "cut"), ("lepton_anti_sel", "cut"), ("doubleCounting_XOR", "cut")]})
    SB_cuts = cfg.add_category("SB_cuts", label="Sideband cuts", label_short="SB cuts", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("nJet34", "cut"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("zerob", "cut"), ("iso_cut", "cut"), ("lepton_sel", "cut"), ("doubleCounting_XOR", "cut")]})
    SR_Anti = cfg.add_category("SR_Anti", label="SR_Anti", label_short="SR_Anti cuts", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("nJet5", "cut"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("zerob", "cut"), ("iso_cut", "cut"), ("lepton_anti_sel", "cut"), ("doubleCounting_XOR", "cut")]})
    All_Lep = cfg.add_category("All_Lep", label="All_Lep", label_short="All_Lep", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("nJet3", "cut"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("zerob", "cut"), ("iso_cut", "cut"), ("all_hard_lepton", "cut"), ("doubleCounting_XOR", "cut")]})
    All_Signal = cfg.add_category("All_Signal", label="All_Signal", label_short="All_Signal", aux={"cuts": [("LT_cut", "cut"), ("HT_cut", "cut"), ("nJet3", "cut"), ("ghost_muon_filter", "cut"), ("subleading_jet", "cut"), ("zerob", "cut"), ("iso_cut", "cut"), ("all_hard_lepton", "cut"), ("doubleCounting_XOR", "cut")]})
    # No_cuts = cfg.add_category("No_cuts", label="No_cuts", label_short="No cuts", aux={"cuts": []})
