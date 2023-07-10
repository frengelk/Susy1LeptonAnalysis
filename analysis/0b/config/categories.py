# coding: utf-8
########################################################################################
# Setup Signal bins for the analysis                                                   #
########################################################################################


def setup_categories(cfg):
    # N1ib = cfg.add_category("N1ib", label="1i btagged jets", label_short="1i btag", aux={"fit": True, "fit_variable": "jet1_pt", "cuts": ["njet_cut", "HLT_Or", "doubleCounting_XOR", "ghost_muon_filter", "subleading_jet", "LT_cut", "HT_cut", "njet_cut", "iso_cut", "multib"]})
    # data_skim = cfg.add_category("N0b", label="0 btagged jets", label_short="0 btag", aux={"cuts":["skim_cut"]})
    # N1ib = cfg.add_category("N1ib", label="1i btagged jets", label_short="1i btag", aux={"cuts":["skim_cut", "selected", "no_veto_lepton", "ghost_muon_filter","HLT_Or", "doubleCounting_XOR", "njet_cut", "iso_cut", "multib"]})
    # for trig in triggers:
    #    cfg.add_category("N0b_" + trig, label="0 btagged jets", label_short="0 btag", aux={"cuts": ["njet_cut", trig, "HLT_Or", "doubleCounting_XOR", "ghost_muon_filter", "subleading_jet", "LT_cut", "HT_cut", "iso_cut", "zerob"]})  # "data_cut",

    #N0b = cfg.add_category("N0b", label="0 btagged jets", label_short="0 btag", aux={"cuts": ["hard_lep", "selected", "no_veto_lepton", "METFilter", "njet_cut", "HLT_Or", "doubleCounting_XOR", "ghost_muon_filter", "subleading_jet", "LT_cut", "HT_cut", "iso_cut", "zerob"]})
    Min_cuts = cfg.add_category("Min_cuts", label="Min_cuts", label_short="min cuts", aux={"cuts": [("HT", "> 2500"), ("LT", "> 50")]})
