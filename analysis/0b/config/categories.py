# coding: utf-8
########################################################################################
# Setup Signal bins for the analysis                                                   #
########################################################################################


def setup_categories(cfg):
    N0b = cfg.add_category("N0b", label="0 btagged jets", label_short="0 btag", aux={"fit": True, "fit_variable": "jet1_pt", "cuts":["baselineSelection","HLT_Or", "doubleCounting_XOR", "zerob"]})
    N1ib = cfg.add_category("N1ib", label="1i btagged jets", label_short="1i btag", aux={"fit": True, "fit_variable": "jet1_pt", "cuts":["baselineSelection","HLT_Or", "doubleCounting_XOR", "multib"]})
