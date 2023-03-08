########################################################################################
# Setup Signal bins for the analysis                                                   #
########################################################################################
def setup_variables(cfg):
    """
    build all variable histogram configuration to fill in coffea
    each needs a name, a Matplotlib x title and a (#bins, start, end) binning
    template
    cfg.add_variable(name="", expression="", binning=(, , ), unit="", x_title=r"")
    """
    from numpy import pi

    nBinsPt = 25
    nBinsEta = 32
    minEta = -pi
    maxEta = pi
    nBinsPhi = 32
    minPhi = -pi
    maxPhi = pi
    nBinsMass = 25
    minMass = 0
    maxMass = 1000
    nBinsHt = 25
    minHt = 500
    maxHt = 5000
    nBinsLt = 25
    minLt = 250
    maxLt = 2000
    nJets = 20
    minNJets = 0
    maxNJets = 20
    minPt = 0
    maxPt = 1000
    nLep = 5
    minLep = 0
    maxLep = 5
    cfg.add_variable(name="metPt", expression="metPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{miss}$")
    cfg.add_variable(name="WBosonMt", expression="WBosonMt", binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{t}^{W}$")
    cfg.add_variable(name="LT", expression="LT", binning=(nBinsHt, minLt, maxLt), unit="GeV", x_title="LT")
    cfg.add_variable(name="HT", expression="HT", binning=(nBinsLt, minHt, maxHt), unit="GeV", x_title="HT")
    cfg.add_variable(name="nJets", expression="nJets", binning=(nJets, minNJets, maxNJets))
    cfg.add_variable(name="nWFatJets", expression="nWFatJets", binning=(nJets, minNJets, maxNJets))
    cfg.add_variable(name="ntFatJets", expression="ntFatJets", binning=(nJets, minNJets, maxNJets))
    # lepton stuff ###############
    cfg.add_variable(name="nMuon", expression="nMuon", binning=(nLep, minLep, maxLep))
    cfg.add_variable(name="nElectron", expression="nElectron", binning=(nLep, minLep, maxLep))
    cfg.add_variable(name="leadMuonPt", expression="leadMuonPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{lep1}$")
    cfg.add_variable(name="leadMuonEta", expression="leadMuonEta", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta^{lep1}$")
    cfg.add_variable(name="leadMuonPhi", expression="leadMuonPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{lep1}$")
    cfg.add_variable(name="leadElectronPt", expression="leadElectronPt", binning=(nBinsPt, minPt, maxPt), x_title=r"$p_{T}^{lep1}$")
    cfg.add_variable(name="leadElectronEta", expression="leadElectronEta", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta^{lep1}$")
    cfg.add_variable(name="leadElectronPhi", expression="leadElectronPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{lep1}$")
    # jet stuff ##################
    cfg.add_variable(name="jetMass_1", expression="jetMass_1", binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{Jet}^{1}$")
    cfg.add_variable(name="jetPt_1", expression="jetPt_1", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{Jet1}$")
    cfg.add_variable(name="jetEta_1", expression="jetEta_1", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta_{Jet}^{1}$")
    cfg.add_variable(name="jetPhi_1", expression="jetPhi_1", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi_{Jet}^{1}$")
    cfg.add_variable(name="jetMass_2", expression="jetMass_1", binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{Jet}^{1}$")
    cfg.add_variable(name="jetPt_2", expression="jetPt_2", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{Jet2}$")
    cfg.add_variable(name="jetEta_2", expression="jetEta_1", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta_{Jet}^{1}$")
    cfg.add_variable(name="jetPhi_2", expression="jetPhi_1", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi_{Jet}^{1}$")
    #########################
    cfg.add_variable(name="dPhi", expression="dPhi", binning=(nBinsPhi, 2 * minPhi, 2 * maxPhi), x_title=r"$ \Delta \Phi $")
