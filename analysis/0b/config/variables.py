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

    nBinsPt = 35  # 100
    nBinsEta = 32
    minEta = -pi
    maxEta = pi
    nBinsPhi = 30
    minPhi = -pi
    maxPhi = pi
    nBinsMass = 25
    minMass = 0
    maxMass = 1000
    nBinsHt = 42  # 25
    minHt = 0  # 500
    maxHt = 3000  # 5000
    nBinsLt = 35  # 25
    minLt = 0  # 250
    maxLt = 1200  # 2000
    nJets = 20
    minNJets = 0
    maxNJets = 20
    minPt = 0
    maxPt = 1000
    nLep = 5
    minLep = 0
    maxLep = 5
    nIso = 20
    minIso = 0
    maxIso = 1000
    nBool = 2
    minBool = 0
    maxBool = 1
    cfg.add_variable(name="metPt", expression="metPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{miss}$", x_discrete=False)
    cfg.add_variable(name="WBosonMt", expression="WBosonMt", binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{t}^{W}$", x_discrete=False)
    cfg.add_variable(name="LT", expression="LT", binning=(nBinsHt, minLt, maxLt), unit="GeV", x_title="LT", x_discrete=False)
    cfg.add_variable(name="HT", expression="HT", binning=(nBinsLt, minHt, maxHt), unit="GeV", x_title="HT", x_discrete=False)
    cfg.add_variable(name="nJets", expression="nJets", binning=(nJets, minNJets, maxNJets), x_discrete=False)
    # cfg.add_variable(name="nbJets", expression="nbJets", binning=(nJets, minNJets, maxNJets), x_discrete=False)
    cfg.add_variable(name="nWFatJets", expression="nWFatJets", binning=(nJets, minNJets, maxNJets), x_discrete=False)
    cfg.add_variable(name="ntFatJets", expression="ntFatJets", binning=(nJets, minNJets, maxNJets), x_discrete=False)
    # lepton stuff ###############
    cfg.add_variable(name="nMuon", expression="nMuon", binning=(nLep, minLep, maxLep), x_discrete=False)
    cfg.add_variable(name="nElectron", expression="nElectron", binning=(nLep, minLep, maxLep), x_discrete=False)
    cfg.add_variable(name="leadMuonPt", expression="leadMuonPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{\mu 1}$", x_discrete=False)
    cfg.add_variable(name="leadMuonEta", expression="leadMuonEta", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta^{\mu 1}$", x_discrete=False)
    cfg.add_variable(name="leadMuonPhi", expression="leadMuonPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{\mu 1}$", x_discrete=False)
    cfg.add_variable(name="leadElectronPt", expression="leadElectronPt", binning=(nBinsPt, minPt, maxPt), x_title=r"$p_{T}^{e 1}$", x_discrete=False)
    cfg.add_variable(name="leadElectronEta", expression="leadElectronEta", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta^{e 1}$", x_discrete=False)
    cfg.add_variable(name="leadElectronPhi", expression="leadElectronPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{e 1}$", x_discrete=False)
    # jet stuff ##################
    cfg.add_variable(name="jetMass_1", expression="jetMass_1", binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{Jet}^{1}$", x_discrete=False)
    cfg.add_variable(name="jetPt_1", expression="jetPt_1", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{Jet1}$", x_discrete=False)
    cfg.add_variable(name="jetEta_1", expression="jetEta_1", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta_{Jet}^{1}$", x_discrete=False)
    cfg.add_variable(name="jetPhi_1", expression="jetPhi_1", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi_{Jet}^{1}$", x_discrete=False)
    cfg.add_variable(name="jetMass_2", expression="jetMass_1", binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{Jet}^{1}$", x_discrete=False)
    cfg.add_variable(name="jetPt_2", expression="jetPt_2", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{Jet2}$", x_discrete=False)
    cfg.add_variable(name="jetEta_2", expression="jetEta_1", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta_{Jet}^{1}$", x_discrete=False)
    cfg.add_variable(name="jetPhi_2", expression="jetPhi_1", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi_{Jet}^{1}$", x_discrete=False)
    #########################
    cfg.add_variable(name="dPhi", expression="dPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$ \Delta \Phi $", x_discrete=False)

    # variables to check cuts
    # cfg.add_variable(name="correctedMetPt", expression="correctedMetPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"Corrected $p_{T}^{miss}$", x_discrete=False)
    # cfg.add_variable(name="isoTrackPt", expression="isoTrackPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"Iso track $p_{T}$", x_discrete=False)
    # cfg.add_variable(name="isoTrackMt2", expression="isoTrackMt2", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"Iso Track $M_{T2}$", x_discrete=False)
    # cfg.add_variable(name="iso_cut", expression="iso_cut", binning=(nBool, minBool, maxBool), x_discrete=False)
    # cfg.add_variable(name="ghost_muon_filter", expression="ghost_muon_filter", binning=(nBool, minBool, maxBool), x_discrete=False)
    # cfg.add_variable(name="doubleCounting_XOR", expression="doubleCounting_XOR", binning=(nBool, minBool, maxBool), x_discrete=False)
    # cfg.add_variable(name="no_veto_lepton", expression="no_veto_lepton", binning=(nBool, minBool, maxBool), x_discrete=False)
    # cfg.add_variable(name="HLT_Or", expression="HLT_Or", binning=(nBool, minBool, maxBool), x_discrete=False)
    # cfg.add_variable(name="hard_lep", expression="hard_lep", binning=(nBool, minBool, maxBool), x_discrete=False)
    # cfg.add_variable(name="selected", expression="selected", binning=(nBool, minBool, maxBool), x_discrete=False)

    # other binning, important to put them last
    # cfg.add_variable(name="HT_binned", expression="HT", binning=[500,750,1000,1250,2500], unit="GeV", x_title="HT rebinned", x_discrete=True)
    # cfg.add_variable(name="LT_binned", expression="LT", binning=[250,350,450,600,1000], unit="GeV", x_title="LT rebinne", x_discrete=True)
