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
    minEta = -2.4
    maxEta = 2.4
    nBinsPhi = 15
    minPhi = -pi
    maxPhi = pi
    nBinsMass = 25
    minMass = 0
    maxMass = 500
    nBinsHt = 30
    minHt = 500
    maxHt = 3000  # 5000
    nBinsLt = 24  # 25
    minLt = 250
    maxLt = 1200  # 2000
    nJets = 20
    minNJets = 0
    maxNJets = 20
    minPt = 25
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
    cfg.add_variable(name="metPhi", expression="metPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{MET}$", x_discrete=False)
    cfg.add_variable(name="WBosonMt", expression="WBosonMt", binning=(nBinsMass, minMass, maxPt), unit="GeV", x_title=r"$m_{t}^{W}$", x_discrete=False)
    cfg.add_variable(name="LP", expression="LP", binning=(30, -0.5, 2.5), x_title=r"LP", x_discrete=False)
    cfg.add_variable(name="LT", expression="LT", binning=(nBinsLt, minLt, maxLt), unit="GeV", x_title="LT", x_discrete=False)
    cfg.add_variable(name="HT", expression="HT", binning=(nBinsHt, minHt, maxHt), unit="GeV", x_title="HT", x_discrete=False)
    cfg.add_variable(name="nJets", expression="nJets", binning=(nJets, minNJets, maxNJets), x_title="nJets", x_discrete=False)
    # cfg.add_variable(name="nbJets", expression="nbJets", binning=(nJets, minNJets, maxNJets), x_discrete=False)
    # cfg.add_variable(name="ntFatJets", expression="ntFatJets", binning=(nJets, minNJets, maxLep), x_discrete=False)
    # deepAK8 tags
    # FIXME, wrong filling, not amount, but real deepAK8 scores getting filled
    # cfg.add_variable(name="nDeepAk8TopMediumId", expression="nDeepAk8TopMediumId", binning=(nLep, minLep, maxLep), x_title="nDeepAk8TopMediumId", x_discrete=False)
    cfg.add_variable(name="nDeepAk8TopLooseId", expression="nDeepAk8TopMediumId", binning=(nLep, minLep, maxLep), x_title="nDeepAk8TopLooseId", x_discrete=False)
    # cfg.add_variable(name="nDeepAk8TopTightId", expression="nDeepAk8TopMediumId", binning=(nLep, minLep, maxLep), x_title="nDeepAk8TopTightId", x_discrete=False)
    # cfg.add_variable(name="nDeepAk8WMediumId", expression="nDeepAk8WMediumId", binning=(nLep, minLep, maxLep), x_title="nDeepAk8WMediumId", x_discrete=False)
    cfg.add_variable(name="nDeepAk8WLooseId", expression="nDeepAk8WMediumId", binning=(nLep, minLep, maxLep), x_title="nDeepAk8WLooseId", x_discrete=False)
    # cfg.add_variable(name="nDeepAk8WTightId", expression="nDeepAk8WMediumId", binning=(nLep, minLep, maxLep), x_title="nDeepAk8WTightId", x_discrete=False)

    # lepton stuff ###############
    """
    cfg.add_variable(name="nMuon", expression="nMuon", binning=(nLep, minLep, maxLep), x_discrete=False)
    cfg.add_variable(name="nElectron", expression="nElectron", binning=(nLep, minLep, maxLep), x_discrete=False)
    cfg.add_variable(name="leadMuonPt", expression="leadMuonPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{\mu 1}$", x_discrete=False)
    cfg.add_variable(name="leadMuonEta", expression="leadMuonEta", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta^{\mu 1}$", x_discrete=False)
    cfg.add_variable(name="leadMuonPhi", expression="leadMuonPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{\mu 1}$", x_discrete=False)
    cfg.add_variable(name="leadElectronPt", expression="leadElectronPt", binning=(nBinsPt, minPt, maxPt), x_title=r"$p_{T}^{e 1}$", x_discrete=False)
    cfg.add_variable(name="leadElectronEta", expression="leadElectronEta", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta^{e 1}$", x_discrete=False)
    cfg.add_variable(name="leadElectronPhi", expression="leadElectronPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{e 1}$", x_discrete=False)
    """
    cfg.add_variable(name="leptonEta", expression="leptonEta", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta^{lep}$", x_discrete=False)
    # cfg.add_variable(name="leptonMass", expression="leptonMass", binning=(nBinsMass, minMass, maxMass), x_discrete=False)
    cfg.add_variable(name="leptonPhi", expression="leptonPhi", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi^{lep}$", x_discrete=False)
    cfg.add_variable(name="leptonPt", expression="leptonPt", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{lep}$", x_discrete=False)
    # cfg.add_variable(name="leptonIso", expression="leptonIso", binning=(nBinsEta, minEta, maxEta), x_title="MiniIso Lepton", x_discrete=False)
    # cfg.add_variable(name="nLepton", expression="nLepton", binning=(nLep, minLep, maxLep), x_discrete=False)
    # jet stuff for first 3 jets##################
    for i in range(1, 4):
        cfg.add_variable(name="jetMass_{}".format(i), expression="jetMass_{}".format(i), binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{{Jet^{0}}}$".format(i), x_discrete=False)
        cfg.add_variable(name="jetPt_{}".format(i), expression="jetPt_{}".format(i), binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{{T}}^{{Jet^{0}}}$".format(i), x_discrete=False)
        cfg.add_variable(name="jetEta_{}".format(i), expression="jetEta_{}".format(i), binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta_{{Jet^{0}}}$".format(i), x_discrete=False)
        cfg.add_variable(name="jetPhi_{}".format(i), expression="jetPhi_{}".format(i), binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi_{{Jet^{0}}}$".format(i), x_discrete=False)
    # cfg.add_variable(name="jetMass_2", expression="jetMass_2", binning=(nBinsMass, minMass, maxMass), unit="GeV", x_title=r"$m_{Jet}^{2}$", x_discrete=False)
    # cfg.add_variable(name="jetPt_2", expression="jetPt_2", binning=(nBinsPt, minPt, maxPt), unit="GeV", x_title=r"$p_{T}^{Jet2}$", x_discrete=False)
    # cfg.add_variable(name="jetEta_2", expression="jetEta_2", binning=(nBinsEta, minEta, maxEta), x_title=r"$\eta_{Jet}^{2}$", x_discrete=False)
    # cfg.add_variable(name="jetPhi_2", expression="jetPhi_2", binning=(nBinsPhi, minPhi, maxPhi), x_title=r"$\Phi_{Jet}^{2}$", x_discrete=False)
    #########################
    cfg.add_variable(name="dPhi", expression="dPhi", binning=(nBinsPhi, 0, maxPhi), x_title=r"$ \Delta \Phi $", x_discrete=False)

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

    # FIXME
    # cfg.add_variable(name="DNNId", expression="DNNId", binning=(nBool, -1, 1), x_discrete=False)

    # other binning, important to put them last
    # only uncomment when running plotting!
    # cfg.add_variable(name="HT_binned", expression="HT", binning=[500,750,1000,1250,2500], unit="GeV", x_title="HT rebinned", x_discrete=True)
    # cfg.add_variable(name="LT_binned", expression="LT", binning=[250,350,450,600,1000], unit="GeV", x_title="LT rebinned", x_discrete=True)
    # cfg.add_variable(name="dPhi_binned", expression="dPhi", binning=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, maxPhi], x_title=r"$ \Delta \Phi $", x_discrete=True)
