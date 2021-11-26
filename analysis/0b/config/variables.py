def setup_variables(cfg):
    """
    build all variable histogram configuration to fill in coffea
    each needs a name, a Matplotlib x title and a (#bins, start, end) binning
    template
    cfg.add_variable(
        name="",
        expression="",
        binning=(, , ),
        unit="",
        x_title=r"",
    )
    """

    cfg.add_variable(
        name="METPt",
        expression="METPt",
        binning=(50, 0.0, 150),
        unit="GeV",
        x_title=r"$p_{T}^{miss}$",
    )

    cfg.add_variable(
        name="mu_loose_pt",
        expression="mu_loose_pt",
        binning=(50, 0.0, 150),
        # unit="",
        x_title=r"$p_{T}^{loose}$",
    )

    cfg.add_variable(
        name="mu_tight_pt",
        expression="mu_tight_pt",
        binning=(50, 0.0, 150),
        # unit="",
        x_title=r"$p_{T}^{tight}$",
    )

    cfg.add_variable(
        name="jet_pt",
        expression="jet_pt",
        binning=(50, 0.0, 150),
        # unit="",
        x_title=r"$p_{T}^{jet}$",
    )

    cfg.add_variable(
        name="loose_mT",
        expression="loose_mT",
        binning=(50, 0.0, 150),
        # unit="",
        x_title=r"$m_{T}^{loose}$",
    )

    cfg.add_variable(
        name="tight_mT",
        expression="tight_mT",
        binning=(50, 0.0, 150),
        # unit="",
        x_title=r"$m_{T}^{tight}$",
    )

    cfg.add_variable(
        name="LP",
        expression="LP",
        binning=(30, 0.0, 1.5),
        unit="GeV",
        x_title=r"LP",
    )
