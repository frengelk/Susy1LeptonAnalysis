# coding: utf-8


def setup_categories(cfg):

    Mu_tight = cfg.add_category(
        "Mu_tight",
        label="1 tight muon, 1 jet",
        label_short="tight mu",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )

    Mu_loose = cfg.add_category(
        "Mu_loose",
        label="1 loose muon, 1 jet",
        label_short="loose mu",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
