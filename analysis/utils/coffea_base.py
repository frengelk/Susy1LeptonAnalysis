"""
This is the base class for the coffea Task
Here we apply our selection and define the categories used for the analysis
Also this write our arrays
"""

import numpy as np
import awkward as ak
from coffea import processor, hist
import ROOT
from coffea.processor.accumulator import (
    dict_accumulator,
    defaultdict_accumulator,
    column_accumulator,
)


class BaseProcessor(processor.ProcessorABC):
    individal_weights = True

    # jes_shifts = False
    # dataset_shifts = False

    def __init__(self, task):
        # self.publish_message = task.publish_message if task.debug else None
        self.config = task.config_inst
        # self.corrections = task.load_corrections()
        self.dataset_axis = hist.Cat("dataset", "Primary dataset")
        # self.dataset_shift_axis = hist.Cat("dataset_shift", "Dataset shift")
        self.category_axis = hist.Cat("category", "Category selection")
        # self.syst_axis = hist.Cat("systematic", "Shift of systematic uncertainty")
        self._accumulator = dict_accumulator(
            n_events=defaultdict_accumulator(int),
            sum_gen_weights=defaultdict_accumulator(float),
            object_cutflow=defaultdict_accumulator(int),
            # cutflow = bh.Histogram(bh.axis.Regular(20, 0, 20)),
            cutflow=hist.Hist("Counts", self.dataset_axis, self.category_axis, self.category_axis, hist.Bin("cutflow", "Cut", 20, 0, 20)),
            n_minus1=hist.Hist("Counts", self.dataset_axis, self.category_axis, self.category_axis, hist.Bin("n_minus1", "Cut", 20, 0, 20)),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def get_dataset(self, events):
        return events.metadata["dataset"]

    def get_dataset_shift(self, events):
        return events.metadata["dataset"][1]

    def get_lfn(self, events):
        ds = self.get_dataset(events)
        fn = events.metadata["filename"].rsplit("/", 1)[-1]

        for lfn in ds.info[self.get_dataset_shift(events)].aux["lfns"]:
            if lfn.endswith(fn):
                return lfn
        else:
            raise RuntimeError("could not find original LFN for: %s" % events.metadata["filename"])

    def get_pu_key(self, events):
        ds = self.get_dataset(events)
        if ds.is_data:
            return "data"
        else:
            lfn = self.get_lfn(events)
            for name, hint in ds.campaign.aux.get("pileup_lfn_scenario_hint", {}).items():
                if hint in lfn:
                    return name
            else:
                return "MC"


class BaseSelection:
    # dtype = np.float32
    debug_dataset = (
        # "QCD_HT100to200"  # "TTTT""QCD_HT200to300"  # "TTTo2L2Nu"  # "WWW_4F"  # "data_B_ee"
    )

    def obj_get_selected_variables(self, X, n, extra=()):
        # TODO
        pass

    def get_selection_as_np(self, X):
        ret = dict(hl=np.stack([ak.to_numpy(X[var]).astype(np.float32) for var in self.config.variables.names()], axis=-1))
        ret.update(dict(DNNId=ak.to_numpy(X["DNNId"]).astype(np.float32)))
        return ret

    def add_to_selection(self, selection, name, array):
        return selection.add(name, ak.to_numpy(array, allow_missing=True))

    def get_fastsim_SF(self, pt, eta):
        root_file = ROOT.TFile.Open("/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/src/Susy1LeptonAnalysis/Susy1LeptonSkimmer/data/fastsim/2016/leptonScaleFactor/El_CBtight_miniIso0p1_Moriond.root")
        histogram = root_file.Get("El_CBtight_miniIso0p1_Moriond;1")

        # Get the bin edges for pt (X axis) and eta (Y axis)
        x_bins = histogram.GetXaxis().GetNbins()
        y_bins = histogram.GetYaxis().GetNbins()

        x_edges = np.array([histogram.GetXaxis().GetBinLowEdge(i) for i in range(1, x_bins + 2)])
        y_edges = np.array([histogram.GetYaxis().GetBinLowEdge(i) for i in range(1, y_bins + 2)])

        # Use numpy digitize to find the bin index for each pt and eta
        pt_bin_indices = np.digitize(pt, x_edges) - 1  # numpy returns bins from 1, ROOT uses 0-indexed
        eta_bin_indices = np.digitize(eta, y_edges) - 1

        # Now calculate the corresponding ROOT bin number from pt and eta indices
        # ROOT's bin numbering is (global bin number) = x + (x_bins + 2) * y
        bin_numbers = (pt_bin_indices + 1) + (x_bins + 2) * (eta_bin_indices + 1)

        # Retrieve the bin contents for all bin numbers
        bin_values = np.array([histogram.GetBinContent(b) for b in bin_numbers])

        return bin_values

    def get_base_variable(self, events):
        ntFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagTvsQCD), value=-999)
        nWFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagWvsQCD), value=-999)
        jetMass_1 = ak.fill_none(ak.firsts(events.JetMass[:, 0:1]), value=-999)
        jetEta_1 = ak.fill_none(ak.firsts(events.JetEta[:, 0:1]), value=-999)
        jetPhi_1 = ak.fill_none(ak.firsts(events.JetPhi[:, 0:1]), value=-999)
        jetMass_2 = ak.fill_none(ak.firsts(events.JetMass[:, 1:2]), value=-999)
        jetEta_2 = ak.fill_none(ak.firsts(events.JetEta[:, 1:2]), value=-999)
        jetPhi_2 = ak.fill_none(ak.firsts(events.JetPhi[:, 1:2]), value=-999)
        jetMass_3 = ak.fill_none(ak.firsts(events.JetMass[:, 2:3]), value=-999)
        jetEta_3 = ak.fill_none(ak.firsts(events.JetEta[:, 2:3]), value=-999)
        jetPhi_3 = ak.fill_none(ak.firsts(events.JetPhi[:, 2:3]), value=-999)
        # nJets = events.nJet
        LT = events.LT
        LP = events.LP
        # calculate per hand since we saw precision errors in C++ skimming
        # HT_old = events.HT
        metPhi = events.MetPhi
        WBosonMt = events.WBosonMt
        # doing abs to stay in sync with old plots
        dPhi = abs(events.DeltaPhi)
        # nbJets = events.nDeepJetMediumBTag
        # # name with underscores lead to problems
        # zerob = nbJets == 0
        multib = nbJets >= 1
        # variables to check cuts
        correctedMetPt = events.CorrectedMetPt
        isoTrackPt = ak.fill_none(ak.firsts(events.IsoTrackPt), value=-999)
        isoTrackMt2 = ak.fill_none(ak.firsts(events.IsoTrackMt2), value=-999)
        return locals()

    def get_gen_variable(self, events):
        genMetPt = events.GenMetPt
        genMetPhi = events.GenMetPhi
        genJetPt_1 = events.GenJetPt_1
        genJetPhi_1 = events.GenJetPhi_1
        genJetEta_1 = events.GenJetEta_1
        genJetMass_1 = events.GenJetMass_1
        genJetPt_2 = events.GenJetPt_2
        genJetPhi_2 = events.GenJetPhi_2
        genJetEta_2 = events.GenJetEta_2
        genJetMass_2 = events.GenJetMass_2
        genElectronPt = events.GenElectronPt_1
        genElectronPhi = events.GenElectronPhi_1
        genElectronEta = events.GenElectronEta_1
        genMuonPt = events.GenMuonPt_1
        genMuonPhi = events.GenMuonPhi_1
        genMuonEta = events.GenMuonEta_1
        return locals()

    def get_muon_variables(self, events):
        # leptons variables
        nMuon = events.nMuon
        leadMuonPt = ak.fill_none(ak.firsts(events.MuonPt[:, 0:1]), -999)
        leadMuonEta = ak.fill_none(ak.firsts(events.MuonEta[:, 0:1]), -999)
        leadMuonPhi = ak.fill_none(ak.firsts(events.MuonPhi[:, 0:1]), -999)
        # MuonMass
        muonCharge = events.MuonCharge
        muonPdgId = events.MuonPdgId
        # vetoMuon = (events.MuonPt[:, 1:2] > 10) & events.MuonLooseId[:, 1:2]

        return locals()

    def get_electron_variables(self, events):
        # leptons variables
        nElectron = events.nElectron
        leadElectronPt = ak.fill_none(ak.firsts(events.ElectronPt[:, 0:1]), -999)
        leadElectronEta = ak.fill_none(ak.firsts(events.ElectronEta[:, 0:1]), -999)
        leadElectronPhi = ak.fill_none(ak.firsts(events.ElectronPhi[:, 0:1]), -999)
        # ElectronMass
        electronCharge = events.ElectronCharge
        electronPdgId = events.ElectronPdgId
        vetoElectron = (events.ElectronPt[:, 1:2] > 10) & events.ElectronLooseId[:, 1:2]

        return locals()

    def base_select(self, events):
        dataset = events.metadata["dataset"]
        # dataset_obj = self.config.get_dataset(dataset)
        proc = self.config.get_process(dataset)
        if proc.is_leaf_process:
            parent_proc = self.config.get_process(proc.parent_processes.names()[0])
        else:
            parent_proc = proc
        shift = events.metadata["shift"]
        summary = self.accumulator.identity()
        size = events.metadata["entrystop"] - events.metadata["entrystart"]
        summary["n_events"][dataset] = size
        summary["n_events"]["sumAllEvents"] = size
        if not events.metadata["isData"]:
            summary["sum_gen_weights"][dataset] = np.sum(events.GenWeight)
        else:
            # just filling a 1 for each event
            summary["sum_gen_weights"][dataset] = 1.0

        # Defining which net to use per event:
        # For now, slightly overpresenting the 1 compared to -1 because it always starts with 1
        # FIXME
        DNNId = np.resize([1, -1], size)

        # Get Variables used for Analysis and Selection
        # locals().update(self.get_base_variable(events))
        # nJets = events.nJet
        LT = events.LT
        LP = events.LP
        WBosonMt = events.WBosonMt
        # doing abs to stay in sync with old plots
        dPhi = abs(events.DeltaPhi)  # FIXME
        # dPhi = events.DeltaPhi
        nbJets = events.nDeepJetMediumBTag
        # name with underscores lead to problems
        zerob = nbJets == 0
        multib = nbJets >= 1
        # variables to check cuts
        correctedMetPt = events.CorrectedMetPt
        isoTrackPt = ak.fill_none(ak.firsts(events.IsoTrackPt), value=-999)
        isoTrackMt2 = ak.fill_none(ak.firsts(events.IsoTrackMt2), value=-999)

        # if events.metadata["isFastSim"]:
        #    locals().update(self.get_gen_variable(events))
        # locals().update(self.get_electron_variables(events))
        # locals().update(self.get_muon_variables(events))
        # doing lep selection
        leptonEta = events.LeptonEta_1
        leptonMass = events.LeptonMass_1
        leptonPhi = events.LeptonPhi_1
        leptonPt = events.LeptonPt_1
        leptonIso = ak.fill_none(ak.firsts(events.MuonMiniIso), 0) + ak.fill_none(ak.firsts(events.ElectronMiniIso), 0)
        nVetoLepton = events.nVetoMuon + events.nVetoElectron
        nGoodLepton = events.nGoodMuon + events.nGoodElectron

        hard_lep = (leptonPt > 25) & (abs(leptonEta) < 2.4)
        # selected = (mu_id | e_id) & ((events.nGoodMuon == 1) | (events.nGoodElectron == 1))
        # no_veto_lepton = (events.nVetoMuon - events.nGoodMuon == 0) & (events.nVetoElectron - events.nGoodElectron == 0)

        # selected leptons, as well as only one good lepton requirement
        ele_sel = (events.ElectronTightId) & (events.ElectronMiniIso < 0.1)
        mu_sel = events.MuonMediumId & (events.MuonMiniIso < 0.2)
        lepton_sel = (ak.fill_none(ak.firsts(mu_sel), False) | ak.fill_none(ak.firsts(ele_sel), False)) & hard_lep & (nVetoLepton == 1) & (nGoodLepton == 1)

        # require anti selcted leptons
        anti_ele_sel = ~((events.ElectronTightId) | (events.ElectronMediumId)) & (events.ElectronMiniIso < 0.4)
        anti_mu_sel = events.MuonMediumId & (events.MuonMiniIso > 0.2)
        lepton_anti_sel = (ak.fill_none(ak.firsts(anti_mu_sel), False) | ak.fill_none(ak.firsts(anti_ele_sel), False)) & hard_lep

        all_ele = (~((events.ElectronTightId) | (events.ElectronMediumId)) | events.ElectronTightId) & (events.ElectronMiniIso < 0.4)
        all_mu = events.MuonMediumId
        all_hard_lepton = (ak.fill_none(ak.firsts(all_ele), False) | ak.fill_none(ak.firsts(all_mu), False)) & hard_lep

        # doing all JetPt calculation here since we need to shift JetPt
        JetPt = events.JetPt
        cleanMask = events.JetIsClean
        # doing all JetPt calculation here since we need to shift JetPt
        JetPt = events.JetPt
        cleanMask = events.JetIsClean
        metPhi = events.MetPhi
        metPt = events.MetPt
        JetMass = events.JetMass

        if shift != "nominal" and shift != "systematic_shifts":
            JetPt = eval("events.JetPt_" + shift)
            cleanMask = eval("events.JetIsClean_" + shift)
            metPt = eval("events.MetPt_" + shift)
            metPhi = eval("events.MetPhi_" + shift)
            if "JER" in shift:
                JetMass = eval("events.JetMass_" + shift)

        # requiring at least loose jets, 30 GeV and inner region, and being cleaned from leptons
        goodJets = (JetPt > 30) & (abs(events.JetEta) < 2.4) & (events.JetId >= 1) & (cleanMask)

        # only feed genuine objects into final histograms
        JetPt = JetPt[goodJets]
        JetEta = events.JetEta[goodJets]
        JetPhi = events.JetPhi[goodJets]

        bTags = events.JetDeepJetMediumId[goodJets]
        zerob = ak.sum(bTags, axis=-1) == 0
        nJets = ak.sum(goodJets, axis=-1)
        nJet3 = nJets >= 3
        nJet5 = nJets >= 5
        nJet34 = (nJets >= 3) & (nJets <= 4)
        # reevaluate variables since shifts change contents
        HT = ak.sum(JetPt, axis=-1)
        HT_cut = HT > 500
        LT = metPt + events.LeptonPt_1
        LT_cut = LT > 250

        jetPt_1 = ak.fill_none(ak.firsts(JetPt[:, 0:1]), value=-999)
        jetPt_2 = ak.fill_none(ak.firsts(JetPt[:, 1:2]), value=-999)
        jetPt_3 = ak.fill_none(ak.firsts(JetPt[:, 2:3]), value=-999)
        subleading_jet = jetPt_2 > 80

        jetMass_1 = ak.fill_none(ak.firsts(JetMass[:, 0:1]), value=-999)
        jetEta_1 = ak.fill_none(ak.firsts(JetEta[:, 0:1]), value=-999)
        jetPhi_1 = ak.fill_none(ak.firsts(JetPhi[:, 0:1]), value=-999)
        jetMass_2 = ak.fill_none(ak.firsts(JetMass[:, 1:2]), value=-999)
        jetEta_2 = ak.fill_none(ak.firsts(JetEta[:, 1:2]), value=-999)
        jetPhi_2 = ak.fill_none(ak.firsts(JetPhi[:, 1:2]), value=-999)
        jetMass_3 = ak.fill_none(ak.firsts(JetMass[:, 2:3]), value=-999)
        jetEta_3 = ak.fill_none(ak.firsts(JetEta[:, 2:3]), value=-999)
        jetPhi_3 = ak.fill_none(ak.firsts(JetPhi[:, 2:3]), value=-999)

        # iso_track = ((events.IsoTrackPt > 10) & (((events.IsoTrackMt2 < 60) & events.IsoTrackIsHadronicDecay) | ((events.IsoTrackMt2 < 80) & ~(events.IsoTrackIsHadronicDecay))))
        # iso_track_cut = ak.sum(iso_track, axis=-1) == 0
        # define selection
        selection = processor.PackedSelection()
        # Baseline PreSelection, split up now
        # baselineSelection = (sortedJets[:, 1] > 80) & (events.LT > 250) & (events.HT > 500) & (ak.num(goodJets) >= 3) & (~events.IsoTrackVeto)
        subleading_jet = ak.fill_none(ak.firsts(JetPt[:, 1:2] > 80), False)

        # check which W/top tags we want to use
        nDeepAk8TopLooseId = events.nDeepAk8TopLooseId
        # nDeepAk8TopMediumId = events.nDeepAk8TopMediumId
        # nDeepAk8TopTightId = events.nDeepAk8TopTightId
        nDeepAk8WLooseId = events.nDeepAk8WLooseId
        # nDeepAk8WMediumId = events.nDeepAk8WMediumId
        # nDeepAk8WTightId = events.nDeepAk8WTightId

        # njet_cut = ak.sum(events.JetIsClean, axis=1) >= 3
        iso_cut = ~events.IsoTrackVeto

        # stitch ttbar at events.LHE_HTIncoming < 600
        if events.metadata["dataset"] == "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8" or events.metadata["dataset"] == "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8":
            LHE_HT_cut = events.LHE_HTIncoming < 600
            # plug it on onto iso_cut, so cutflow is consistent
            iso_cut = iso_cut & LHE_HT_cut
        # do all signal masspoint stuff here
        if events.metadata["isSignal"]:
            if "masspoint" in events.metadata:
                mGlu_cut = events.mGluino == events.metadata["masspoint"][0]
                mNeu_cut = events.mNeutralino == events.metadata["masspoint"][1]
            elif proc.has_aux("masspoint"):
                mGlu_cut = events.mGluino == proc.aux["masspoint"][0]
                mNeu_cut = events.mNeutralino == proc.aux["masspoint"][1]
            else:
                # make sure only masses with defined xsec are in sample
                mGlu_cut = events.mGluino % 5 == 0
                mNeu_cut = events.mNeutralino % 5 == 0
            # expand cut here as well to check for T5
            iso_cut = iso_cut & (mGlu_cut) & (mNeu_cut) & (events.isT5qqqqWW)
        # require correct lepton IDs, applay cut depending on tree name
        # ElectronIdCut = ak.fill_none(ak.firsts(events.ElectronTightId[:, 0:1]), False)
        # MuonIdCut = ak.fill_none(ak.firsts(events.MuonMediumId[:, 0:1]), False)
        # prevent double counting in data
        doubleCounting_XOR = (not events.metadata["isData"]) | ((events.metadata["PD"] == "isSingleElectron") & events.HLT_EleOr) | ((events.metadata["PD"] == "isSingleMuon") & events.HLT_MuonOr & ~events.HLT_EleOr) | ((events.metadata["PD"] == "isMet") & events.HLT_MetOr & ~events.HLT_MuonOr & ~events.HLT_EleOr)
        # if campaign name ends on 2018, HEM only happening then
        if self.config.name.split("_")[-1] == "2018":
            doubleCounting_XOR = (not events.metadata["isData"]) | ((events.metadata["PD"] == "isEGamma") & events.HLT_EleOr) | ((events.metadata["PD"] == "isSingleMuon") & events.HLT_MuonOr & ~events.HLT_EleOr) | ((events.metadata["PD"] == "isMet") & events.HLT_MetOr & ~events.HLT_MuonOr & ~events.HLT_EleOr)
            # Checking for HEM cut in 2018
            # Veto events with any electron with pT > 30GeV, -3.0 < eta < -1.4, and -1.57 < phi < -0.87
            # Setting every event not vetoed to true
            HEM_cut_ele = ak.fill_none(ak.firsts(~((events.nVetoElectron == 1) & ((events.ElectronEta > -3.0) & (events.ElectronEta < -1.4)) & ((events.ElectronPhi > -1.57) & (events.ElectronPhi < -0.87)) & (events.ElectronPt > 30))), value=True)
            # Veto events with any jet with pT > 30 GeV, DeltaPhi (jet, HT,miss) < 0.5, -3.2 <eta< -1.2, and -1.77 < phi < -0.67 (veto enlarged by half the jet cone)
            HT_phi = ak.sum(JetPhi, axis=1)
            HEM_cut_jet = ak.fill_none(~ak.any((((JetEta > -3.2) & (JetEta < -1.2)) & ((JetPhi > -1.77) & (JetPhi < -0.67)) & (JetPt > 30) & (abs(HT_phi - JetPhi) < 0.5)), axis=-1), value=True)

            # this is True for events that are unaffected
            HEM_cut = HEM_cut_jet & HEM_cut_ele

            if events.metadata["isData"]:
                # only affected data sets shall be corrected
                not_affected_events = events.Run < 319077
                HEM_data_cut = not_affected_events | HEM_cut

                iso_cut = iso_cut & HEM_data_cut

        # HLT Combination
        HLT_Or = (not events.metadata["isData"]) | (events.HLT_MuonOr | events.HLT_MetOr | events.HLT_EleOr)
        # ghost muon filter
        ghost_muon_filter = events.MetPt / events.CaloMET_pt <= 5
        common = ["baselineSelection", "doubleCounting_XOR", "HLT_Or"]
        weights = processor.Weights(size, storeIndividual=self.individal_weights)
        if not events.metadata["isData"]:
            lumi = events.metadata["Luminosity"]
            # although 2016 was skimmed in postVFP, the signal has to be scaled to full 2016 lumi since we merge
            if parent_proc.aux["isSignal"] and "2016" in self.config.name:
                lumi = 36
            weights.add("Luminosity", lumi)
            weights.add("GenWeight", events.GenWeight)
            weights.add("sumGenWeight", 1 / events.metadata["sumGenWeight"])

            # not well defined for signal
            if not events.metadata["isSignal"]:
                weights.add("xSec", events.metadata["xSec"] * 1000)  # account for pb / fb
                weights.add(
                    "PreFireWeight",
                    events.PreFireWeight,
                    weightDown=events.PreFireWeightDown,
                    weightUp=events.PreFireWeightUp,
                )
            if events.metadata["isSignal"]:
                weights.add("xSec", events.susyXSectionNLLO * 1000)
                sfs_fast = ["MuonMediumFastSf", "ElectronTightFastSf", "ElectronTightMVAFastSf"]
                for sf in sfs_fast:
                    weights.add(
                        sf,
                        ak.fill_none(ak.firsts(getattr(events, sf)), 1.0),
                        weightDown=ak.fill_none(ak.firsts(getattr(events, sf + "Down")), 1.0),
                        weightUp=ak.fill_none(ak.firsts(getattr(events, sf + "Up")), 1.0),
                    )

            # All leptons at once, don't seperate dependent on lepton selection
            sfs = ["MuonMediumSf", "MuonTriggerSf", "MuonMediumIsoSf", "ElectronTightSf", "ElectronRecoSf"]
            for sf in sfs:
                weights.add(
                    sf,
                    ak.fill_none(ak.firsts(getattr(events, sf)), 1.0),
                    weightDown=ak.fill_none(ak.firsts(getattr(events, sf + "Down")), 1.0),
                    weightUp=ak.fill_none(ak.firsts(getattr(events, sf + "Up")), 1.0),
                )
            # weights.add(
            # "nISRWeight_Mar17",
            # events.nISRWeight_Mar17,
            # weightDown=events.nISRWeightDown_Mar17,
            # weightUp=events.nISRWeightDown_Mar17,
            # )

            weights.add(
                "PileUPWeight",
                events.PileUpWeight,
                weightDown=events.PileUpWeightDown,
                weightUp=events.PileUpWeightUp,
            )
            # loading and applying calculated btag SF
            # by design, order should be the same for events, so we load only part of the batch
            btagSF = np.load(events.metadata["btagSF"])[events.metadata["entrystart"] : events.metadata["entrystop"]]
            btagSF_up = np.load(events.metadata["btagSF_up"])[events.metadata["entrystart"] : events.metadata["entrystop"]]
            btagSF_down = np.load(events.metadata["btagSF_down"])[events.metadata["entrystart"] : events.metadata["entrystop"]]
            weights.add(
                "JetDeepJetMediumSf",
                btagSF,
                weightDown=btagSF_down,
                weightUp=btagSF_up,
            )

            # HEM weight is centrally computed
            if self.config.name.split("_")[-1] == "2018":
                weights.add("HEM", ak.where(HEM_cut, 1, (1 - 0.66)))

        cat = self.config.get_category(events.metadata["category"])
        # for cat in events.metadata["category"]:
        for cut in cat.get_aux("cuts"):
            if cut[1] == "cut":
                self.add_to_selection(selection, " ".join(cut), eval(cut[0]))
            else:
                self.add_to_selection(selection, " ".join(cut), eval("events.{} {}".format(cut[0], cut[1])))
        categories = {cat.name: [" ".join(cut) for cut in cat.get_aux("cuts")]}  # for cat in self.config.categories}
        return locals()


class ArrayAccumulator(column_accumulator):
    """column_accumulator with delayed concatenate"""

    def __init__(self, value):
        self._empty = value[:0]
        self._value = [value]

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.value)

    def identity(self):
        return self.__class__(self._empty)

    def add(self, other):
        assert self._empty.shape == other._empty.shape
        assert self._empty.dtype == other._empty.dtype
        self._value.extend(v for v in other._value if len(v))

    @property
    def value(self):
        if len(self._value) > 1:
            self._value = [np.concatenate(self._value)]
        return self._value[0]

    def __len__(self):
        return sum(map(len, self._value))


class ArrayExporter(BaseProcessor, BaseSelection):
    output = "*.npy"
    dtype = None
    sep = "_"

    def __init__(self, task, Lepton, additional_plots=False):
        super().__init__(task)
        self.Lepton = Lepton
        self.additional_plots = additional_plots

        self._accumulator["arrays"] = dict_accumulator()

    def categories(self, select_output):
        # For reference the categories here are e.g. 0b or multi b
        # Creates dict where all selection are applied -> {category: combined selection per category}
        selection = select_output.get("selection")
        categories = select_output.get("categories")
        return {cat: selection.all(*cuts) for cat, cuts in categories.items()} if selection and categories else {"all": slice(None)}

    def select(self, events):
        # applies selction and returns all variables and all defined objects
        out = self.base_select(events)
        return out

    def process(self, events):
        # Applies indivudal selection per category and then combines them
        selected_output = self.select(events)
        categories = self.categories(selected_output)
        output = selected_output["summary"]
        arrays = self.get_selection_as_np(selected_output)
        # setting weights as extra axis in arrays
        # arrays.setdefault("weights", np.stack([np.full_like(weights, 1), weights], axis=-1))
        weights = selected_output["weights"]
        cat = list(categories.keys())[0]
        # if we selected a shift beforehand, we don't set nominal weight but instead shift one weight accordingly
        # Data only should get nominal which is the default value
        arrays.setdefault("weights", weights.weight())
        if selected_output["events"].metadata["shift"] in self.config.get_aux("systematic_variable_shifts"):
            arrays.setdefault(selected_output["events"].metadata["shift"], weights.weight())
        elif selected_output["events"].metadata["shift"] == "systematic_shifts":
            # calling the weight object with the wanted shift varies the nominal by the Up/Down
            all_shifts = self.config.get_aux("systematic_shifts")
            if events.metadata["isSignal"]:
                all_shifts += self.config.get_aux("systematic_shifts_signal")
            for shift in all_shifts:
                # Prefire not well defined in signal NanoAOD, so we forge a syst of 1 here
                if events.metadata["isSignal"] and "PreFireWeight" in shift:
                    shift_weight = weights.weight()
                else:
                    shift_weight = weights.weight(shift)
                arrays.setdefault(shift, shift_weight)

        if np.max(arrays["hl"]) > 1e20:
            print("ATTENTION!!! Likely error in processing")
        output["arrays"] = dict_accumulator({category + "_" + selected_output["dataset"]: dict_accumulator({key: ArrayAccumulator(array[cut, ...]) for key, array in arrays.items()}) for category, cut in categories.items()})
        # option to do cutflow and N1 plots on the fly
        if self.additional_plots:
            for cat in selected_output["categories"].keys():
                # filling cutflow hist
                cutflow_cuts = set()
                output["cutflow"].fill(
                    dataset=selected_output["dataset"],
                    category=cat,
                    cutflow=np.array([0]),
                    weight=np.array([weights.weight().sum()]),
                )
                selection = selected_output.get("selection")
                categories = selected_output.get("categories")
                for i, cutflow_cut in enumerate(categories[cat]):
                    cutflow_cuts.add(cutflow_cut)
                    cutflow_cut = selection.all(*cutflow_cuts)
                    output["cutflow"].fill(
                        dataset=selected_output["dataset"],
                        category=cat,
                        cutflow=np.array([i + 1]),
                        weight=np.array([weights.weight()[cutflow_cut].sum()]),
                    )
                # filling N-1 plots
                output["n_minus1"].fill(
                    dataset=selected_output["dataset"],
                    category=cat,
                    n_minus1=np.array([0]),
                    weight=np.array([weights.weight().sum()]),
                )
                allCuts = set(categories[cat])
                for i, cut in enumerate(categories[cat]):
                    output["n_minus1"].fill(
                        dataset=selected_output["dataset"],
                        category=cat,
                        n_minus1=np.array([i + 1]),
                        weight=np.array([weights.weight()[selection.all(*(allCuts - {cut}))].sum()]),
                    )
            # output["n_events"]["sumAllEvents"] += selected_output["size"]
        return output

    def postprocess(self, accumulator):
        return accumulator


class Histogramer(BaseProcessor, BaseSelection):
    def variables(self):
        return self.config.variables

    def __init__(self, task):
        super().__init__(task)

        self._accumulator["histograms"] = dict_accumulator(
            {
                var.name: hist.Hist(
                    "Counts",
                    self.dataset_axis,
                    self.category_axis,
                    # self.syst_axis,
                    hist.Bin(
                        var.name,
                        var.x_title,
                        var.binning[0],
                        var.binning[1],
                        var.binning[2],
                    ),
                )
                for var in self.variables()
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    def categories(self, select_output):
        # For reference the categories here are e.g. 0b or multi b
        # Creates dict where all selection are applied -> {category: combined selection per category}
        selection = select_output.get("selection")
        categories = select_output.get("categories")
        return selection, categories
        # return {cat: selection.all(*cuts) for cat, cuts in categories.items()} if selection and categories else {"all": slice(None)}

    def process(self, events):
        output = self.accumulator.identity()
        selected_output = self.base_select(events)
        weights = selected_output["weights"]
        for cat in selected_output["categories"].keys():
            for var_name in self.variables().names():
                weight = weights.weight()
                # value = out[var_name]
                # generate blank mask for variable values
                mask = np.ones(len(selected_output[var_name]), dtype=bool)
                # combine cuts together: problem, some have None values
                for cut in selected_output["categories"][cat]:  # FIXME
                    cut_mask = ak.to_numpy(selected_output[cut])
                    if type(cut_mask) is np.ma.core.MaskedArray:
                        cut_mask = cut_mask.mask
                    mask = np.logical_and(mask, cut_mask)  # .mask

                values = {}
                values["dataset"] = selected_output["dataset"]
                values["category"] = cat
                # we just want to hist every entry so flatten works since we don't wont to deal with nested array structures
                values[var_name] = ak.flatten(selected_output[var_name][mask], axis=None)
                # weight = weights.weight()[cut]
                values["weight"] = weight[mask]
                output["histograms"][var_name].fill(**values)

            # filling cutflow hist
            cutflow_cuts = set()
            output["cutflow"].fill(
                dataset=selected_output["dataset"],
                category=cat,
                cutflow=np.array([0]),
                weight=np.array([weights.weight().sum()]),
            )
            selection, categories = self.categories(selected_output)
            for i, cutflow_cut in enumerate(categories[cat]):
                cutflow_cuts.add(cutflow_cut)
                cutflow_cut = selection.all(*cutflow_cuts)
                output["cutflow"].fill(
                    dataset=selected_output["dataset"],
                    category=cat,
                    cutflow=np.array([i + 1]),
                    weight=np.array([weights.weight()[cutflow_cut].sum()]),
                )
            # filling N-1 plots
            output["n_minus1"].fill(
                dataset=selected_output["dataset"],
                category=cat,
                cutflow=np.array([0]),
                weight=np.array([weights.weight().sum()]),
            )
            allCuts = set(categories[cat])
            for i, cut in enumerate(categories[cat]):
                output["n_minus1"].fill(
                    dataset=selected_output["dataset"],
                    category=cat,
                    cutflow=np.array([i + 1]),
                    weight=np.array([weights.weight()[selection.all(*(allCuts - {cut}))].sum()]),
                )
        output["n_events"]["sumAllEvents"] += selected_output["size"]
        return output

    def postprocess(self, accumulator):
        return accumulator
