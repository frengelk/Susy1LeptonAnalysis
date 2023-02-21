import coffea

# from coffea.processor import ProcessorABC
# import law
import numpy as np
import uproot as up
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import hist, processor
from coffea.hist.hist_tools import DenseAxis, Hist

from coffea.processor.accumulator import (
    dict_accumulator,
    defaultdict_accumulator,
    column_accumulator,
    set_accumulator,
)
from coffea.processor.executor import WorkQueueExecutor

# register our candidate behaviors
# from coffea.nanoevents.methods import candidate
# ak.behavior.update(candidate.behavior)


class BaseProcessor(processor.ProcessorABC):
    individal_weights = False

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
            cutflow=hist.Hist(
                "Counts",
                self.dataset_axis,
                self.category_axis,
                self.category_axis,
                hist.Bin("cutflow", "Cut index", 10, 0, 10),
            ),
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
            raise RuntimeError(
                "could not find original LFN for: %s" % events.metadata["filename"]
            )

    def get_pu_key(self, events):
        ds = self.get_dataset(events)
        if ds.is_data:
            return "data"
        else:
            lfn = self.get_lfn(events)
            for name, hint in ds.campaign.aux.get(
                "pileup_lfn_scenario_hint", {}
            ).items():
                if hint in lfn:
                    return name
            else:
                return "MC"


class BaseSelection:

    # common = ("energy", "x", "y", "z")  # , "pt", "eta")
    hl = (
        "MetPt",
        "W_mt",
    )

    # dtype = np.float32
    debug_dataset = (
        # "QCD_HT100to200"  # "TTTT""QCD_HT200to300"  # "TTTo2L2Nu"  # "WWW_4F"  # "data_B_ee"
    )

    def obj_arrays(self, X, n, extra=()):
        assert 0 < n and n == int(n)
        cols = self.hl
        return np.stack(
            [
                getattr(X, a).pad(n, clip=True).fillna(0).regular().astype(np.float32)
                for a in cols
            ],
            axis=-1,
        )

    def arrays(self, X):
        # var_list = []
        # for var in self.config.variables.names():
        # #if var == "jet_mass_1":
        # #from IPython import embed;embed()
        # #c = ak.zeros_like(X[var])
        # a = ak.flatten(X[var], axis=None) # + c
        # #print(var, a, len(a), len(X[var]))
        # b = ak.to_numpy(a).astype(np.float32)
        # var_list.append(b)
        # from IPython import embed;embed()
        return dict(
            # lep=self.obj_arrays(X["good_leptons"], 1, ("pdgId", "charge")),
            # jet=self.obj_arrays(X["good_jets"], 4, ("btagDeepFlavB",)),
            hl=np.stack(
                [
                    ak.to_numpy(X[var]).astype(np.float32)
                    for var in self.config.variables.names()
                ],
                axis=-1,
            ),
            # meta=X["event"].astype(np.int64),
        )
        # print(var_list)
        # from IPython import embed;embed()
        # return dict(hl=np.stack(var_list, axis=1))

    def add_to_selection(self, selection, name, array):
        return selection.add(name, ak.to_numpy(array, allow_missing=True))

    def Muon_select(self, events):

        # set up stuff to fill

        output = self.accumulator.identity()
        selection = processor.PackedSelection()
        size = events.metadata["entrystop"] - events.metadata["entrystart"]
        weights = processor.Weights(size, storeIndividual=self.individal_weights)

        # branches = file.get("nominal")
        dataset = events.metadata["dataset"]
        output["n_events"][dataset] = size
        output["n_events"]["sum_all_events"] = size

        # access instances
        # cheating for now with overwriting
        # dataset = "data_mu_B"  # FIXME
        # from IPython import embed;embed()
        data = self.config.get_dataset(dataset)
        process = self.config.get_process(dataset)

        # print(process.name)

        # leptons variables
        n_leptons = ak.num(events.MuonPt)  # events.nMuon
        n_muon = events.nMuon
        n_electron = events.nElectron
        lead_lep_pt = ak.fill_none(ak.firsts(events.MuonPt[:, 0:1]), 0)
        lead_lep_eta = ak.fill_none(ak.firsts(events.MuonEta[:, 0:1]), 0)
        lead_lep_phi = ak.fill_none(ak.firsts(events.MuonPhi[:, 0:1]), 0)
        # MuonMass
        # tight_lep = events.MuonTightId[:, 0:1]
        lep_charge = events.MuonCharge
        lep_pdgid = events.MuonPdgId

        # construct lepton veto mask
        # hacky: sort leptons around and then use the  == 2 case
        # veto_lepton = np.where(
        # events.nMuon == 2
        # ,ak.sort(events.MuonPt, ascending=True) [:,0] < 10
        # ,(n_leptons == 1)
        # )
        # two_lep = ak.mask(events.MuonPt, (events.nMuon == 2))
        # veto_lepton = two_lep[:, 1] < 10
        veto_lepton = (events.MuonPt[:, 1:2] > 10) & events.MuonLooseId[:, 1:2]

        """
        om gosh, that can be solved so much easier
        look at
        for i in range(500):
            ...:     print(events.MuonPt[:,1:2][i], events.MuonPt[i])
        events.MuonPt[:,1:2] has only the second element, and if it doesnt exist, its empty
        events.MuonPt[:,1 :2] > 10 produces [], False or True, as intended

        works as well:
        cut=(n_leptons>0)
        events.MuonPt.mask[cut]

        In you want to consider combinations of all good particles in each event, so there are functions for constructing that.
        ak.combinations(array.muons, 2).type
        mu1, mu2 = ak.unzip(ak.combinations(array.muons, 2))
        """

        # from IPython import embed;embed()
        # APPLY TRIGGER!!! done before in c++?
        # mu_trigger = events.HLT__IsoMu24

        # muon selection
        muon_selection = events.MuonMediumId[:, 0:1] & (abs(lep_pdgid[:, 0:1]) == 13)

        # lep selection
        lep_selection = (
            (lead_lep_pt > 25)
            # & (veto_lepton | (events.nMuon == 1))
            # & (n_leptons == 1)
            # & (ak.num(veto_lepton) == 0)
            & (muon_selection)
            & (events.nGoodMuon == 1)
            # & (ak.num(veto_lepton) == 0)
            & ((events.nVetoElectron == 0) & (events.nVetoMuon == 0))
            # & ((ak.sum(events.ElectronIsVeto, axis=-1) == 0) & (ak.sum(events.MuonIsVeto, axis=-1) == 0) )
        )

        """
        b=ak.where(ak.num(lead_lep_phi) > 0, lead_lep_pt > 10, False)
        ak.flatten(b, axis=None)
        """

        self.add_to_selection(selection, "lep_selection", ak.firsts(lep_selection))

        # jet variables
        n_jets = events.nJet
        n_b_jets = (
            events.nDeepJetMediumBTag
        )  # nMediumDFBTagJet  # 'nMediumCSVBTagJet' ?
        # used for tagging
        n_W_tags = ak.fill_none(ak.firsts(events.FatJetDeepTagTvsQCD), value=0)
        n_t_tags = ak.fill_none(ak.firsts(events.FatJetDeepTagWvsQCD), value=0)
        jet_mass_1 = ak.fill_none(ak.firsts(events.JetMass[:, 0:1]), value=0)
        jet_pt_1 = ak.fill_none(
            ak.firsts(events.JetPt[:, 0:1]), value=0
        )  # events.JetPt[:, 0:1]
        # unpack nested list, set not existing second jets to 0 -> depends on other cuts
        jet_pt_2 = ak.fill_none(ak.firsts(events.JetPt[:, 1:2]), value=0)
        jet_eta_1 = ak.fill_none(ak.firsts(events.JetEta[:, 0:1]), value=0)
        jet_phi_1 = ak.fill_none(ak.firsts(events.JetPhi[:, 0:1]), value=0)

        # jest isolation selection
        jet_iso_sel = (
            (events.IsoTrackIsHadronicDecay)
            & (events.IsoTrackPt > 10)
            & (events.IsoTrackMt2 < 60)
        )
        # values of variables seem faulty, #FIXME
        # self.add_to_selection(selection,"jet_iso_sel", ~jet_iso_sel[:,0])

        # event variables
        # look at all possibilities with dir(events)
        METPt = events.MetPt
        W_mt = events.WBosonMt
        Dphi = events.DeltaPhi
        LT = events.LT
        HT = events.HT

        good_jets = (events.JetPt > 30) & (abs(events.JetEta) < 2.4)

        # after the tutorial
        # this can be much easier a=sorted_jets[:,2:3]
        sorted_jets = ak.mask(
            events.JetPt, (events.nJet >= 3)
        )  # ak.sort(events.JetPt, ascending=False)

        iso_track = events.IsoTrackVeto

        baseline_selection = (
            lep_selection
            # (lead_lep_pt > 25)
            # &veto lepton > 10
            # &No isolated track with p T ≥ 10 GeV and M T2 < 60 GeV (80 GeV) for hadronic (leptonic)
            & (sorted_jets[:, 1] > 80)
            & (LT > 250)
            & (HT > 500)
            & (ak.num(good_jets) >= 3)  # kinda double, but keep it for now
            & (iso_track)
        )

        # data trigger
        trigger_HLT_Or = events.HLT_MuonOr | events.HLT_MetOr | events.HLT_EleOr
        self.add_to_selection(selection, "trigger_HLT_Or", trigger_HLT_Or)

        # base selection
        zero_b = n_b_jets == 0
        multi_b = n_b_jets >= 1
        self.add_to_selection(
            selection,
            "baseline_selection",
            ak.fill_none(ak.firsts(baseline_selection), False),
        )  # ) baseline_selection)
        self.add_to_selection(selection, "zero_b", zero_b)
        self.add_to_selection(selection, "multi_b", multi_b)

        # W tag?
        # events.nGenMatchedW

        # add trigger selections
        HLT_MuonOr = events.HLT_MuonOr
        HLT_MetOr = events.HLT_MetOr
        HLT_EleOr = events.HLT_EleOr

        self.add_to_selection(selection, "HLT_EleOr", HLT_EleOr)
        self.add_to_selection(selection, "HLT_MuonOr", HLT_MuonOr)
        self.add_to_selection(selection, "HLT_MetOr", HLT_MetOr)

        # prevent double counting in data
        # ~(events.metadata["IsData"]) |
        doubleCounting_XOR = (
            ((events.metadata["PD"] == "isSingleElectron") & HLT_EleOr)
            | ((events.metadata["PD"] == "isSingleMuon") & HLT_MuonOr & ~HLT_EleOr)
            | (
                (events.metadata["PD"] == "isMet")
                & HLT_MetOr
                & ~HLT_MuonOr
                & ~HLT_EleOr
            )
        )
        self.add_to_selection(selection, "doubleCounting_XOR", doubleCounting_XOR)

        # apply some weights,  MC/data check beforehand
        if not process.is_data:
            # if ~(events.metadata["IsData"]):
            weights.add("x_sec", process.xsecs[13.0].nominal)

            # # some weights have more than weight, not always consistent
            # # take only first weight, since everything with more than 1 lep gets ejected
            #
            # weights.add(
            # "MuonSFTrigger",
            # events.MuonSFTrigger[:, 0:1],
            # weightUp=events.MuonSFTriggerUp[:, 0:1],
            # weightDown=events.MuonSFTriggerDown[:, 0:1],
            # )
            #
            # weights.add(
            # "MuonSFIsolation",
            # events.MuonSFIsolation[:, 0:1],
            # weightDown=events.MuonSFIsolationDown[:, 0:1],
            # weightUp=events.MuonSFIsolationUp[:, 0:1],
            # )
            #
            # weights.add(
            # "MuonSFMVA",
            # events.MuonSFMVA[:, 0:1],
            # weightDown=events.MuonSFMVADown[:, 0:1],
            # weightUp=events.MuonSFMVAUp[:, 0:1],
            # )
            #
            # weights.add(
            # "MuonSFGSF",
            # events.MuonSFGSF[:, 0:1],
            # weightDown=events.MuonSFGSFDown[:, 0:1],
            # weightUp=events.MuonSFGSFUp[:, 0:1],
            # )
            #
            # weights.add(
            # "MuonSFId",
            # events.MuonSFId[:, 0:1],
            # weightDown=events.MuonSFIdDown[:, 0:1],
            # weightUp=events.MuonSFIdUp[:, 0:1],
            # )
            #
            # weights.add(
            # "nISRWeight_Mar17",
            # events.nISRWeight_Mar17,
            # weightDown=events.nISRWeightDown_Mar17,
            # weightUp=events.nISRWeightDown_Mar17,
            # )

            # weights.add(
            # "PileUpWeight",
            # events.PileUpWeight[:, 0],
            # weightDown=events.PileUpWeightDown[:, 0],
            # weightUp=events.PileUpWeightUp[:, 0],
            # )

            # weights.add("JetMediumCSVBTagSF", events.JetMediumCSVBTagSF,
            # weightUp = events.JetMediumCSVBTagSFUp,
            # weightDown= events.JetMediumCSVBTagSFDown,
            # )
        """
        veto leptons:
        muons loose working point
        electrons: veto WP of cut based electron id without cut on relative isolation
        medium wp of vut based muon ID used for good muon selection
        good electrons tight wp of cut based electron ID without relative isolation cut
        conversion veto& zero lost hits in innter tracker for good electrons, reject converted photons

        isolation variable: pt sum of all objects in cone divided by lep pt
        p T < 50 GeV, R = 0.2; for 50 GeV < p T < 200 GeV,
        R = 10 GeV/p T ; and for p T > 200 GeV, R = 0.05.

        In addition to the lepton veto, we also veto isolated tracks that could stem from not well iden-
        tified leptons. Charged PF tracks from the primary vertex with p T > 5 GeV are selected, and an
        isolation variable Rel Iso is defined as the p T sum of all charged tracks within a cone of R = 0.3
        around the track candidate (excluding the candidate itself), divided by the track p T . We require
        Rel Iso < 0.1 ( 0.2 ) for hadronic (leptonic) tracks. The isolated track with the the highest-p T and
        opposite charge with respect to the selected lepton is chosen.
        """

        common = [
            "baseline_selection",
            "lep_selection",
            "trigger_HLT_Or",
            "doubleCounting_XOR",
        ]  # "HLT_MuonOr", "HLT_MetOr"]
        # signalRegion = events.signalRegion == 1
        # controlRegion = events.signalRegion == 0
        delta_phi_SR = events.DeltaPhi > 0.9
        delta_phi_CR = events.DeltaPhi < 0.9

        # self.add_to_selection(selection, "signalRegion", signalRegion)
        # self.add_to_selection(selection, "controlRegion", controlRegion)
        self.add_to_selection(selection, "delta_phi_SR", delta_phi_SR)
        self.add_to_selection(selection, "delta_phi_CR", delta_phi_CR)
        # if you need to add more options
        # signal_region = ["signalRegion"]
        # control_region = ["controlRegion"]

        categories = dict(
            N0b=common + ["zero_b"],
            N1ib=common + ["multi_b"],
            # N0b_SR=common + ["zero_b", "delta_phi_SR"],
            # N1b_SR=common + ["multi_b", "delta_phi_SR"],
            # N0b_CR=common + ["zero_b", "delta_phi_CR"],
            # N1b_CR=common + ["multi_b", "delta_phi_CR"],
        )

        return locals()

    ##################Electron####################

    def Electron_select(self, events):

        # print(dir(events))

        # set up stuff to fill
        output = self.accumulator.identity()
        selection = processor.PackedSelection()
        size = events.metadata["entrystop"] - events.metadata["entrystart"]
        weights = processor.Weights(size, storeIndividual=self.individal_weights)

        # branches = file.get("nominal")
        dataset = events.metadata["dataset"]
        output["n_events"][dataset] = size
        output["n_events"]["sum_all_events"] = size

        # access instances
        data = self.config.get_dataset(dataset)
        process = self.config.get_process(dataset)

        # leptons variables
        n_leptons = ak.num(events.ElectronPt)  #
        n_muon = events.nMuon
        n_electron = events.nElectron
        lead_lep_pt = ak.fill_none(ak.firsts(events.ElectronPt[:, 0:1]), 0)
        lead_lep_eta = ak.fill_none(ak.firsts(events.ElectronEta[:, 0:1]), 0)
        lead_lep_phi = ak.fill_none(ak.firsts(events.ElectronPhi[:, 0:1]), 0)
        # MuonMass
        lep_charge = events.ElectronCharge
        lep_pdgid = events.ElectronPdgId
        # veto_lepton = events.ElectronPt[:, 1:2] > 10

        # ele selection, have a good leading leptoni
        electron_selection = events.ElectronTightId[:, 0:1] & (
            abs(lep_pdgid[:, 0:1]) == 11
        )

        # lep selection
        lep_selection = (
            (lead_lep_pt > 25)
            # & (veto_lepton | (events.nMuon == 1))
            & (events.nGoodElectron == 1)  # n_leptons == 1)
            # & (ak.num(veto_lepton) == 0)
            & ((events.nVetoElectron == 0) & (events.nVetoMuon == 0))
            # & ((ak.sum(events.ElectronIsVeto, axis=-1) == 0) & (ak.sum(events.MuonIsVeto, axis=-1) == 0) )
            & (electron_selection)
        )
        self.add_to_selection(selection, "lep_selection", ak.firsts(lep_selection))

        # jet variables
        n_jets = events.nJet
        n_b_jets = (
            events.nDeepJetMediumBTag
        )  # nMediumDFBTagJet  # 'nMediumCSVBTagJet' ?
        # used for tagging
        n_W_tags = ak.fill_none(ak.firsts(events.FatJetDeepTagTvsQCD), value=0)
        n_t_tags = ak.fill_none(ak.firsts(events.FatJetDeepTagWvsQCD), value=0)
        jet_mass_1 = ak.fill_none(ak.firsts(events.JetMass[:, 0:1]), value=0)
        jet_pt_1 = ak.fill_none(
            ak.firsts(events.JetPt[:, 0:1]), value=0
        )  # events.JetPt[:, 0:1]
        # unpack nested list, set not existing second jets to 0 -> depends on other cuts
        jet_pt_2 = ak.fill_none(ak.firsts(events.JetPt[:, 1:2]), value=0)
        jet_eta_1 = ak.fill_none(ak.firsts(events.JetEta[:, 0:1]), value=0)
        jet_phi_1 = ak.fill_none(ak.firsts(events.JetPhi[:, 0:1]), value=0)

        good_jets = (events.JetPt > 30) & (abs(events.JetEta) < 2.4)

        # jest isolation selection
        jet_iso_sel = (
            (events.IsoTrackIsHadronicDecay)
            & (events.IsoTrackPt > 10)
            & (events.IsoTrackMt2 < 60)
        )
        # values of variables seem faulty, #FIXME
        # self.add_to_selection(selection,"jet_iso_sel", ~jet_iso_sel[:,0])

        # event variables
        # look at all possibilities with dir(events)
        METPt = events.MetPt
        W_mt = events.WBosonMt
        Dphi = events.DeltaPhi
        LT = events.LT
        HT = events.HT

        # after the tutorial
        # this can be much easier a=sorted_jets[:,2:3]
        sorted_jets = ak.mask(
            events.JetPt, (events.nJet >= 3)
        )  # ak.sort(events.JetPt, ascending=False)

        iso_track = events.IsoTrackVeto
        # iso_tracks = (
        # (events.IsoTrackPt > 10)
        # & (
        # (events.IsoTrackMt2 < 60) & events.IsoTrackIsHadronicDecay)
        # | ((events.IsoTrackMt2 < 80) & ~(events.IsoTrackIsHadronicDecay))
        # )

        # ghost muon filter
        ghost_muon_filter = events.MetPt / events.CaloMET_pt <= 5

        baseline_selection = (
            lep_selection
            # (lead_lep_pt > 25)
            # &veto lepton > 10
            # &No isolated track with p T ≥ 10 GeV and M T2 < 60 GeV (80 GeV) for hadronic (leptonic)
            & (sorted_jets[:, 1] > 80)
            & (LT > 250)
            & (HT > 500)
            & (ak.num(good_jets) >= 3)  # kinda double, but keep it for now
            # & (ak.num(iso_tracks) ==0)
            & (iso_track)
            # & (ghost_muon_filter)
        )

        # data trigger
        trigger_HLT_Or = events.HLT_MuonOr | events.HLT_MetOr | events.HLT_EleOr
        self.add_to_selection(selection, "trigger_HLT_Or", trigger_HLT_Or)

        # base selection
        zero_b = n_b_jets == 0
        multi_b = n_b_jets >= 1
        self.add_to_selection(
            selection,
            "baseline_selection",
            ak.fill_none(ak.firsts(baseline_selection), False),
        )  # ) baseline_selection)
        self.add_to_selection(selection, "zero_b", zero_b)
        self.add_to_selection(selection, "multi_b", multi_b)

        # W tag?
        # events.nGenMatchedW

        # add trigger selections
        HLT_MuonOr = events.HLT_MuonOr
        HLT_MetOr = events.HLT_MetOr
        HLT_EleOr = events.HLT_EleOr

        self.add_to_selection(selection, "HLT_EleOr", events.HLT_EleOr)
        self.add_to_selection(selection, "HLT_MuonOr", events.HLT_MuonOr)
        self.add_to_selection(selection, "HLT_MetOr", events.HLT_MetOr)

        # prevent double counting in data
        doubleCounting_XOR = (
            # ~(events.metadata["IsData"]) |
            ((events.metadata["PD"] == "isSingleElectron") & HLT_EleOr)
            | ((events.metadata["PD"] == "isSingleMuon") & HLT_MuonOr & ~HLT_EleOr)
            | (
                (events.metadata["PD"] == "isMet")
                & HLT_MetOr
                & ~HLT_MuonOr
                & ~HLT_EleOr
            )
        )
        self.add_to_selection(selection, "doubleCounting_XOR", doubleCounting_XOR)

        # apply some weights,  MC/data check beforehand
        if not process.is_data:
            weights.add("x_sec", process.xsecs[13.0].nominal)

        common = [
            "baseline_selection",
            "lep_selection",
            "trigger_HLT_Or",
            "doubleCounting_XOR",
        ]  # "HLT_MuonOr", "HLT_MetOr"]
        # "baseline_selection",

        delta_phi_SR = events.DeltaPhi > 0.9
        delta_phi_CR = events.DeltaPhi < 0.9

        self.add_to_selection(selection, "delta_phi_SR", delta_phi_SR)
        self.add_to_selection(selection, "delta_phi_CR", delta_phi_CR)

        categories = dict(
            N0b=common + ["zero_b"],
            N1ib=common + ["multi_b"],
        )

        return locals()


# turning off progress bar
# class BaseWorkQueueExecutor(WorkQueueExecutor):


class array_accumulator(column_accumulator):
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

    # def select(self, events):
    # out = super().self.select(events)

    # from IPython import embed;embed()

    def process(self, events):
        output = self.accumulator.identity()
        out = self.Moun_select(events)
        weights = out["weights"]
        for var_name in self.variables().names():
            for cat in out["categories"].keys():
                weight = weights.weight()
                # value = out[var_name]
                # generate blank mask for variable values
                mask = np.ones(len(out[var_name]), dtype=bool)
                # from IPython import embed;embed()
                # combine cuts together: problem, some have None values
                for cut in out["categories"][cat][:1]:  # FIXME
                    # print(cut, "\n")

                    cut_mask = ak.to_numpy(out[cut])
                    if type(cut_mask) is np.ma.core.MaskedArray:
                        cut_mask = cut_mask.mask
                    mask = np.logical_and(mask, cut_mask)  # .mask
                    # print(np.sum(mask))
                    # value = value[out[cut]]

                # from IPython import embed;embed()

                # mask = ak.to_numpy(mask).mask
                # print(var_name)
                values = {}
                values["dataset"] = out["dataset"]
                values["category"] = cat
                # we just want to hist every entry so flatten works since we don't wont to deal with nested array structures
                values[var_name] = ak.flatten(out[var_name][mask], axis=None)
                # weight = weights.weight()[cut]
                values["weight"] = weight[mask]
                # if var_name == 'jet_mass_1' and cat == 'N1b_SR':
                #    from IPython import embed;embed()
                output["histograms"][var_name].fill(**values)

        # output["n_events"] = len(MetPt)
        return output

    def postprocess(self, accumulator):
        return accumulator


class ArrayExporter(BaseProcessor, BaseSelection):
    output = "*.npy"
    dtype = None
    sep = "_"

    def __init__(self, task, Lepton):
        self.Lepton = Lepton
        super().__init__(task)
        self._accumulator["arrays"] = dict_accumulator()

    # def arrays(self, select_output):
    # """
    # select_output is the output of self.select
    # this function should return an dict of numpy arrays, the "weight" key is reserved
    # """
    # pass

    def categories(self, select_output):
        selection = select_output.get("selection")
        categories = select_output.get("categories")
        # from IPython import embed;embed()
        return (
            {cat: selection.all(*cuts) for cat, cuts in categories.items()}
            if selection and categories
            else {"all": slice(None)}
        )

    def select(self, events):  # , unc, shift):
        if self.Lepton == "Muon":
            out = super().Muon_select(events)  # , unc, shift)
        if self.Lepton == "Electron":
            out = super().Electron_select(events)  # , unc, shift)
        dataset = self.get_dataset(events)
        # (process,) = dataset.processes.values()
        # xsec_weight = (
        #    1
        #    if process.is_data
        #    else process.xsecs[13].nominal * self.config.campaign.get_aux("lumi")
        # )
        # out["weights"].add("xsec", xsec_weight)
        return out

    def process(self, events):
        select_output = self.select(events)  # , unc="nominal", shift=None)
        categories = self.categories(select_output)
        weights = select_output["weights"]
        output = select_output["output"]

        # from IPython import embed;embed()

        arrays = self.arrays(select_output)
        if self.dtype:
            arrays = {key: array.astype(self.dtype) for key, array in arrays.items()}

        output["arrays"] = dict_accumulator(
            {
                category
                + "_"
                + select_output["dataset"]: dict_accumulator(
                    {
                        key: array_accumulator(array[cut, ...])
                        for key, array in arrays.items()
                    }
                )
                for category, cut in categories.items()
            }
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
