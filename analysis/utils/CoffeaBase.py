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
        "mu_loose_pt",
        "mu_tight_pt",
        "jet_pt",
        "met",
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

    def add_to_selection(self, selection, name, array):
        return selection.add(name, ak.to_numpy(array, allow_missing=True))

    def delta_R(self, muon, jet):
        return np.sqrt((muon.eta - jet.eta) ** 2 + (muon.phi - jet.phi) ** 2)

    def mT(self, lep, met_pt, met_phi):
        return np.sqrt(2 * lep.pt * met_pt * (1 - np.cos(lep.phi - met_phi)))

    def LP(self, lep, W_pt, W_phi):
        return lep.pt / W_pt * np.cos(abs(lep.phi - W_phi))

    def select(self, events):

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

        # print("\n",process.name, "\n")

        # mu_trigger =events.HLT_IsoMu24
        # events.Muon_pt > 30
        # abs(events.Muon_eta) < 2.4
        # events.Muon_pfRelIso04_all < 0.4
        # #events.Muon_looseId 0.4
        # #events.Muon_tightId 0.15
        # events.Jet_pt > 30
        # abs(events.Jet_eta) < 2.4
        # # delta R jet muon > 0.4
        # #events.Jet_phi
        # events.FatJet_pt > 170
        # abs(events.FatJet_eta) <2.4
        # events.FatJet_mass > 40

        muons = ak.zip(
            {
                "pt": events.Muon_pt,
                "eta": events.Muon_eta,
                "phi": events.Muon_phi,
                "mass": events.Muon_mass,
                "charge": events.Muon_charge,
            },
            with_name="PtEtaPhiMCandidate",
        )

        good_muon_cut = (abs(events.Muon_eta) < 2.4) & (events.Muon_pt > 30)

        good_muons = muons[good_muon_cut]

        loose_cut = events.Muon_looseId  # (events.Muon_pfRelIso04_all < 0.4) &
        tight_cut = events.Muon_tightId  # (events.Muon_pfRelIso04_all < 0.15) &

        jets = ak.zip(
            {
                "pt": events.Jet_pt,
                "eta": events.Jet_eta,
                "phi": events.Jet_phi,
                "mass": events.Jet_mass,
            },
            with_name="PtEtaPhiMCandidate",
        )

        good_jet_cut = (abs(events.Jet_eta) < 2.4) & (events.Jet_pt > 30)
        good_jets = jets[good_jet_cut]

        good_loose_muons = muons[good_muon_cut]  # & loose_cut]#[:,:1]
        good_tight_muons = muons[good_muon_cut]  # & tight_cut]#[:,:1]

        dR_jets_cut = []

        # # jet cleaning from muons
        # for i,arr in enumerate(good_jets):
        # bool_arr=[]
        # if ak.any(arr):
        # for jet in arr:
        # # fill false for not existing muons
        # if ak.any(good_loose_muons[i]):
        # delR = self.delta_R(good_loose_muons[i], jet)
        # for dR in delR:
        # if dR>0.4:
        # bool_arr.append(jet)

        # dR_jets_cut.append(bool_arr)

        # surviving_jets = ak.Array(dR_jets_cut)
        surviving_jets = good_jets

        # d_R_jmu = self.delta_R(good_loose_muons, good_jets[:,:1])
        # good_jets = good_jets[d_R_jmu_cut > 0.4]

        # from IPython import embed;embed()

        fatjets = ak.zip(
            {
                "pt": events.FatJet_pt,
                "eta": events.FatJet_eta,
                "phi": events.FatJet_phi,
                "mass": events.FatJet_mass,
            },
            with_name="PtEtaPhiMCandidate",
        )

        W_reco = ak.zip(
            {
                "pt": events.MET_pt + good_muons.pt,
                "eta": good_muons.eta,
                "phi": events.MET_phi + good_muons.phi,
                "mass": good_muons.mass,
            },
            with_name="PtEtaPhiMCandidate",
        )

        good_fatjet_cut = (
            (fatjets.pt > 170) & (abs(fatjets.eta) < 2.4) & (fatjets.mass > 40)
        )

        good_fatjets = fatjets[good_fatjet_cut]

        nJets = events.nJet
        nFatjets = events.nFatJet
        nMuons = events.nMuon

        one_loose = ak.num(good_loose_muons) == 1
        one_tight = ak.num(good_tight_muons) == 1

        one_jet = ak.num(surviving_jets) == 1

        zero_fatjet = ak.num(fatjets) == 0

        #

        # delta_R_cut_final = self.delta_R(good_muons, surviving_jets) > 0.7
        # hacky, but if we cut on one jet, just take the first one
        # loose_dR_cut = (self.delta_R(ak.firsts(surviving_jets), ak.firsts(good_loose_muons)) > 0.7)
        # tight_dR_cut = (self.delta_R(ak.firsts(surviving_jets), ak.firsts(good_tight_muons)) > 0.7)

        jet_sel = (
            (one_jet)
            & (zero_fatjet)
            #    & (delta_R_cut_final)
        )

        """
        mT = sqrt(2*LepPt*Met*(1-cos(LepPhi - MetPhi))

        LP = lepPt/WPt*cos(abs(lepPhi - WPhi))
        """

        # from IPython import embed;embed()

        self.add_to_selection(selection, "one_tight", one_tight)
        self.add_to_selection(selection, "one_loose", one_loose)
        self.add_to_selection(selection, "jet_sel", jet_sel)
        # self.add_to_selection(selection, "loose_dR_cut", loose_dR_cut)
        # self.add_to_selection(selection, "tight_dR_cut", tight_dR_cut)

        #
        mu_loose_pt = ak.fill_none(ak.firsts(good_loose_muons.pt), value=-999)
        mu_tight_pt = ak.fill_none(ak.firsts(good_tight_muons.pt), value=-999)

        jet_pt = ak.fill_none(ak.firsts(surviving_jets.pt), value=-999)

        METPt = events.MET_pt

        loose_mT = ak.fill_none(
            ak.firsts(self.mT(good_loose_muons, events.MET_pt, events.MET_phi)),
            value=-999,
        )
        tight_mT = ak.fill_none(
            ak.firsts(self.mT(good_tight_muons, events.MET_pt, events.MET_phi)),
            value=-999,
        )

        LP = ak.fill_none(
            ak.firsts(self.LP(good_muons, W_reco.pt, W_reco.phi)), value=-999
        )

        # eject je seelction for now
        categories = dict(
            Mu_tight=["one_tight"],  # "jet_sel", , "tight_dR_cut"
            Mu_loose=["one_loose"],  # "jet_sel", , "loose_dR_cut"
        )

        # from IPython import embed;embed()

        # print("done", "\n")

        return locals()


class array_accumulator(column_accumulator):
    """ column_accumulator with delayed concatenate """

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
        out = self.select(events)
        weights = out["weights"]

        for var_name in self.variables().names():
            for cat in out["categories"].keys():
                weight = weights.weight()
                # value = out[var_name]
                # generate blank mask for variable values
                # mask = np.ones(len(out[var_name]), dtype=bool)

                # combine cuts together: problem, some have None values
                """
                for cut in out["categories"][cat]:
                    # rint(cut, "\n")
                    cut_mask = ak.to_numpy(out[cut])
                    # catch outlier cases
                    if type(cut_mask) is np.ma.core.MaskedArray:
                        cut_mask = cut_mask.mask
                    mask = np.logical_and(mask, cut_mask)  # .mask
                    # print(np.sum(mask))
                    # value = value[out[cut]]
                """
                # mask = out[out["categories"][cat][0]] & out[out["categories"][cat][1]] & out[out["categories"][cat][2]]
                # only one cut atm
                mask = out[out["categories"][cat][0]]
                mask = ak.to_numpy(ak.fill_none(mask, value=False))

                # from IPython import embed;embed()

                # mask = ak.to_numpy(mask).mask
                values = {}
                values["dataset"] = out["dataset"]
                values["category"] = cat
                values[var_name] = out[var_name][mask]
                # weight = weights.weight()[cut]
                values["weight"] = weight[mask]
                # print(var_name, ":", out[var_name][mask], "\n")
                # from IPython import embed;embed()
                output["histograms"][var_name].fill(**values)

        # output["n_events"] = len(METPt)
        return output

    def postprocess(self, accumulator):
        return accumulator


class ArrayExporter(BaseProcessor, BaseSelection):
    output = "*.npy"
    dtype = None
    sep = "_"

    def __init__(self, task):
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
        out = super().select(events)  # , unc, shift)
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
