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
        "METPt",
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

        # print(process.name)
        # from IPython import embed;embed()

        # leptons variables
        n_leptons = events.nLepton
        lead_lep_pt = events.LeptonPt[:, 0]
        lead_lep_eta = events.LeptonEta[:, 0]
        lead_lep_phi = events.LeptonPhi[:, 0]
        # LeptonMass
        # tight_lep = events.LeptonTightId[:, 0]
        lep_charge = events.LeptonCharge
        lep_pdgid = events.LeptonPdgId

        # construct lepton veto mask
        # hacky: sort leptons around and then use the  == 2 case
        # veto_lepton = np.where(
        # events.nLepton == 2
        # ,ak.sort(events.LeptonPt, ascending=True) [:,0] < 10
        # ,(n_leptons == 1)
        # )
        two_lep = ak.mask(events.LeptonPt, (events.nLepton == 2))
        veto_lepton = two_lep[:, 1] < 10

        """
        om gosh, that can be solved so much easier
        look at
        for i in range(500):
            ...:     print(events.LeptonPt[:,1:2][i], events.LeptonPt[i])
        events.LeptonPt[:,1:2] has only the second element, and if it doesnt exist, its empty
        events.LeptonPt[:,1:2] > 10 produces [], False or True, as intended

        works as well:
        cut=(n_leptons>0)
        events.LeptonPt.mask[cut]

        In you want to consider combinations of all good particles in each event, so there are functions for constructing that.
        ak.combinations(array.muons, 2).type
        mu1, mu2 = ak.unzip(ak.combinations(array.muons, 2))
        """

        # from IPython import embed;embed()
        # APPLY TRIGGER!!! done before in c++?
        # mu_trigger = events.HLT_IsoMu24

        # muon selection
        muon_selection = events.LeptonMediumId[:, 0] & (abs(lep_pdgid[:, 0]) == 13)
        # ele selection
        electron_selection = events.LeptonTightId[:, 0] & (abs(lep_pdgid[:, 0]) == 11)

        # lep selection
        lep_selection = (
            (lead_lep_pt > 25)
            & (veto_lepton | (events.nLepton == 1))
            & (muon_selection | electron_selection)
        )
        self.add_to_selection(selection, "lep_selection", lep_selection)

        # jet variables
        n_jets = events.nJet
        n_btags = events.nMediumDFBTagJet  # 'nMediumCSVBTagJet' ?
        jet_mass_1 = events.JetMass[:, 0]
        jet_pt_1 = events.JetPt[:, 0]
        # unpack nested list, set not existing second jets to 0 -> depends on other cuts
        jet_pt_2 = ak.fill_none(ak.firsts(events.JetPt[:, 1:2]), value=0)
        jet_eta_1 = events.JetEta[:, 0]
        jet_phi_1 = events.JetPhi[:, 0]

        # jest isolation selection
        jet_iso_sel = (
            (events.IsoTrackHadronicDecay)
            & (events.IsoTrackPt > 10)
            & (events.IsoTrackMt2 < 60)
        )
        # values of variables seem faulty, #FIXME
        # self.add_to_selection(selection,"jet_iso_sel", ~jet_iso_sel[:,0])

        # event variables
        # look at all possibilities with dir(events)
        METPt = events.METPt
        W_mt = events.WBosonMt
        Dphi = events.DeltaPhi
        LT = events.LT
        HT = events.HT

        # after the tutorial
        # this can be much easier a=sorted_jets[:,2:3]
        sorted_jets = ak.mask(
            events.JetPt, (events.nJet >= 3)
        )  # ak.sort(events.JetPt, ascending=False)

        baseline_selection = (
            # (lead_lep_pt > 25)
            # &veto lepton > 10
            # &No isolated track with p T ≥ 10 GeV and M T2 < 60 GeV (80 GeV) for hadronic (leptonic)
            (sorted_jets[:, 1] > 80)
            & (LT > 250)
            & (HT > 500)
            & (n_jets >= 3)  # kinda double, but keep it for now
        )

        # base selection
        zero_b = n_btags == 0
        multi_b = n_btags >= 1
        self.add_to_selection(selection, "baseline_selection", baseline_selection)
        self.add_to_selection(selection, "zero_b", zero_b)
        self.add_to_selection(selection, "multi_b", multi_b)

        # W tag?
        # events.nGenMatchedW

        # add trigger selections
        HLTLeptonOr = events.HLTLeptonOr
        HLTMETOr = events.HLTMETOr
        HLTElectronOr = events.HLTElectronOr
        HLTMuonOr = events.HLTMuonOr

        self.add_to_selection(selection, "HLTElectronOr", events.HLTElectronOr)
        self.add_to_selection(selection, "HLTLeptonOr", events.HLTLeptonOr)
        self.add_to_selection(selection, "HLTMETOr", events.HLTMETOr)
        self.add_to_selection(selection, "HLTMuonOr", events.HLTMuonOr)

        # apply some weights,  MC/data check beforehand
        if not process.is_data:
            weights.add("x_sec", process.xsecs[13.0].nominal)

            # some weights have more than weight, not always consistent
            # take only first weight, since everything with more than 1 lep gets ejected

            weights.add(
                "LeptonSFTrigger",
                events.LeptonSFTrigger[:, 0],
                weightUp=events.LeptonSFTriggerUp[:, 0],
                weightDown=events.LeptonSFTriggerDown[:, 0],
            )

            weights.add(
                "LeptonSFIsolation",
                events.LeptonSFIsolation[:, 0],
                weightDown=events.LeptonSFIsolationDown[:, 0],
                weightUp=events.LeptonSFIsolationUp[:, 0],
            )

            weights.add(
                "LeptonSFMVA",
                events.LeptonSFMVA[:, 0],
                weightDown=events.LeptonSFMVADown[:, 0],
                weightUp=events.LeptonSFMVAUp[:, 0],
            )

            weights.add(
                "LeptonSFGSF",
                events.LeptonSFGSF[:, 0],
                weightDown=events.LeptonSFGSFDown[:, 0],
                weightUp=events.LeptonSFGSFUp[:, 0],
            )

            weights.add(
                "LeptonSFId",
                events.LeptonSFId[:, 0],
                weightDown=events.LeptonSFIdDown[:, 0],
                weightUp=events.LeptonSFIdUp[:, 0],
            )

            weights.add(
                "nISRWeight_Mar17",
                events.nISRWeight_Mar17,
                weightDown=events.nISRWeightDown_Mar17,
                weightUp=events.nISRWeightDown_Mar17,
            )

            # weights.add(
            # 'PileUpWeight',
            # events.PileUpWeight[:,0],
            # weightDown=events.PileUpWeightMinus[:,0],
            # weightUp=events.PileUpWeightPlus[:,0],
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

        common = ["baseline_selection", "lep_selection", "HLTLeptonOr", "HLTMETOr"]

        signalRegion = events.signalRegion == 1
        controlRegion = events.signalRegion == 0
        delta_phi = Dphi > 0.9
        # from IPython import embed;embed()

        self.add_to_selection(selection, "signalRegion", signalRegion)
        self.add_to_selection(selection, "controlRegion", controlRegion)
        self.add_to_selection(selection, "delta_phi", delta_phi)
        # if you need to add more options
        # signal_region = ["signalRegion"]
        # control_region = ["controlRegion"]

        categories = dict(
            # N0b_SR=common + ["zero_b", "signalRegion"],
            N1b_SR=common + ["multi_b", "signalRegion"],
            # N0b_CR=common + ["zero_b", "controlRegion"],
            N1b_CR=common + ["multi_b", "controlRegion"],
        )

        return locals()


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
        out = self.select(events)
        weights = out["weights"]

        for var_name in self.variables().names():
            for cat in out["categories"].keys():
                weight = weights.weight()
                # value = out[var_name]
                # generate blank mask for variable values
                mask = np.ones(len(out[var_name]), dtype=bool)

                # combine cuts together: problem, some have None values
                for cut in out["categories"][cat]:
                    # rint(cut, "\n")
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
                values[var_name] = out[var_name][mask]
                # weight = weights.weight()[cut]
                values["weight"] = weight[mask]
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
