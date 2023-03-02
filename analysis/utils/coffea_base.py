"""
This is the base class for the coffea Task
Here we apply our selection and define the categories used for the analysis
Also this write our arrays
"""

import coffea
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
            cutflow=hist.Hist("Counts", self.dataset_axis, self.category_axis, self.category_axis, hist.Bin("cutflow", "Cut index", 10, 0, 10)),
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
        return dict(hl=np.stack([ak.to_numpy(X[var]).astype(np.float32) for var in self.config.variables.names()], axis=-1))

    def add_to_selection(self, selection, name, array):
        return selection.add(name, ak.to_numpy(array, allow_missing=True))

    def get_base_variable(self, events):
        ntFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagTvsQCD), value=0)
        nWFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagWvsQCD), value=0)
        jetMass_1 = ak.fill_none(ak.firsts(events.JetMass[:, 0:1]), value=0)
        jetPt_1 = ak.fill_none(ak.firsts(events.JetPt[:, 0:1]), value=0)
        jetEta_1 = ak.fill_none(ak.firsts(events.JetEta[:, 0:1]), value=0)
        jetPhi_1 = ak.fill_none(ak.firsts(events.JetPhi[:, 0:1]), value=0)
        jetMass_2 = ak.fill_none(ak.firsts(events.JetMass[:, 1:2]), value=0)
        jetPt_2 = ak.fill_none(ak.firsts(events.JetPt[:, 1:2]), value=0)
        jetEta_2 = ak.fill_none(ak.firsts(events.JetEta[:, 1:2]), value=0)
        jetPhi_2 = ak.fill_none(ak.firsts(events.JetPhi[:, 1:2]), value=0)
        nJets = events.nJet
        LT = events.LT
        HT = events.HT
        metPt = events.MetPt
        WBosonMt = events.WBosonMt
        dPhi = events.DeltaPhi
        nbJets = events.nDeepJetMediumBTag
        zero_b = nbJets == 0
        multi_b = nbJets >= 1
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
        return locals()

    def get_muon_variables(self, events):
        # leptons variables
        nMuon = events.nMuon
        leadMuonPt = ak.fill_none(ak.firsts(events.MuonPt[:, 0:1]), 0)
        leadMuonEta = ak.fill_none(ak.firsts(events.MuonEta[:, 0:1]), 0)
        leadMuonPhi = ak.fill_none(ak.firsts(events.MuonPhi[:, 0:1]), 0)
        # MuonMass
        muonCharge = events.MuonCharge
        muonPdgId = events.MuonPdgId
        vetoMuon = (events.MuonPt[:, 1:2] > 10) & events.MuonLooseId[:, 1:2]
        genMuonPt = events.GenMuonPt_1
        genMuonPhi = events.GenMuonPhi_1
        genMuonEta = events.GenMuonEta_1

        return locals()

    def get_electron_variables(self, events):
        # leptons variables
        nElectron = events.nElectron
        leadElectronPt = ak.fill_none(ak.firsts(events.ElectronPt[:, 0:1]), 0)
        leadElectronEta = ak.fill_none(ak.firsts(events.ElectronEta[:, 0:1]), 0)
        leadElectronPhi = ak.fill_none(ak.firsts(events.ElectronPhi[:, 0:1]), 0)
        # ElectronMass
        electronCharge = events.ElectronCharge
        electronPdgId = events.ElectronPdgId
        vetoElectron = (events.ElectronPt[:, 1:2] > 10) & events.ElectronLooseId[:, 1:2]
        genElectronPt = events.GenElectronPt_1
        genElectronPhi = events.GenElectronPhi_1
        genElectronEta = events.GenElectronEta_1

        return locals()

    def base_select(self, events):
        dataset = events.metadata["dataset"]
        dataset_obj = self.config.get_dataset(dataset)
        process_obj = self.config.get_process(dataset)

        summary = self.accumulator.identity()
        size = events.metadata["entrystop"] - events.metadata["entrystart"]
        summary["n_events"][dataset] = size
        summary["n_events"]["sumAllEvents"] = size



        # Get Variables used for Analysis and Selection
        locals().update(self.get_base_variable(events))
        if events.metadata["isFastSim"]:
            locals().update(self.get_gen_variable(events))
        locals().update(self.get_electron_variables(events))
        locals().update(self.get_muon_variables(events))
        sortedJets = ak.mask(events.JetPt, (events.nJet >= 3))
        goodJets = (events.JetPt > 30) & (abs(events.JetEta) < 2.4)
        # Baseline PreSelection
        baselineSelection = (sortedJets[:, 1] > 80) & (events.LT > 250) & (events.HT > 500) & (ak.num(goodJets) >= 3) & ~(events.IsoTrackVeto)
        # prevent double counting in data
        doubleCounting_XOR = (not events.metadata["isData"]) | ((events.metadata["PD"] == "isSingleElectron") & events.HLT_EleOr) | ((events.metadata["PD"] == "isSingleMuon") & events.HLT_MuonOr & ~events.HLT_EleOr) | ((events.metadata["PD"] == "isMet") & events.HLT_MetOr & ~events.HLT_MuonOr & ~events.HLT_EleOr)
        # define selection
        selection = processor.PackedSelection()
        self.add_to_selection(selection, "doubleCounting_XOR", doubleCounting_XOR)
        self.add_to_selection((selection), "HLT_Or", events.HLT_MuonOr | events.HLT_MetOr | events.HLT_EleOr)
        self.add_to_selection((selection), "baselineSelection", ak.fill_none(baselineSelection, False))
        X = locals()
        self.add_to_selection((selection), "zero_b", X["zero_b"])
        self.add_to_selection((selection), "multi_b", X["multi_b"])
        # apply some weights,  MC/data check beforehand
        weights = processor.Weights(size, storeIndividual=self.individal_weights)
        if not process_obj.is_data:
            weights.add("xsecs", process_obj.xsecs[13.0].nominal)
        common = ["baselineSelection", "HLT_Or"]
        categories = dict(N0b=["zero_b"], N1ib=["multi_b"])
        return locals()

    def add_to_selection(self, selection, name, array):
        return selection.add(name, ak.to_numpy(array, allow_missing=True))


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

    def __init__(self, task, Lepton):
        super().__init__(task)
        self.Lepton = Lepton

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
        print("categories:", categories)
        # weights = selected_output["weights"]
        output = selected_output["summary"]
        arrays = self.get_selection_as_np(selected_output)
        if self.dtype:
            arrays = {key: array.astype(self.dtype) for key, array in arrays.items()}
        output["arrays"] = dict_accumulator({category + "_" + selected_output["dataset"]: dict_accumulator({key: ArrayAccumulator(array[cut, ...]) for key, array in arrays.items()}) for category, cut in categories.items()})
        return output

    def postprocess(self, accumulator):
        return accumulator
