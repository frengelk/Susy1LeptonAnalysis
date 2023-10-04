"""
This is the base class for the coffea Task
Here we apply our selection and define the categories used for the analysis
Also this write our arrays
"""

import numpy as np
import awkward as ak
from coffea import hist, processor

from coffea.processor.accumulator import (
    dict_accumulator,
    defaultdict_accumulator,
    column_accumulator,
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
            # cutflow = bh.Histogram(bh.axis.Regular(20, 0, 20)),
            cutflow=hist.Hist("Counts", self.dataset_axis, self.category_axis, self.category_axis, hist.Bin("cutflow", "Cut", 20, 0, 20)),
            n_minus1=hist.Hist("Counts", self.dataset_axis, self.category_axis, self.category_axis, hist.Bin("Nminus1", "Cut", 20, 0, 20)),
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

    def get_base_variable(self, events):
        ntFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagTvsQCD), value=-999)
        nWFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagWvsQCD), value=-999)
        jetMass_1 = ak.fill_none(ak.firsts(events.JetMass[:, 0:1]), value=-999)
        jetPt_1 = ak.fill_none(ak.firsts(events.JetPt[:, 0:1]), value=-999)
        jetEta_1 = ak.fill_none(ak.firsts(events.JetEta[:, 0:1]), value=-999)
        jetPhi_1 = ak.fill_none(ak.firsts(events.JetPhi[:, 0:1]), value=-999)
        jetMass_2 = ak.fill_none(ak.firsts(events.JetMass[:, 1:2]), value=-999)
        jetPt_2 = ak.fill_none(ak.firsts(events.JetPt[:, 1:2]), value=-999)
        jetEta_2 = ak.fill_none(ak.firsts(events.JetEta[:, 1:2]), value=-999)
        jetPhi_2 = ak.fill_none(ak.firsts(events.JetPhi[:, 1:2]), value=-999)
        nJets = events.nJet
        LT = events.LT
        HT = events.HT
        metPt = events.MetPt
        WBosonMt = events.WBosonMt
        # doing abs to stay in sync with old plots
        dPhi = abs(events.DeltaPhi)
        nbJets = events.nDeepJetMediumBTag
        # name with underscores lead to problems
        zerob = nbJets == 0
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
        locals().update(self.get_base_variable(events))
        # if events.metadata["isFastSim"]:
        #    locals().update(self.get_gen_variable(events))
        locals().update(self.get_electron_variables(events))
        locals().update(self.get_muon_variables(events))
        sortedJets = ak.mask(events.JetPt, (events.nJet >= 3))
        goodJets = (events.JetPt > 30) & (abs(events.JetEta) < 2.4)
        # iso_track = ((events.IsoTrackPt > 10) & (((events.IsoTrackMt2 < 60) & events.IsoTrackIsHadronicDecay) | ((events.IsoTrackMt2 < 80) & ~(events.IsoTrackIsHadronicDecay))))
        # iso_track_cut = ak.sum(iso_track, axis=-1) == 0

        # define selection
        selection = processor.PackedSelection()
        # Baseline PreSelection, split up now
        # baselineSelection = (sortedJets[:, 1] > 80) & (events.LT > 250) & (events.HT > 500) & (ak.num(goodJets) >= 3) & (~events.IsoTrackVeto)
        # subleading_jet = sortedJets[:, 1] > 80
        # from IPython import embed; embed()
        subleading_jet = ak.fill_none(ak.firsts(events.JetPt[:, 1:2] > 80), False)

        # doing lep selection
        mu_pt = ak.fill_none(ak.firsts(events.MuonPt[:, 0:1]), -999)
        e_pt = ak.fill_none(ak.firsts(events.ElectronPt[:, 0:1]), -999)
        mu_eta = ak.fill_none(ak.firsts(events.MuonEta[:, 0:1]), -999)
        e_eta = ak.fill_none(ak.firsts(events.ElectronEta[:, 0:1]), -999)
        mu_id = ak.fill_none(ak.firsts(events.MuonMediumId[:, 0:1]), False)
        e_id = ak.fill_none(ak.firsts(events.ElectronTightId[:, 0:1]), False)

        hard_lep = ((mu_pt > 25) | (e_pt > 25)) & ((abs(mu_eta) < 2.4) | (abs(e_eta) < 2.4))
        selected = (mu_id | e_id) & ((events.nGoodMuon == 1) | (events.nGoodElectron == 1))
        no_veto_lepton = (events.nVetoMuon - events.nGoodMuon == 0) & (events.nVetoElectron - events.nGoodElectron == 0)

        # njet_cut = ak.num(goodJets) >= 3
        njet_cut = ak.sum(events.JetIsClean, axis=1) >= 3
        iso_cut = ~events.IsoTrackVeto

        # stitch ttbar at events.LHE_HTIncoming < 600
        if events.metadata["dataset"] == "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8" or events.metadata["dataset"] == "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8":
            LHE_HT_cut = events.LHE_HTIncoming < 600
            # plug it on onto iso_cut, so cutflow is consistent
            iso_cut = iso_cut & LHE_HT_cut

        if events.metadata["isSignal"]:
            if proc.has_aux("masspoint"):
                mGlu_cut = events.mGluino == proc.aux["masspoint"][0]
                mNeu_cut = events.mNeutralino == proc.aux["masspoint"][1]
            else:
                mGlu_cut = events.mGluino % 5 == 0
                mNeu_cut = events.mNeutralino % 5 == 0
            iso_cut = iso_cut & (mGlu_cut) & (mNeu_cut)
            # from IPython import embed; embed()

        # require correct lepton IDs, applay cut depending on tree name
        # ElectronIdCut = ak.fill_none(ak.firsts(events.ElectronTightId[:, 0:1]), False)
        # MuonIdCut = ak.fill_none(ak.firsts(events.MuonMediumId[:, 0:1]), False)
        # prevent double counting in data
        doubleCounting_XOR = (not events.metadata["isData"]) | ((events.metadata["PD"] == "isSingleElectron") & events.HLT_EleOr) | ((events.metadata["PD"] == "isSingleMuon") & events.HLT_MuonOr & ~events.HLT_EleOr) | ((events.metadata["PD"] == "isMet") & events.HLT_MetOr & ~events.HLT_MuonOr & ~events.HLT_EleOr)
        # HLT Combination
        HLT_Or = (not events.metadata["isData"]) | (events.HLT_MuonOr | events.HLT_MetOr | events.HLT_EleOr)
        # ghost muon filter
        ghost_muon_filter = events.MetPt / events.CaloMET_pt <= 5
        common = ["baselineSelection", "doubleCounting_XOR", "HLT_Or"]  # , "{}IdCut".format(events.metadata["treename"])]
        # data cut for control plots
        data_cut = (events.LT > 250) & (events.HT > 500) & (ak.num(goodJets) >= 3)
        # skim_cut = (events.LT > 150) & (events.HT > 350)
        # triggers = [
        # "HLT_Ele115_CaloIdVT_GsfTrkIdT",
        # "HLT_Ele15_IsoVVVL_PFHT450",
        # "HLT_Ele35_WPTight_Gsf",
        # "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
        # "HLT_EleOr",
        # "HLT_IsoMu27",
        # "HLT_MetOr",
        # "HLT_Mu15_IsoVVVL_PFHT450",
        # "HLT_Mu50",
        # "HLT_MuonOr",
        # "HLT_PFMET100_PFMHT100_IDTight",
        # "HLT_PFMET110_PFMHT110_IDTight",
        # "HLT_PFMET120_PFMHT120_IDTight",
        # "HLT_PFMETNoMu100_PFMHTNoMu100_IDTight",
        # "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight",
        # "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
        # ]

        # for trig in triggers:
        #    locals().update({trig: events[trig]})
        # self.add_to_selection(selection, trig, events[trig])

        # MET_Filter = "HLT_PFMET100_PFMHT100_IDTight | HLT_PFMET110_PFMHT110_IDTight | HLT_PFMET120_PFMHT120_IDTight | HLT_PFMETNoMu100_PFMHTNoMu100_IDTight | HLT_PFMETNoMu110_PFMHTNoMu110_IDTight |HLT_PFMETNoMu120_PFMHTNoMu120_IDTight"
        # METFilter = eval(MET_Filter)        # apply weights,  MC/data check beforehand
        weights = processor.Weights(size, storeIndividual=self.individal_weights)
        if not events.metadata["isData"]:
            weights.add("Luminosity", events.metadata["Luminosity"])
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

            if events.metadata["treename"] == "Muon":
                sfs = ["MuonMediumSf", "MuonTriggerSf", "MuonMediumIsoSf"]
                for sf in sfs:
                    weights.add(
                        sf,
                        getattr(events, sf)[:, 0],
                        weightDown=getattr(events, sf + "Down")[:, 0],
                        weightUp=getattr(events, sf + "Up")[:, 0],
                    )

            if events.metadata["treename"] == "Electron":
                sfs = ["ElectronTightSf", "ElectronRecoSf"]
                for sf in sfs:
                    weights.add(
                        sf,
                        getattr(events, sf)[:, 0],
                        weightDown=getattr(events, sf + "Down")[:, 0],
                        weightUp=getattr(events, sf + "Up")[:, 0],
                    )

            # weights.add(
            # "nISRWeight_Mar17",
            # events.nISRWeight_Mar17,
            # weightDown=events.nISRWeightDown_Mar17,
            # weightUp=events.nISRWeightDown_Mar17,
            # )

            weights.add(
                "PileUpWeight",
                events.PileUpWeight,
                weightDown=events.PileUpWeightDown,
                weightUp=events.PileUpWeightUp,
            )

            # weights.add(
            # 'JetDeepJetMediumSf',
            # events.JetDeepJetMediumSf[:,0],
            # )

            # weights.add("JetMediumCSVBTagSF", events.JetMediumCSVBTagSF,
            # weightUp = events.JetMediumCSVBTagSFUp,
            # weightDown= events.JetMediumCSVBTagSFDown,
            # )

        for cat in self.config.categories:
            for cut in cat.get_aux("cuts"):
                if cut[1] == "cut":
                    self.add_to_selection(selection, " ".join(cut), eval(cut[0]))
                else:
                    self.add_to_selection(selection, " ".join(cut), eval("events.{} {}".format(cut[0], cut[1])))

        # categories = dict(N0b=common + ["zerob"], N1ib=common + ["multib"])  # common +
        categories = {cat.name: [" ".join(cut) for cut in cat.get_aux("cuts")] for cat in self.config.categories}
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
        weights = selected_output["weights"].weight()
        output = selected_output["summary"]
        arrays = self.get_selection_as_np(selected_output)
        # setting weights as extra axis in arrays
        # arrays.setdefault("weights", np.stack([np.full_like(weights, 1), weights], axis=-1))
        arrays.setdefault("weights", weights)
        if self.dtype:
            arrays = {key: array.astype(self.dtype) for key, array in arrays.items()}
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
                for cut in selected_output["categories"][cat][:1]:  # FIXME
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
