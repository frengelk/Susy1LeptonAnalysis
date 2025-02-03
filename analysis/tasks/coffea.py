# coding: utf-8

import json
import time

import law
import law.contrib.coffea
import numpy as np
import uproot as up
from coffea import processor, hist
from luigi import BoolParameter, ListParameter, Parameter
from rich.console import Console

# other modules
from tasks.base import AnalysisTask, DatasetTask, HTCondorWorkflow
from utils.coffea_base import ArrayExporter, ArrayAccumulator
from utils.signal_regions import signal_regions_0b
from tasks.makefiles import WriteDatasetPathDict, CollectInputData, CalcBTagSF, CollectMasspoints
from utils.coffea_base import ArrayExporter


class CoffeaTask(AnalysisTask):  # DatasetTask
    """
    token task to define attributes
    shared functions between tasks also defined here
    """

    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter(default=False)
    channel = ListParameter(default=["LeptonIncl"])
    # debug_dataset = Parameter(default="data_mu_C")  # take a small set to reduce computing time
    # debug_str = Parameter(default="/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root")
    # file = Parameter(
    # default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root")
    # job_number = IntParameter(default=1)
    # data_key = Parameter(default="SingleMuon")
    # parameter with selection we use coffea
    lepton_selection = Parameter(default="Muon")
    datasets_to_process = ListParameter(default=["WJets"])
    shift = Parameter(default="nominal")

    def get_proc_list(self, datasets):
        # task to return subprocesses for list of datasets
        proc_list = []
        for dat in datasets:
            proc = self.config_inst.get_process(dat)
            if not proc.aux["isData"]:
                proc_list.extend(proc.processes.names())
            if proc.aux["isData"] or proc.processes.names() == []:
                proc_list.append(proc.name)
        return proc_list

    def load_job_dict(self):
        with open(self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + "_" + self.category + ".json")) as f:
            data_list = json.load(f)
        # redoing datalist to incorporate mass points for signal
        """
        for dat in list(data_list.keys()):
            if self.config_inst.get_aux("signal_process") in dat:
                for masspoint in self.config_inst.get_aux("signal_masspoints"):
                    new_dat = dat + "_{}_{}".format(masspoint[0], masspoint[1])
                    data_list[new_dat] = data_list[dat]
        """

        job_number = 0  # len(data_list.keys())
        job_number_dict, process_dict = {}, {}
        data_path = data_list["directory_path"]
        # start with wanted process names, than check for leafs
        for dat in self.datasets_to_process:
            # proc = self.config_inst.get_process(dat)
            proc_list = self.get_proc_list([dat])
            # for key, files in data_list.items():
            for name in proc_list:
                files = data_list[name]
                for i, file in enumerate(sorted(files)):
                    # This section is designed to catch empty files causing uproot to crash
                    # f = up.open(data_path + "/" + file)
                    # # check for empty root file
                    # if f.keys() == []:
                    # print("Empty file:", file)
                    # job_number -= 1
                    # continue
                    job_number_dict.update({job_number + i: file})
                    process_dict.update({job_number + i: name})
                job_number += len(files)
        if self.debug:
            job_number = 1
            job_number_dict = {0: job_number_dict[0]}
        return data_list, job_number, job_number_dict, process_dict

    # for Shift Arrays and Inference
    def unpack_shifts(self):
        shifts_long = []
        for sh in self.shifts:
            if sh == "systematic_shifts":
                shifts_long.extend(self.config_inst.get_aux("systematic_shifts"))
            else:
                shifts_long.append(sh)
        return shifts_long


class CoffeaProcessor(CoffeaTask, HTCondorWorkflow, law.LocalWorkflow):
    additional_plots = BoolParameter(default=False)
    # require more ressources
    RAM = 5000
    hours = 2
    """
    this is a HTCOndor workflow, normally it will get submitted with configurations defined
    in the htcondor_bottstrap.sh or the basetasks.HTCondorWorkflow
    If you want to run this locally, just use --workflow local in the command line
    Overall task to execute Coffea
    Config and actual code is found in utils
    """

    def __init__(self, *args, **kwargs):
        super(CoffeaProcessor, self).__init__(*args, **kwargs)

    def requires(self):
        # debugging does not need to have btag SF calculated.
        if self.debug:
            return {"files": WriteDatasetPathDict.req(self), "weights": CollectInputData.req(self)}
        return {"files": WriteDatasetPathDict.req(self), "weights": CollectInputData.req(self), "btagSF": CalcBTagSF.req(self, debug=False)}

    def create_branch_map(self):
        # define job number according to number of files of the dataset that you want to process
        job_number = self.load_job_dict()[1]
        return list(range(job_number))

    def output(self):
        files, job_number, job_number_dict, process_dict = self.load_job_dict()
        out = {
            cat
            + "_"
            + dat.split("/")[0]
            + "_"
            + str(job): {
                "array": self.local_target(cat + "_" + dat + "_" + str(job) + ".npy"),
                "weights": self.local_target(cat + "_" + dat + "_" + str(job) + "_weights.npy"),
                "DNNId": self.local_target(cat + "_" + dat + "_" + str(job) + "_DNNId.npy"),
                "cutflow": self.local_target(cat + "_" + dat + "_" + str(job) + "cutflow.coffea"),
                "n_minus1": self.local_target(cat + "_" + dat + "_" + str(job) + "n_minus1.coffea"),
            }
            for cat in [self.category]  # self.config_inst.categories.names()
            for job, dat in process_dict.items()
        }
        if self.shift == "systematic_shifts":
            out = {
                cat
                + "_"
                + dat.split("/")[0]
                + "_"
                + str(job): {
                    "array": self.local_target(cat + "_" + dat + "_" + str(job) + ".npy"),
                    "weights": self.local_target(cat + "_" + dat + "_" + str(job) + "_weights.npy"),
                    "DNNId": self.local_target(cat + "_" + dat + "_" + str(job) + "_DNNId.npy"),
                    "cutflow": self.local_target(cat + "_" + dat + "_" + str(job) + "cutflow.coffea"),
                    "n_minus1": self.local_target(cat + "_" + dat + "_" + str(job) + "n_minus1.coffea"),
                    "systematic_shifts": {shift: self.local_target(cat + "_" + dat + "_" + shift + "_" + str(job) + "_shifted_weights.npy") for shift in self.config_inst.get_aux("systematic_shifts")},
                }
                for cat in [self.category]  # self.config_inst.categories.names()
                for job, dat in process_dict.items()
            }
        return out

    def store_parts(self):
        parts = (self.analysis_choice, self.processor, self.lepton_selection)
        if self.debug:
            parts += ("debug",)
        if self.shift != "nominal":
            parts += (self.shift,)
        if self.additional_plots:
            parts += ("additional_plots",)
        return super(CoffeaProcessor, self).store_parts() + parts

    @law.decorator.timeit(publish_message=True)
    def run(self):
        data_dict = self.input()["files"]["dataset_dict"].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["files"]["dataset_path"].load()
        sum_gen_weights_dict = self.input()["weights"]["sum_gen_weights"].load()
        # declare processor
        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter(self, Lepton=self.lepton_selection, additional_plots=self.additional_plots)
        if self.processor == "Histogramer":
            processor_inst = Histogramer(self)
        # building together the respective strings to use for the coffea call
        files, job_number, job_number_dict, process_dict = self.load_job_dict()
        treename = self.lepton_selection
        subset = job_number_dict[self.branch]
        # dataset = subset.split("/")[0]  # split("_")[0]
        dataset = process_dict[self.branch]
        if dataset == "merged":
            dataset = data_path.split("/")[-2]
        proc = self.config_inst.get_process(dataset)
        print(subset, dataset)
        # check for empty dataset
        empty = False

        with up.open(data_path + "/" + subset) as file:
            # data_path + "/" + subset[self.branch]
            primaryDataset = file["MetaData"]["primaryDataset"].array()[0]
            # temporary fix for EGamma skim
            if "EGamma" in file["MetaData"]["SampleName"].array()[0]:
                primaryDataset = "isEGamma"
            isData = file["MetaData"]["IsData"].array()[0]
            isFastSim = file["MetaData"]["IsFastSim"].array()[0]
            if proc.is_leaf_process:
                # check parent
                isSignal = proc.parent_processes.get_first().get_aux("isSignal")
            else:
                isSignal = proc.get_aux("isSignal")
            if not isData:
                # assert all events with the same Xsec in scope with float precision
                # assert abs(np.mean(file["MetaData"]["xSection"].array()) - file["MetaData"]["xSection"].array()[0]) < file["MetaData"]["xSection"].array()[0] * 1e-5
                # FIXME
                xSec = file["MetaData"]["xSection"].array()[0]
                if xSec == 1:
                    xSec = proc.xsecs[13].nominal
                # xSec = proc.xsecs[13].nominal
                lumi = file["MetaData"]["Luminosity"].array()[0]
                # find the calculated btag SFs per file and save path
                subsub = subset.split("/")[1]
                if not self.debug:  # FIXME
                    btagSF = self.input()["btagSF"][treename + "_" + subsub]["weights"].path
                    btagSF_up = self.input()["btagSF"][treename + "_" + subsub]["up"].path
                    btagSF_down = self.input()["btagSF"][treename + "_" + subsub]["down"].path
                else:
                    btag_out = CalcBTagSF.req(self, debug=False).output()
                    btagSF = btag_out[treename + "_" + subsub]["weights"].path
                    btagSF_up = btag_out[treename + "_" + subsub]["up"].path
                    btagSF_down = btag_out[treename + "_" + subsub]["down"].path

                # print("\n max value", np.max(file["Electron"]["ElectronPt"].array()))
                # if empty skip and construct placeholder output
                if len(file[treename]["Event"].array()) == 0:
                    empty = True
                    out = {
                        "cutflow": hist.Hist("Counts", hist.Bin("cutflow", "Cut", 20, 0, 20)),
                        "n_minus1": hist.Hist("Counts", hist.Bin("Nminus1", "Cut", 20, 0, 20)),
                        "arrays": {
                            self.category + "_" + dataset: {"hl": ArrayAccumulator(np.reshape(np.array([], dtype=np.float64), (0, len(self.config_inst.variables)))), "weights": ArrayAccumulator(np.array([], dtype=np.float64)), "DNNId": ArrayAccumulator(np.array([], dtype=np.float64))},
                        },
                    }
                    if self.shift == "systematic_shifts":
                        for shift in self.config_inst.get_aux("systematic_shifts"):
                            out["arrays"][self.category + "_" + dataset][shift] = ArrayAccumulator(np.array([], dtype=np.float64))
                            # elif self.shift != "nominal":
                            #     out[self.shift]=ArrayAccumulator(np.array([], dtype=np.float64)
                            # "N1ib_" + dataset: {"hl": ArrayAccumulator(np.reshape(np.array([], dtype=np.float64), (0, len(self.config_inst.variables)))), "weights": ArrayAccumulator(np.array([], dtype=np.float64)), "DNNId": ArrayAccumulator(np.array([], dtype=np.float64))},
                # sum_gen_weight = np.sum(file["MetaData"]["SumGenWeight"].array())
            else:
                # filler values so they are defined
                xSec = 1
                lumi = 1
                btagSF = 1  # shouldn't be accessed during processing, would fail since this isn't a path
                btagSF_up = 1
                btagSF_down = 1
        fileset = {
            dataset: {
                "files": [data_path + "/" + subset],  # file for file in
                "metadata": {"PD": primaryDataset, "isData": isData, "isFastSim": isFastSim, "isSignal": isSignal, "xSec": xSec, "Luminosity": lumi, "sumGenWeight": sum_gen_weights_dict[dataset], "btagSF": btagSF, "btagSF_up": btagSF_up, "btagSF_down": btagSF_down, "shift": self.shift, "category": self.category},
            }
        }
        if not empty:
            start = time.time()
            # call imported processor, magic happens here
            out = processor.run_uproot_job(
                fileset,
                treename=treename,
                processor_instance=processor_inst,
                # pre_executor=processor.futures_executor,
                # pre_args=dict(workers=32),
                executor=processor.iterative_executor,
                executor_args=dict(status=False),  # desc="", unit="Trolling"), # , desc="Trolling"
                # metadata_cache = 'MetaData',
                # schema=BaseSchema,),
                chunksize=10000,
            )
            # show summary
            console = Console()
            all_events = out["n_events"]["sumAllEvents"]
            total_time = time.time() - start
            console.print("\n[u][bold magenta]Summary metrics:[/bold magenta][/u]")
            console.print(f"* Total time: {total_time:.2f}s")
            console.print(f"* Total events: {all_events:e}")
            console.print(f"* Events / s: {all_events/total_time:.0f}")
        # save outputs, seperated for processor, both need different touch calls
        if self.processor == "ArrayExporter":
            self.output().popitem()[1]["array"].parent.touch()
            for cat in out["arrays"]:
                self.output()[cat + "_" + str(self.branch)]["weights"].dump(out["arrays"][cat]["weights"].value)
                self.output()[cat + "_" + str(self.branch)]["DNNId"].dump(out["arrays"][cat]["DNNId"].value)
                self.output()[cat + "_" + str(self.branch)]["array"].dump(out["arrays"][cat]["hl"].value)
                self.output()[cat + "_" + str(self.branch)]["cutflow"].dump(out["cutflow"])
                self.output()[cat + "_" + str(self.branch)]["n_minus1"].dump(out["n_minus1"])
                if self.shift == "systematic_shifts":
                    for shift in self.config_inst.get_aux("systematic_shifts"):
                        self.output()[cat + "_" + str(self.branch)]["systematic_shifts"][shift].dump(out["arrays"][cat][shift].value)


class CollectCoffeaOutput(CoffeaTask):
    def requires(self):
        return {
            sel: CoffeaProcessor.req(
                self,
                lepton_selection=sel,
                # workflow="local",
            )
            for sel in ["Electron", "Muon"]
        }

    # def output(self):
    def output(self):
        return {
            "event_counts": self.local_target("event_counts.json"),
            "signal_bin_counts": self.local_target("signal_bin_counts.json"),
        }

    def store_parts(self):
        return super(CollectCoffeaOutput, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        in_dict = self.input()  # ["collection"].targets

        # making clear which index belongs to which variable
        var_names = self.config_inst.variables.names()
        print(var_names)
        # signal_events = 0
        event_counts = {}
        # initialize
        signal_bin_counts = {k: 0 for k in signal_regions_0b.keys()}
        # iterate over the indices for each file
        for key, value in in_dict.items():
            np_dict = value["collection"].targets[0]
            for dat in self.datasets_to_process:
                tot_events, signal_events = 0, 0
                # different key for each file, we ignore it for now, only interested in values
                for file, value in np_dict.items():
                    cat = "N0b"  # or loop over self.config_inst.categories.names()
                    if cat in file and dat in file:
                        np_0b = np_dict[file]["array"].load()
                        # np_1ib = np.load(value["N1ib_" + dataset])

                        Dphi = np_0b[:, var_names.index("dPhi")]
                        LT = np_0b[:, var_names.index("LT")]
                        HT = np_0b[:, var_names.index("HT")]
                        n_jets = np_0b[:, var_names.index("nJets")]

                        for ke in signal_regions_0b.keys():
                            signal_bin_counts[ke] += len(np_0b[eval(signal_regions_0b[ke][0]) & eval(signal_regions_0b[ke][1]) & eval(signal_regions_0b[ke][2]) & eval(signal_regions_0b[ke][3])])

                        # at some point, we have to define the signal regions
                        LT1_nj5 = np_0b[(LT > 250) & (LT < 450) & (Dphi > 1) & (HT > 500) & (n_jets == 5)]
                        LT1_nj67 = np_0b[(LT > 250) & (LT < 450) & (Dphi > 1) & (HT > 500) & (n_jets >= 6) & (n_jets <= 7)]
                        LT1_nj8i = np_0b[(LT > 250) & (LT < 450) & (Dphi > 1) & (HT > 500) & (n_jets >= 8)]

                        LT2_nj5 = np_0b[(LT > 450) & (LT < 650) & (Dphi > 0.75) & (HT > 500) & (n_jets == 5)]
                        LT2_nj67 = np_0b[(LT > 450) & (LT < 650) & (Dphi > 0.75) & (HT > 500) & (n_jets >= 6) & (n_jets <= 7)]
                        LT2_nj8i = np_0b[(LT > 450) & (LT < 650) & (Dphi > 0.75) & (HT > 500) & (n_jets >= 8)]

                        LT3_nj5 = np_0b[(LT > 650) & (Dphi > 0.5) & (HT > 500) & (n_jets == 5)]
                        LT3_nj67 = np_0b[(LT > 650) & (Dphi > 0.5) & (HT > 500) & (n_jets >= 6) & (n_jets <= 7)]
                        LT3_nj8i = np_0b[(LT > 650) & (Dphi > 0.5) & (HT > 500) & (n_jets >= 8)]

                        signal_events += len(LT1_nj5) + len(LT1_nj67) + len(LT1_nj8i) + len(LT2_nj5) + len(LT2_nj67) + len(LT2_nj8i) + len(LT3_nj5) + len(LT3_nj67) + len(LT3_nj8i)

                        tot_events += len(np_0b)

                        # print(signal_events)
                        itot = 0
                        for it in signal_bin_counts.values():
                            itot += it
                        # print(itot, "\n")
                count_dict = {
                    key
                    + "_"
                    + dat: {
                        "tot_events": tot_events,
                        "signal_events": signal_events,
                    }
                }
                print(count_dict)
                event_counts.update(count_dict)

        print(signal_bin_counts)
        self.output()["event_counts"].dump(event_counts)
        self.output()["signal_bin_counts"].dump(signal_bin_counts)

        vals = 0
        for key in signal_bin_counts.keys():
            vals += signal_bin_counts[key]

        print("Sum over signal region:", vals)
