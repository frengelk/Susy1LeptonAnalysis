# coding: utf-8

import json
import logging
import os
import time

import law
import law.contrib.coffea
import numpy as np
import uproot as up
from coffea import processor
from coffea.nanoevents import BaseSchema, NanoAODSchema, TreeMakerSchema
from luigi import BoolParameter, IntParameter, ListParameter, Parameter
from rich.console import Console

# other modules
from tasks.base import DatasetTask, HTCondorWorkflow
from utils.coffea_base import ArrayExporter
from utils.signal_regions import signal_regions_0b
from tasks.makefiles import WriteDatasetPathDict, WriteDatasets
from tqdm import tqdm
from utils.coffea_base import *


class CoffeaTask(DatasetTask):
    """
    token task to define attributes
    """

    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter(default=False)
    debug_dataset = Parameter(default="data_mu_C")  # take a small set to reduce computing time
    debug_str = Parameter(default="/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root")
    file = Parameter(
        default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root"
        # "/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root"
    )
    # job_number = IntParameter(default=1)
    # data_key = Parameter(default="SingleMuon")
    # parameter with selection we use coffea
    lepton_selection = Parameter(default="Muon")
    datasets_to_process = ListParameter(default=["WJets"])


class CoffeaProcessor(CoffeaTask, HTCondorWorkflow, law.LocalWorkflow):
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
        return WriteDatasetPathDict.req(self)
        # return WriteDatasets.req(self)

    def create_branch_map(self):
        # define job number according to number of files of the dataset that you want to process
        job_number = self.load_job_dict()[1]
        return list(range(job_number))

    def output(self):
        files, job_number, job_number_dict = self.load_job_dict()
        out = {
            cat + "_" + dat.split("/")[0] + "_" + str(job): {"array": self.local_target(cat + "_" + dat.split("/")[0] + "_" + str(job) + ".npy"), "weights": self.local_target(cat + "_" + dat.split("/")[0] + "_" + str(job) + "_weights.npy"), "cutflow": self.local_target(cat + "_" + dat.split("/")[0] + "_" + str(job) + "cutflow.coffea"), "n_minus1": self.local_target(cat + "_" + dat.split("/")[0] + "_" + str(job) + "n_minus1.coffea")}
            for cat in self.config_inst.categories.names()
            for job, dat in job_number_dict.items()
            # for i in range(job_number)  + "_" + str(job_number)
        }
        return out

    def store_parts(self):
        parts = (self.analysis_choice, self.processor, self.lepton_selection)
        if self.debug:
            parts += ("debug",)
        return super(CoffeaProcessor, self).store_parts() + parts

    def load_job_dict(self):
        with open(self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + ".json")) as f:
            data_list = json.load(f)
        job_number = 0  # len(data_list.keys())
        job_number_dict = {}
        data_path = data_list["directory_path"]
        for key, files in data_list.items():
            check_list = []
            for dat in self.datasets_to_process:
                if dat in key:
                    check_list.append(True)

            # this is so ugly, I want to do
            # any(dat for dat in self.datasets_to_process if(dat in key))
            if not any(check_list):
                continue
            for i, file in enumerate(sorted(files)):
                f = up.open(data_path + "/" + file)
                # check for empty root file
                if f.keys() == []:
                    print("Empty file:", file)
                    job_number -= 1
                    continue
                job_number_dict.update({job_number + i: file})
            job_number += len(files)

        if self.debug:
            job_number = 1
            job_number_dict = {0: job_number_dict[0]}
        return data_list, job_number, job_number_dict

    @law.decorator.timeit(publish_message=True)
    def run(self):
        data_dict = self.input()["dataset_dict"].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["dataset_path"].load()
        # declare processor
        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter(self, Lepton=self.lepton_selection)
        if self.processor == "Histogramer":
            processor_inst = Histogramer(self)
        # building together the respective strings to use for the coffea call
        files, job_number, job_number_dict = self.load_job_dict()
        treename = self.lepton_selection
        # key_name = self.datasets_to_process[self.branch] # list(data_dict.keys())[0]
        subset = job_number_dict[self.branch]
        dataset = subset.split("/")[0]  # split("_")[0]
        if dataset == "merged":
            dataset = data_path.split("/")[-2]

        with up.open(data_path + "/" + subset) as file:
            # data_path + "/" + subset[self.branch]
            primaryDataset = file["MetaData"]["primaryDataset"].array()[0]
            isData = file["MetaData"]["IsData"].array()[0]
            isFastSim = file["MetaData"]["IsFastSim"].array()[0]
            if not isData:
                # all events with the same Xsec
                assert abs(np.mean(file["MetaData"]["xSection"].array()) - file["MetaData"]["xSection"].array()[0]) < 1e-9
                xSec = file["MetaData"]["xSection"].array()[0]
                lumi = file["MetaData"]["Luminosity"].array()[0]
            else:
                # filler values so they are defined
                xSec = 1
                lumi = 1
        fileset = {
            dataset: {
                "files": [data_path + "/" + subset],  # file for file in
                "metadata": {"PD": primaryDataset, "isData": isData, "isFastSim": isFastSim, "xSec": xSec, "Luminosity": lumi},
            }
        }
        print(fileset, treename)
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
                #    reweighting by sum over genweights in whole dataset
                self.output()[cat + "_" + str(self.branch)]["weights"].dump(out["arrays"][cat]["weights"].value / out["sum_gen_weights"][dataset])
                self.output()[cat + "_" + str(self.branch)]["array"].dump(out["arrays"][cat]["hl"].value)
                self.output()[cat + "_" + str(self.branch)]["cutflow"].dump(out["cutflow"])
                self.output()[cat + "_" + str(self.branch)]["n_minus1"].dump(out["n_minus1"])


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
