# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
from luigi import BoolParameter, Parameter, IntParameter, ListParameter
from coffea import processor
from coffea.nanoevents import TreeMakerSchema, BaseSchema, NanoAODSchema
import json
import time
import numpy as np
import uproot as up
from rich.console import Console
from tqdm import tqdm

# other modules
from tasks.base import DatasetTask, HTCondorWorkflow
from utils.coffea_base import *
from tasks.makefiles import WriteDatasetPathDict, WriteDatasets


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
    job_number = IntParameter(default=1)
    data_key = Parameter(default="SingleMuon")
    # parameter with selection we use coffea
    lepton_selection = Parameter(default="Muon")
    datasets_to_process = ListParameter(default=["T5qqqqVV"])


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
        # return list(range(self.job_dict[self.data_key]))  # self.job_number

    def output(self):
        files, job_number, job_number_dict = self.load_job_dict()
        return {
            cat + "_" + dat.split("/")[0] + "_" + str(job): self.local_target(cat + "_" + dat.split("/")[0] + "_" + str(job) + ".npy")
            for cat in self.config_inst.categories.names()
            for job, dat in job_number_dict.items()
            # for i in range(job_number)  + "_" + str(job_number)
        }

    def store_parts(self):
        parts = (self.analysis_choice, self.processor, self.lepton_selection)
        return super(CoffeaProcessor, self).store_parts() + parts

    def load_job_dict(self):
        with open(self.config_inst.get_aux("job_dict")) as f:
            data_list = json.load(f)
        job_number = 0  # len(data_list.keys())
        job_number_dict = {}
        for key, files in data_list.items():
            if key not in self.datasets_to_process:
                continue
            for i, file in enumerate(sorted(files)):
                job_number_dict.update({job_number + i: file})
            job_number += len(files)
        return data_list, job_number, job_number_dict

    @law.decorator.timeit(publish_message=True)
    def run(self):
        data_dict = self.input()["dataset_dict"].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["dataset_path"].load()
        # declare processor
        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter(self, Lepton=self.lepton_selection)
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
            primaryDataset = "MC"  # file["MetaData"]["primaryDataset"].array()[0]
            isData = file["MetaData"]["IsData"].array()[0]
            isFastSim = file["MetaData"]["IsFastSim"].array()[0]
        fileset = {
            dataset: {
                "files": [data_path + "/" + subset],  # file for file in
                "metadata": {"PD": primaryDataset, "isData": isData, "isFastSim": isFastSim},
            }
        }
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
        all_events = out["n_events"]["sum_all_events"]
        total_time = time.time() - start
        console.print("\n[u][bold magenta]Summary metrics:[/bold magenta][/u]")
        console.print(f"* Total time: {total_time:.2f}s")
        console.print(f"* Total events: {all_events:e}")
        console.print(f"* Events / s: {all_events/total_time:.0f}")
        # save outputs, seperated for processor, both need different touch calls
        if self.processor == "ArrayExporter":
            self.output().popitem()[1].parent.touch()
            for cat in out["arrays"]:
                self.output()[cat + "_" + str(self.branch)].dump(out["arrays"][cat]["hl"].value)
