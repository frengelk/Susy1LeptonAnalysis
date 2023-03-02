# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
from luigi import BoolParameter, Parameter, IntParameter
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
from tasks.makefiles import WriteDatasetPathDict


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
        # return WriteFileset.req(self)

    def create_branch_map(self):
        return list(range(1)) # FIXME arbitrary
        # return list(range(self.job_dict[self.data_key]))  # self.job_number

    def output(self):
        return {
            cat: self.local_target(cat + "_signal.npy")
            for cat in self.config_inst.categories.names()
        }

        #self.local_target("signal.npy")
        # datasets = self.config_inst.datasets.names()
        # if self.debug:
        #     datasets = [self.debug_dataset]



        #     out = {
        #         cat + "_" + dat: self.local_target(cat + "_" + dat + ".npy")
        #         for dat in datasets
        #         for cat in self.config_inst.categories.names()
        #     }
        #     # from IPython import embed;embed()
        #     if self.processor == "Histogramer":
        #         out = self.local_target("hists.coffea")
        #     return out

    def store_parts(self):
        parts = (self.analysis_choice, self.processor, self.lepton_selection)
        return super(CoffeaProcessor, self).store_parts() + parts

    @law.decorator.timeit(publish_message=True)
    def run(self):

        data_dict = self.input()["dataset_dict"].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["dataset_path"].load()
        # declare processor
        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter(self, Lepton=self.lepton_selection)
        # building together the respective strings to use for the coffea call
        treename = self.lepton_selection
        key_name = list(data_dict.keys())[0]
        subset = sorted(data_dict[key_name])
        dataset = key_name.split("_")[0]
        if dataset == "merged":
            dataset = data_path.split("/")[-2]
        if self.branch >= len(subset):
            import sys
            sys.exit()
        with up.open(data_path + "/" + subset[self.branch]) as file:
            # data_path + "/" + subset[self.branch]
            primaryDataset = "MC"  # file["MetaData"]["primaryDataset"].array()[0]
            isData = file["MetaData"]["IsData"].array()[0]
            isFastSim = file["MetaData"]["IsFastSim"].array()[0]
        fileset = {
            dataset: {
                "files": [data_path + "/" + subset[self.branch]],
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
                for key in self.output().keys():
                    if key in cat:
                        self.output()[key].dump(out["arrays"][cat]["hl"].value)
