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
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from utils.CoffeaBase import *
from utils.SignalRegions import signal_regions_0b
from tasks.makefiles import WriteFileset, WriteDatasets


class CoffeaTask(DatasetTask):
    """
    token task to define attributes
    """

    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter(default=False)
    debug_dataset = Parameter(
        default="data_mu_C"
    )  # take a small set to reduce computing time
    debug_str = Parameter(
        default="/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root"
    )
    file = Parameter(
        default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root"
        # "/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root"
    )
    job_number = IntParameter(default=1)
    data_key = Parameter(default="SingleMuon")
    # parameter with selection we use coffea
    lepton_selection = Parameter(default="Muon")
    job_dict = {
        "SingleElectron": 244,
        "MET": 285,
        "SingleMuon": 424,
    }


class CoffeaProcessor(
    CoffeaTask, HTCondorWorkflow, law.LocalWorkflow
):  # AnalysisTask):

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
        return WriteDatasets.req(self)
        # return WriteFileset.req(self)

    def create_branch_map(self):
        if self.debug:
            return list(range(1))
        return list(range(self.job_dict[self.data_key]))  # self.job_number

    def output(self):
        datasets = self.config_inst.datasets.names()
        if self.debug:
            datasets = [self.debug_dataset]
        # out = {
        # cat + "_" + dat: self.local_target(cat + "_" + dat + ".npy")
        # for dat in datasets
        # for cat in self.config_inst.categories.names()
        # }
        out = {
            "job_{}_{}".format(job, cat): self.local_target(
                "job_{}_{}.npy".format(job, cat)
            )
            for job in range((self.job_dict[self.data_key]))
            for cat in self.config_inst.categories.names()
        }
        # overwrite array export logic if we want to histogram
        if self.processor == "Histogramer":
            out = self.local_target("hists.coffea")
        return out

    def store_parts(self):
        parts = (
            self.analysis_choice,
            self.processor,
            self.data_key,
            self.lepton_selection,
        )
        if self.debug:
            parts += (
                "debug",
                self.debug_dataset,
                self.debug_str.split("/")[-1].replace(".root", ""),
            )
        return super(CoffeaProcessor, self).store_parts() + parts

    @law.decorator.timeit(publish_message=True)
    def run(self):
        # FIXME
        # with open(self.input().path, "r") as read_file:
        # fileset = json.load(read_file)
        data_dict = self.input()[
            "dataset_dict"
        ].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["dataset_path"].load()
        # fileset = {
        # (dsname, shift): [target.path for target in files.targets if target.exists()]
        # for (dsname, shift), files in fileset.items()
        # if self.config_inst.datasets.has(dsname)
        # and (processor_inst.dataset_shifts or shift == "nominal")
        # }
        # # remove empty datasets
        # empty = [key for key, paths in fileset.items() if not paths]
        # if empty:
        # for e in empty:
        # del fileset[e]
        # logger.warning("skipping empty datasets: %s", ", ".join(map(str, sorted(empty))))

        # declare professor
        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter(self, Lepton=self.lepton_selection)
        if self.processor == "Histogramer":
            processor_inst = Histogramer(self, self.lepton_selection)

        # unneccessary for now, good pdractice if we want to add utility later
        lepton_dict = {
            "SingleElectron": "data_e_",
            "MET": "data_MET_",
            "SingleMuon": "data_mu_",
        }

        # building together the respective strings to use for the coffea call
        treename = self.lepton_selection
        subset = sorted(data_dict[self.data_key])
        dataset = (
            lepton_dict[self.data_key]
            + subset[self.branch].split("Run" + self.year)[1][0]
        )
        with up.open(data_path + "/" + subset[self.branch]) as file:
            # data_path + "/" + subset[self.branch]
            primaryDataset = file["MetaData"]["primaryDataset"].array()[0]
            isData = file["MetaData"]["IsData"].array()[0]
        fileset = {
            dataset: {
                "files": [data_path + "/" + subset[self.branch]],
                "metadata": {
                    "PD": primaryDataset,
                    "IsData": isData,
                },
            }
        }
        if self.debug:
            from IPython import embed

            # embed()
            # fileset = {self.debug_dataset: [fileset[self.debug_dataset][0]]}
            # fileset = {self.debug_dataset: [self.debug_str]
            # self.branch =11
            with up.open(data_path + "/" + subset[self.branch]) as file:
                # data_path + "/" + subset[self.branch]
                primaryDataset = file["MetaData"]["primaryDataset"].array()[0]
            fileset = {
                dataset: {
                    "files": [data_path + "/" + subset[self.branch]],
                    "metadata": {"PD": primaryDataset},
                }
            }

        # , metrics
        tic = time.time()
        # call imported processor, magic happens here
        print(fileset)
        out = processor.run_uproot_job(
            fileset,
            treename=treename,
            processor_instance=processor_inst,
            # pre_executor=processor.futures_executor,
            # pre_args=dict(workers=32),
            executor=processor.iterative_executor,
            executor_args=dict(
                status=False
            ),  # desc="", unit="Trolling"), # , desc="Trolling"
            # metadata_cache = 'MetaData',
            # schema=BaseSchema,),
            chunksize=10000,
        )
        # executor_args=dict(
        # nano=True,
        # savemetrics=True,
        # xrootdtimeout=30,
        # align_clusters=True,
        # processor_compression=None,
        # **ea,
        # ),

        # show summary
        console = Console()
        all_events = out["n_events"]["sum_all_events"]
        toc = time.time()
        # print(np.round(toc - tic, 2), "s")

        total_time = toc - tic
        console.print("\n[u][bold magenta]Summary metrics:[/bold magenta][/u]")
        console.print(f"* Total time: {total_time:.2f}s")
        console.print(f"* Total events: {all_events:e}")
        console.print(f"* Events / s: {all_events/total_time:.0f}")

        # save outputs
        # seperated for processor, both need different touch calls
        if self.processor == "ArrayExporter":
            self.output().popitem()[1].parent.touch()
            for cat in out["arrays"]:
                self.output()[
                    "job_{}_{}".format(self.branch, cat.split("_data")[0])
                ].dump(out["arrays"][cat]["hl"].value)
                # self.output()[cat].dump(out["arrays"][cat]["hl"].value)
            # hacky way of defining if task is done FIXME
            # self.output().dump(np.array([1]))

        if self.processor == "Histogramer":
            self.output().parent.touch()
            self.output().dump(out["histograms"])


class SubmitCoffeaPerDataset(CoffeaTask):  # , HTCondorWorkflow , law.LocalWorkflow
    # def create_branch_map(self):
    # """
    # Jobs on 2023/01/18
    # SingleElectron 240
    # MET 497
    # SingleMuon 464
    # """
    # # return a job for every dataset that has to be processed
    # return list(
    # range(self.job_number)
    # )  # by hand for now, same length as list of good files

    def requires(self):
        return WriteDatasets.req(self)

    def output(self):
        # return self.local_target("joblist_{}.json".format(self.branch))
        return self.local_target("joblist.json")

    def store_parts(self):
        return super(SubmitCoffeaPerDataset, self).store_parts() + (
            self.analysis_choice,
            self.processor,
            # self.dataset,
            self.file.split("/")[-1].replace(".root", ""),
        )

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        print("\nIn Submit\n")

        # goal of this task is to fill a dict with all the coffea locations
        # dataset = Parameter(default="data_e_C")

        joblist = {}

        # test_file = "/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root"
        # test_file = "/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root"

        # test_dict = {"TTJets_sl_fromt": [test_file]}
        test_dict = self.input()[
            "dataset_dict"
        ].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["dataset_path"].load()
        job_number_dict = self.input()["job_number_dict"].load()

        # at some point, we have to select which dataset to process
        # good_file_numbers = ["105", "108", "137", "138", "139"]

        # doing a loop for each small file on the naf
        for key in self.job_dict.keys():
            for sel in ["Muon", "Electron"]:
                # if key != "SingleMuon":
                #    continue
                # for file in sorted(test_dict[key]):
                # file = sorted(test_dict[key])[self.branch]
                # if not "105" in file and not "106" in file:
                # if not good_file_numbers[self.branch] in file:
                #    continue
                # if not "Run2017C" in file:
                #    continue
                print("loopdaloop")
                # defining how to convert dataset names
                # data_dict = {
                #    'SingleMuon' : "data_mu_C"
                # }
                # FIXME: building corresponding dataset per filename
                # dataset = "data_mu_" + file.split("Run" + self.year)[1][0]
                # define coffea Processor instance for this one dataset
                # from IPython import embed; embed()
                cof_proc = CoffeaProcessor.req(
                    self,
                    processor=self.processor,  # "ArrayExporter",
                    # job_number=job_number_dict[key],
                    data_key=key,
                    lepton_selection=sel,
                    # debug=True,
                    # debug_dataset=dataset,  # data_dict[key],
                    # debug_str=data_path + "/" + file,
                    # no_poll=True,  #just submit, do not initiate status polling
                    # workflow="local",
                )
                # find output of the coffea processor
                out_target = cof_proc.localize_output().args[0]["collection"].targets
                # out_target = cof_proc.localize_output().args[0]

                new_target = {}
                # unpack Localfiletargers, since json dump wont work otherwise
                if self.processor == "ArrayExporter":
                    # for i in range(len(out_target)):
                    for path in out_target[0].keys():
                        # FIXME
                        new_target[path] = out_target[0][path].path

                # if self.processor == "Histogramer":
                # out_target = out_target.path

                joblist.update({key + "_" + sel: new_target})

                # generates new graph at runtime
                # test = yield cof_proc

                # and lets submit this job
                # run = cof_proc.run()
                print("running branch for file:", key, sel, cof_proc)
                # self.output().dump(joblist)
                test = yield cof_proc
        self.output().dump(joblist)
        # with open(self.output().path, "w") as file:
        #    json.dump(joblist, file)


class CollectCoffeaOutput(CoffeaTask):
    def requires(self):

        # return SubmitCoffeaPerDataset.req(
        # self,
        # dataset=self.debug_dataset,
        # processor=self.processor,
        # job_number=self.job_number,
        # )
        return {
            "{}_{}".format(sel, dat): CoffeaProcessor.req(
                self,
                data_key=dat,
                lepton_selection=sel,
                job_number=self.job_dict[dat],
                # workflow="local",
            )
            for sel in ["Muon", "Electron"]
            for dat in ["SingleMuon", "MET", "SingleElectron"]
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
            tot_events, signal_events = 0, 0
            np_dict = value["collection"].targets[0]
            # different key for each file, we ignore it for now, only interested in values
            for i in tqdm(range(len(np_dict) // 2)):
                # dataset = "data_mu_" + key.split("Run" + self.year)[1][0]

                np_0b = np.load(np_dict["job_{}_N0b".format(i)].path)
                # np_1ib = np.load(value["N1ib_" + dataset])

                Dphi = np_0b[:, var_names.index("Dphi")]
                LT = np_0b[:, var_names.index("LT")]
                HT = np_0b[:, var_names.index("HT")]
                n_jets = np_0b[:, var_names.index("n_jets")]

                for ke in signal_regions_0b.keys():
                    signal_bin_counts[ke] += len(
                        np_0b[
                            eval(signal_regions_0b[ke][0])
                            & eval(signal_regions_0b[ke][1])
                            & eval(signal_regions_0b[ke][2])
                            & eval(signal_regions_0b[ke][3])
                        ]
                    )

                # at some point, we have to define the signal regions
                LT1_nj5 = np_0b[
                    (LT > 250) & (LT < 450) & (Dphi > 1) & (HT > 500) & (n_jets == 5)
                ]
                LT1_nj67 = np_0b[
                    (LT > 250)
                    & (LT < 450)
                    & (Dphi > 1)
                    & (HT > 500)
                    & (n_jets > 5)
                    & (n_jets < 8)
                ]
                LT1_nj8i = np_0b[
                    (LT > 250) & (LT < 450) & (Dphi > 1) & (HT > 500) & (n_jets > 7)
                ]

                LT2_nj5 = np_0b[
                    (LT > 450) & (LT < 650) & (Dphi > 0.75) & (HT > 500) & (n_jets == 5)
                ]
                LT2_nj67 = np_0b[
                    (LT > 450)
                    & (LT < 650)
                    & (Dphi > 0.75)
                    & (HT > 500)
                    & (n_jets > 5)
                    & (n_jets < 8)
                ]
                LT2_nj8i = np_0b[
                    (LT > 450) & (LT < 650) & (Dphi > 0.75) & (HT > 500) & (n_jets > 7)
                ]

                LT3_nj5 = np_0b[(LT > 650) & (Dphi > 0.75) & (HT > 500) & (n_jets == 5)]
                LT3_nj67 = np_0b[
                    (LT > 650)
                    & (Dphi > 0.75)
                    & (HT > 500)
                    & (n_jets > 5)
                    & (n_jets < 8)
                ]
                LT3_nj8i = np_0b[(LT > 650) & (Dphi > 0.75) & (HT > 500) & (n_jets > 7)]

                signal_events += (
                    len(LT1_nj5)
                    + len(LT1_nj67)
                    + len(LT1_nj8i)
                    + len(LT2_nj5)
                    + len(LT2_nj67)
                    + len(LT2_nj8i)
                    + len(LT3_nj5)
                    + len(LT3_nj67)
                    + len(LT3_nj8i)
                )

                tot_events += len(np_0b)

            count_dict = {
                key: {
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


# class CollectCoffeaOutput(CoffeaTask):
# def requires(self):
# # return SubmitCoffeaPerDataset.req(
# # self,
# # dataset=self.debug_dataset,
# # processor=self.processor,
# # job_number=self.job_number,
# # )
# return {
# "{}_{}".format(sel, dat) :
# CoffeaProcessor.req(
# self,
# data_key = dat,
# lepton_selection= sel,
# job_number =1,#self.job_dict[dat],
# workflow="local",
# )
# for sel in ["Muon", "Electron"]
# for dat in ["SingleMuon", "MET", "SingleElectron"]
# }
#
# def output(self):
# return self.local_target("event_counts.json")
#
# def store_parts(self):
# return super(CollectCoffeaOutput, self).store_parts() + (self.analysis_choice,)
#
# @law.decorator.timeit(publish_message=True)
# @law.decorator.safe_output
# def run(self):
# in_dict = self.input()#["collection"].targets
#
# # making clear which index belongs to which variable
# var_names = self.config_inst.variables.names()
# print(var_names)
# event_counts = {}
# # iterate over the indices for each file
# for key, value in in_dict.items():
# tot_events, signal_events = 0, 0
# np_dict = value["collection"].targets[0]
# # different key for each file, we ignore it for now, only interested in values
# for i in tqdm(range(len(np_dict)//2)):
#
# np_0b = np.load(np_dict["job_{}_N0b".format(i)].path)
#
# LT = np_0b[:, var_names.index("LT")]
# HT = np_0b[:, var_names.index("HT")]
# n_jets = np_0b[:, var_names.index("n_jets")]
#
# tot_events += len(np_0b)
# signal_events += len(np_0b[(LT > 250) & (HT > 500) & (n_jets > 4)])
#
# count_dict = {key: {
# "tot_events": tot_events,
# "signal_events": signal_events,
# }}
# print(count_dict)
# event_counts.update(count_dict)
# #print(event_counts)ä
# self.output().dump(event_counts)
# #from IPython import embed; embed()


class GroupCoffeaProcesses(DatasetTask):

    """
    Task to group coffea hist together if needed (e.g. get rid of an axis)
    Or reproduce root files for Combine
    """

    # histogram naming template:
    template = "{variable}_{process}_{category}"

    def requires(self):
        return CoffeaProcessor.req(self, processor="Histogramer")

    def output(self):
        return self.local_target("legacy_hists.root")
        # return dict(
        #    coffea=self.local_target("legacy_hists.coffea"),
        #    root=self.local_target("legacy_hists.root"),
        # )

    def store_parts(self):
        return super(GroupCoffeaProcesses, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):

        import uproot3 as up3

        # needed for newtrees
        # hists = coffea.util.load(self.input().path)
        hists = self.input()["collection"][0].load()

        datasets = self.config_inst.datasets.names()
        categories = self.config_inst.categories.names()

        # create dir and write coffea to root hists
        self.output().parent.touch()
        with up3.recreate(self.output().path) as root_file:
            var_keys = hists.keys()
            # var1='METPt'
            for var in var_keys:
                for dat in datasets:
                    for cat in categories:

                        # hacky way to get the hidden array of dict values
                        # arr=list(hists[var1][('TTJets_sl_fromt','N1b_CR')].values())[0]

                        # root_file[categories[0]] = up3.newtree({self.template.format(variable=var1, process=datasets[0], category=categories[0]):np.float64})
                        # root_file[categories[0]].extend({self.template.format(variable=var1, process=datasets[0], category=categories[0]):arr})

                        hist_name = self.template.format(
                            variable=var, process=dat, category=cat
                        )
                        if "data" in dat:
                            hist_name = self.template.format(
                                variable=var, process="data_obs", category=cat
                            )

                        # from IPython import embed;embed()
                        root_file[hist_name] = coffea.hist.export1d(
                            hists[var][(dat, cat)].project(var)
                        )

                        # cutflow variable may have to be an exception
