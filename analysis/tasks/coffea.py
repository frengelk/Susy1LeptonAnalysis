# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
from luigi import BoolParameter, Parameter
from coffea import processor
from coffea.nanoevents import TreeMakerSchema, BaseSchema, NanoAODSchema
from coffea.nanoevents import schemas
import json
import time
import numpy as np
import uproot as up
from rich.console import Console

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from utils.CoffeaBase import *
from tasks.makefiles import WriteFileset


class CoffeaProcessor(
    DatasetTask, HTCondorWorkflow, law.LocalWorkflow
):  # AnalysisTask):
    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter(default=False)
    debug_dataset = "Data_David"
    """
    this is a HTCOndor workflow, normally it will get submitted with configurations defined
    in the htcondor_bottstrap.sh or the basetasks.HTCondorWorkflow
    If you want to run this locally, just use --workflow local in the command line
    """

    def __init__(self, *args, **kwargs):
        super(CoffeaProcessor, self).__init__(*args, **kwargs)

    # def requires(self):
    #    return WriteFileset.req(self)

    # def load_corrections(self):
    # return {corr: target.load() for corr, target in
    # self.input()["corrections"].items()}

    def output(self):
        datasets = self.config_inst.datasets.names()
        if self.debug:
            datasets = [self.debug_dataset]
        out = {
            cat + "_" + dat: self.local_target(cat + "_" + dat + ".npy")
            for dat in datasets
            for cat in ["data_david"]  # self.config_inst.categories.names()
        }
        # from IPython import embed;embed()
        if self.processor == "Histogramer":
            out = self.local_target("hists.coffea")
        return out

    def store_parts(self):
        parts = (self.analysis_choice, self.processor)
        if self.debug:
            parts += ("debug", self.debug_dataset)
        return super(CoffeaProcessor, self).store_parts() + parts

    @law.decorator.timeit(publish_message=True)
    def run(self):

        # with open(self.input().path, "r") as read_file:
        #    fileset = json.load(read_file)

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
            processor_inst = ArrayExporter(self)
        if self.processor == "Histogramer":
            processor_inst = Histogramer(self)

        tic = time.time()

        if self.debug:
            # from IPython import embed;embed()
            fileset = {
                self.debug_dataset: [
                    self.config_inst.get_dataset(self.debug_dataset).keys[0]
                ]
            }

        # , metrics
        # call imported processor, magic happens here
        out = processor.run_uproot_job(
            fileset,
            treename="Events",
            processor_instance=processor_inst,
            # pre_executor=processor.futures_executor,
            # pre_args=dict(workers=32),
            executor=processor.iterative_executor,
            # executor_args=dict(savemetrics=1,
            # executor_args=dict(
            #    #schema=processor.NanoAODSchema,),
            #    schema=schemas.BaseSchema,
            #    ),
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

        # from IPython import embed;embed()
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
                self.output()[cat].dump(out["arrays"][cat]["hl"].value)
            # hacky way of defining if task is done FIXME
            # self.output().dump(np.array([1]))

        if self.processor == "Histogramer":
            self.output().parent.touch()
            self.output().dump(out["histograms"])


class GroupCoffeaProcesses(DatasetTask):

    debug = BoolParameter(default=False)
    """
    Task to group coffea hist together if needed (e.g. get rid of an axis)
    Or reproduce root files for Combine
    """

    # histogram naming template:
    template = "{variable}_{process}_{category}"

    def requires(self):
        return CoffeaProcessor.req(self, processor="Histogramer", debug=self.debug)

    def output(self):
        return self.local_target("legacy_hists.root")
        # return dict(
        #    coffea=self.local_target("legacy_hists.coffea"),
        #    root=self.local_target("legacy_hists.root"),
        # )

    def store_parts(self):

        parts = (self.analysis_choice,)
        if self.debug:
            parts += ("debug", "debug")

        return super(GroupCoffeaProcesses, self).store_parts() + parts

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):

        import uproot3 as up3

        # from IPython import embed;embed()

        # needed for newtrees
        # hists = coffea.util.load(self.input().path)
        hists = self.input()["collection"][0].load()

        datasets = ["Data_David"]  # self.config_inst.datasets.names()
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
                        # if "data" in dat:
                            # hist_name = self.template.format(
                                # variable=var, process="data_obs", category=cat
                            # )

                        #from IPython import embed;embed()
                        root_file[hist_name] = coffea.hist.export1d(
                            hists[var][(dat, cat)].project(var)
                        )

                        # cutflow variable may have to be an exception
