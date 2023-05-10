# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import uproot as up
import coffea


# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask
from tasks.makefiles import WriteDatasetPathDict, WriteDatasets


class GroupCoffea(CoffeaTask):

    """
    Plotting cutflow produced by coffea
    Utility for doing log scale, only debug plotting
    """

    def requires(self):
        return CoffeaProcessor.req(self, processor="Histogramer", lepton_selection=self.lepton_selection)

    def output(self):
        return {
            "cutflow": self.local_target("cutflow.coffea"),
            "n_minus1": self.local_target("n_minus1.coffea"),
        }

    def run(self):
        inp = self.input()["collection"].targets[0]
        cut0 = inp[list(inp.keys())[0]]["cutflow"].load()
        minus0 = inp[list(inp.keys())[0]]["n_minus1"].load()
        for key in list(inp.keys())[1:]:
            cut0 += inp[key]["cutflow"].load()
            minus0 += inp[key]["n_minus1"].load()

        self.output()["cutflow"].dump(cut0)
        self.output()["n_minus1"].dump(minus0)


class MergeArrays(CoffeaTask):
    channel = luigi.ListParameter(default=["Muon", "Electron"])

    def requires(self):
        return {
            sel: CoffeaProcessor.req(
                self,
                lepton_selection=sel,
                # workflow="local",
            )
            for sel in self.channel
        }

    def output(self):
        return {cat: self.local_target("merged_{}.npy".format(cat)) for cat in self.config_inst.categories.names()}

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        for cat in self.config_inst.categories.names():
            cat_list = []
            for lep in self.input().keys():
                np_dict = self.input()[lep]["collection"].targets[0]
                for key, arr in np_dict.items():
                    if cat in key:
                        cat_list.append(arr.load())

            full_arr = np.concatenate(cat_list)
            self.output()[cat].parent.touch()
            self.output()[cat].dump(full_arr)
