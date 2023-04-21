# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import boost_histogram as bh
import mplhep as hep
import coffea
from tqdm.auto import tqdm

# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask


class GroupCoffea(CoffeaTask):

    """
    Plotting cutflow produced by coffea
    Utility for doing log scale, only debug plotting
    """

    def requires(self):
        return CoffeaProcessor.req(self, processor="Histogramer")

    def output(self):
        return self.local_target("cutflow.coffea")

    def run(self):
        inp = self.input()["collection"].targets[0]
        cut0 = inp[list(inp.keys())[0]]["cutflow"].load()

        for key in list(inp.keys())[1:]:
            cut0 += inp[key]["cutflow"].load()

        self.output().dump(cut0)


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

        self.local_target("cutflow.npy")

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
