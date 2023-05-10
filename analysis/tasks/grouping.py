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


class ComputeEfficiencies(CoffeaTask):
    channel = luigi.Parameter(default="Synchro")
    category = luigi.Parameter(default="N0b")

    def requires(self):
        return {
            "initial": WriteDatasetPathDict.req(self),
            "cuts": GroupCoffea.req(self),
        }
        # return WriteDatasets.req(self)

    def store_parts(self):
        return super(ComputeEfficiencies, self).store_parts() + (
            self.analysis_choice,
            self.category,
        )

    def output(self):
        return self.local_target("efficiencies.json")

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        dataset_dict = self.input()["initial"]["dataset_dict"].load()
        dataset_path = self.input()["initial"]["dataset_path"].load()
        total_count_dict = {}
        cut_count_dict = {}
        for key in dataset_dict.keys():
            tot_counts = 0
            cut_counts = 0
            for fname in dataset_dict[key]:
                with up.open(dataset_path + "/" + fname) as file:
                    values = file["cutflow_{}".format(self.channel)].values()
                    initial_value = values[0]
                    last_value = values[np.max(np.nonzero(values))]
                    tot_counts += initial_value
                    cut_counts += last_value
            total_count_dict.update({key: tot_counts})
            cut_count_dict.update({key: cut_counts})
        print(total_count_dict)
        print(cut_count_dict)
        self.output().dump(total_count_dict)
        # computing percentage
        cutflow = self.input()["cuts"]["cutflow"].load().values()
        # mu_N0b = cutflow.values()[("Single" + self.channel, self.category, self.category)]
        for key in cutflow.keys():
            print("\n", key[0], key[1])
            mu_N0b = cutflow[key]
            mu_entry_point = mu_N0b[0]  # cut_count_dict["Single" + self.channel]
            ratio = np.round(mu_N0b / mu_N0b[1], 3)
            # cuts = ["entry_point"] + self.config_inst.get_category(self.category).get_aux("cuts")
            cuts = ["entry_point"] + self.config_inst.get_category(key[1]).get_aux("cuts")
            for i, cut in enumerate(cuts):
                print(cut, mu_N0b[i], ratio[i])
            # index = np.max(np.nonzero(mu_N0b))
