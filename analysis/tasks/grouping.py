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
from tasks.base import HTCondorWorkflow

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

            # float 16 so arrays can be saved
            full_arr = np.concatenate(cat_list, dtype=np.float16)
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
                print(cut, "        ", mu_N0b[i], "      ", ratio[i])
            # index = np.max(np.nonzero(mu_N0b))


class YieldsFromArrays(CoffeaTask):
    channel = luigi.ListParameter(default=["Muon", "Electron"])

    def requires(self):
        return MergeArrays.req(self)

    def output(self):
        return {cat: self.local_target("merged_{}.npy".format(cat)) for cat in self.config_inst.categories.names()}

    def output(self):
        return self.local_target("cutflow.json")

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        #np_0b = self.input()["No_cuts"].load()
        np_0b = self.input()["Min_cuts"].load()

        var_names = self.config_inst.variables.names()
        print(var_names)

        Dphi = np_0b[:, var_names.index("dPhi")]
        LT = np_0b[:, var_names.index("LT")]
        HT = np_0b[:, var_names.index("HT")]
        n_jets = np_0b[:, var_names.index("nJets")]
        jetPt_2 = np_0b[:, var_names.index("jetPt_2")]

        cut_list = [("HT", ">2500"),("jetPt_2", ">80"),  ("nJets", ">7"), ("LT", ">50"), ("LT", "<500")
        #,('ghost_muon_filter', "==1"),("iso_cut", "==1"), ('hard_lep', "==1"), ('selected', "==1"), ("no_veto_lepton", "==1"), ('HLT_Or', "==1"), ('doubleCounting_XOR', "==1")
        #("LT", "> 250"), ("LT", "> 500"), ("LT", "> 1000"), ("LT", "> 1500"), ("LT", "> 2000"), ("LT", "> 2500"), ("LT", "> 5000")]
        #[("nJets", ">2"), ("nJets", ">3"), ("nJets", ">4"), ("nJets", ">5"), ("nJets", ">6"), ("nJets", ">7"), ("nJets", ">8"), ("nJets", ">9")]
        #[("HT", "> 2500"), ("LT", ">500"), ("nJets", ">2"), ("leadMuonPt", ">25")]
        ]

        yields = {}

        mask = np.full(len(np_0b), True)
        print("\nEntry point:", len(np_0b))
        for cut in cut_list:
            yiel = eval("np.sum(np_0b[:, var_names.index('{0}')] {1})".format(cut[0], cut[1]))
            mask = mask & eval("np_0b[:, var_names.index('{0}')] {1}".format(cut[0], cut[1]))
            ratio = np.round(np.sum(mask) / len(np_0b), 3)
            print(" ".join(cut), ":", yiel, np.sum(mask), ratio)
            yields.update({" ".join(cut): [float(yiel), float(np.sum(mask)), float(ratio)]})

        print("Remaining events:", np.sum(mask), "\n")
        self.output().dump(yields)
