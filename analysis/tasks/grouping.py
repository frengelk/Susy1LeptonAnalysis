# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import uproot as up
import coffea
from tqdm import tqdm


# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask
from tasks.makefiles import WriteDatasetPathDict, WriteDatasets
from tasks.base import HTCondorWorkflow


class GroupCoffea(CoffeaTask):

    """
    Plotting cutflow produced by coffea
    """

    def requires(self):
        return CoffeaProcessor.req(self, lepton_selection=self.lepton_selection, additional_plots=True)

    def output(self):
        return {
            self.lepton_selection + "_cutflow": self.local_target(self.lepton_selection + "_cutflow.coffea"),
            self.lepton_selection + "_n_minus1": self.local_target(self.lepton_selection + "_n_minus1.coffea"),
        }

    def run(self):
        inp = self.input()["collection"].targets[0]
        cut0 = inp[list(inp.keys())[0]]["cutflow"].load()
        minus0 = inp[list(inp.keys())[0]]["n_minus1"].load()
        for key in list(inp.keys())[1:]:
            cut0 += inp[key]["cutflow"].load()
            minus0 += inp[key]["n_minus1"].load()

        print(cut0.values())
        self.output()[self.lepton_selection + "_cutflow"].dump(cut0)
        self.output()[self.lepton_selection + "_n_minus1"].dump(minus0)


class MergeArrays(CoffeaTask):  # , law.LocalWorkflow, HTCondorWorkflow):
    channel = luigi.ListParameter(default=["Muon", "Electron"])  # , "Electron"])

    # def create_branch_map(self):
    # # 1 job only
    # return list(range(1))

    def requires(self):
        inp = {
            sel: CoffeaProcessor.req(
                self,
                lepton_selection=sel,
                # workflow="local",
                datasets_to_process=self.datasets_to_process,
            )
            for sel in self.channel
        }
        return inp

    def output(self):
        out = {cat + "_" + dat: {"array": self.local_target("merged_{}_{}.npy".format(cat, dat)), "weights": self.local_target("weights_{}_{}.npy".format(cat, dat))} for cat in self.config_inst.categories.names() for dat in self.datasets_to_process}
        # out.update({"sum_gen_weights": self.local_target("sum_gen_weights.json")})
        return out

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        # construct an inverse map to corrently assign coffea outputs to respective datasets
        procs = self.get_proc_list(self.datasets_to_process)
        _, _, job_number_dict = self.load_job_dict()
        inverse_np_dict = {}
        for p in procs:
            for ind, file in job_number_dict.items():
                if p == file.split("/")[0]:
                    if p not in inverse_np_dict.keys():
                        inverse_np_dict[p] = [ind]
                    else:
                        inverse_np_dict[p] += [ind]

        for dat in tqdm(self.datasets_to_process):
            # check if job either in root process or leafes
            proc_list = self.get_proc_list([dat])
            if dat == "TTbar":
                proc_list = [p for p in proc_list if "TTTo" in p]
            for cat in self.config_inst.categories.names():
                cat_list = []
                weights_list = []
                # merging different lepton channels together according to self.channel
                for lep in self.channel:
                    np_dict = self.input()[lep]["collection"].targets[0]
                    # looping over all keys each time is rather slow
                    # but constructing keys yourself is tricky since there can be multiple jobs with different numbers
                    # so now I loop over possible keys for each dataset and append the correct arrays
                    for p in proc_list:
                        for ind in inverse_np_dict[p]:
                            key = cat + "_" + p + "_" + str(ind)
                            print(key)
                            cat_list.append(np_dict[key]["array"].load())
                            # get weights as well for each process
                            weights_list.append(np_dict[key]["weights"].load())

                # float 16 so arrays can be saved easily
                full_arr = np.concatenate(cat_list)  # , dtype=np.float16
                weights_arr = np.concatenate(weights_list)  # , dtype=np.float16) -> leads to inf
                print(dat, cat, full_arr, weights_arr)
                self.output()[cat + "_" + dat]["array"].parent.touch()
                self.output()[cat + "_" + dat]["array"].dump(full_arr)
                self.output()[cat + "_" + dat]["weights"].dump(weights_arr)


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
        # np_0b = self.input()["No_cuts"].load()
        np_0b = self.input()["No_cuts"].load()

        var_names = self.config_inst.variables.names()
        print(var_names)

        Dphi = np_0b[:, var_names.index("dPhi")]
        LT = np_0b[:, var_names.index("LT")]
        HT = np_0b[:, var_names.index("HT")]
        n_jets = np_0b[:, var_names.index("nJets")]
        jetPt_2 = np_0b[:, var_names.index("jetPt_2")]

        cut_list = [
            ("HLT_Or", "==1"),
            ("hard_lep", "==1"),
            ("selected", "==1"),
            ("no_veto_lepton", "==1"),
            ("iso_cut", "==1"),
            ("ghost_muon_filter", "==1"),
            ("HT", ">500"),
            ("jetPt_2", ">80"),
            ("nJets", ">=3"),
            ("LT", ">350"),
            ("doubleCounting_XOR", "==1")
            # ,('ghost_muon_filter', "==1"),("iso_cut", "==1"), ('hard_lep', "==1"), ('selected', "==1"), ("no_veto_lepton", "==1"), ('HLT_Or', "==1"), ('doubleCounting_XOR', "==1")
            # ("LT", "> 250"), ("LT", "> 500"), ("LT", "> 1000"), ("LT", "> 1500"), ("LT", "> 2000"), ("LT", "> 2500"), ("LT", "> 5000")]
            # [("nJets", ">2"), ("nJets", ">3"), ("nJets", ">4"), ("nJets", ">5"), ("nJets", ">6"), ("nJets", ">7"), ("nJets", ">8"), ("nJets", ">9")]
            # [("HT", "> 2500"), ("LT", ">500"), ("nJets", ">2"), ("leadMuonPt", ">25")]
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
