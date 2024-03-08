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
from tasks.coffea import CoffeaProcessor, CoffeaTask, AntiProcessor
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
        # if self.category == "Anti_cuts":
        #     inp = {
        #     sel: AntiProcessor.req(
        #         self,
        #         lepton_selection=sel,
        #         #workflow="local",
        #         datasets_to_process=self.datasets_to_process,
        #     )
        #     for sel in self.channel
        # }

        return inp

    def output(self):
        out = {cat + "_" + dat: {"array": self.local_target("merged_{}_{}.npy".format(cat, dat)), "weights": self.local_target("weights_{}_{}.npy".format(cat, dat)), "DNNId": self.local_target("DNNId_{}_{}.npy".format(cat, dat))} for cat in [self.category] for dat in self.datasets_to_process}
        # out.update({"sum_gen_weights": self.local_target("sum_gen_weights.json")})
        # if self.category == "Anti_cuts":
        #     out = {cat + "_" + dat: {"array": self.local_target("merged_{}_{}.npy".format(cat, dat)), "weights": self.local_target("weights_{}_{}.npy".format(cat, dat))} for cat in [self.category] for dat in self.datasets_to_process}
        return out

    def store_parts(self):
        return super(MergeArrays, self).store_parts() + ("_".join(self.channel),)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        # debugging stuff
        var = self.config_inst.get_variable("HT")

        # construct an inverse map to corrently assign coffea outputs to respective datasets
        procs = self.get_proc_list(self.datasets_to_process)
        _, _, job_number_dict, proc_dict = self.load_job_dict()
        inverse_np_dict = {}
        for p in procs:
            for ind, file in proc_dict.items():
                if p == file:  # .split("/")[0]:
                    if p not in inverse_np_dict.keys():
                        inverse_np_dict[p] = [ind]
                    else:
                        inverse_np_dict[p] += [ind]

        for dat in tqdm(self.datasets_to_process):
            # check if job either in root process or leafes
            proc_list = self.get_proc_list([dat])
            # if dat == "TTbar":
            # proc_list = [p for p in proc_list if "TTTo" in p]
            cat = self.category
            # for cat in self.config_inst.categories.names():
            cat_list = []
            weights_list = []
            DNNId_list = []
            # merging different lepton channels together according to self.channel
            for lep in self.channel:
                np_dict = self.input()[lep]["collection"].targets[0]
                # looping over all keys each time is rather slow
                # but constructing keys yourself is tricky since there can be multiple jobs with different numbers
                # so now I loop over possible keys for each dataset and append the correct arrays
                for p in proc_list:
                    for ind in inverse_np_dict[p]:
                        key = cat + "_" + p + "_" + str(ind)
                        arr = np_dict[key]["array"].load()
                        if len(arr) > 0:
                            if len(arr[0]) > 21:
                                print(lep, key, len(arr[0]))
                        weights = np_dict[key]["weights"].load()
                        IDs = np_dict[key]["DNNId"].load()
                        if len(np_dict[key]["array"].load()) > 1:
                            if np.max(np_dict[key]["array"].load()) > 1e6:
                                maxis = arr[arr > 100000]
                                print("WARNING, C++ skimming messed up, removing ", len(np.unique(maxis)), " entries from:", key)
                                ind_maxis = []
                                for maxi in maxis:
                                    i, j = np.where(np.isclose(arr, maxi))
                                    ind_maxis.append(i)
                                ind_maxis = np.concatenate(ind_maxis)
                                arr = np.delete(arr, ind_maxis, axis=0)
                                weights = np.delete(weights, ind_maxis, axis=0)
                                IDs = np.delete(IDs, ind_maxis, axis=0)

                        # get weights as well for each process
                        # if p == "QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8" and lep == "Electron":
                        #     # something wrong with weights in this sample
                        #     weights_list.append(np.ones(len(np_dict[key]["weights"].load())))
                        # else:
                        cat_list.append(arr)
                        weights_list.append(weights)
                        DNNId_list.append(IDs)
                        # import boost_histogram as bh
                        # boost_hist =  bh.Histogram(bh.axis.Regular(var.binning[0], var.binning[1], var.binning[2]))
                        # boost_hist.fill(np_dict[key]["array"].load()[:,3])
                        # print(key, boost_hist.values())

            # float 16 so arrays can be saved easily
            full_arr = np.concatenate(cat_list)  # , dtype=np.float16
            weights_arr = np.concatenate(weights_list)  # , dtype=np.float16) -> leads to inf
            # print(cat, dat, np.sum(weights_arr), np.mean(weights_arr))
            # print(dat, cat, full_arr, weights_arr)
            self.output()[cat + "_" + dat]["array"].parent.touch()
            self.output()[cat + "_" + dat]["array"].dump(full_arr)
            self.output()[cat + "_" + dat]["weights"].dump(weights_arr)
            DNNId_arr = np.concatenate(DNNId_list)
            self.output()[cat + "_" + dat]["DNNId"].dump(DNNId_arr)
            print(dat, " len: ", len(weights_arr), " sum: ", sum(weights_arr))
        # exporting variables to be able to reconstruct array content
        np.save(self.output()[cat + "_" + dat]["array"].parent.path + "/variables.npy", self.config_inst.variables.names())


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


# to check all requirements. comment out the Workflow dependencies, no idea why)
class MergeShiftArrays(CoffeaTask, HTCondorWorkflow, law.LocalWorkflow):
    # require more ressources
    RAM = 2500
    hours = 1
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    shifts = luigi.ListParameter(default=["systematic_shifts"])

    def create_branch_map(self):
        # define job number according to number of files of the dataset that you want to process
        return list(range(len(self.datasets_to_process)))

    def requires(self):
        inp = {
            shift
            + "_"
            + sel: CoffeaProcessor.req(
                self,
                lepton_selection=sel,
                datasets_to_process=self.datasets_to_process,
                shift=shift,  # workflow="local"
            )
            for sel in self.channel
            for shift in self.shifts  # ("nominal",)+
        }
        return inp

    def output(self):
        shifts_long = self.unpack_shifts()
        out = {cat + "_" + dat + "_" + shift: {"array": self.local_target("merged_{}_{}_{}.npy".format(cat, dat, shift)), "weights": self.local_target("weights_{}_{}_{}.npy".format(cat, dat, shift)), "DNNId": self.local_target("DNNId_{}_{}_{}.npy".format(cat, dat, shift))} for cat in [self.category] for dat in self.datasets_to_process for shift in shifts_long}  # self.config_inst.categories.names()
        return out

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        # np_0b = self.input()["No_cuts"].load()
        inp = self.input()

        # construct an inverse map to corrently assign coffea outputs to respective datasets
        procs = self.get_proc_list(self.datasets_to_process)
        _, _, job_number_dict, proc_dict = self.load_job_dict()
        inverse_np_dict = {}
        for p in procs:
            for ind, file in proc_dict.items():
                if p == file:  # .split("/")[0]:
                    if p not in inverse_np_dict.keys():
                        inverse_np_dict[p] = [ind]
                    else:
                        inverse_np_dict[p] += [ind]

        # for dat in tqdm(self.datasets_to_process):
        dat = self.datasets_to_process[self.branch]
        print(dat)
        # check if job either in root process or leafes
        proc_list = self.get_proc_list([dat])
        # if dat == "TTbar":
        # proc_list = [p for p in proc_list if "TTTo" in p]
        # for cat in self.config_inst.categories.names():
        cat = self.category
        # replace systematic shifts in self.shifts so we can properly loop
        shifts_long = self.unpack_shifts()
        for shift in tqdm(shifts_long):
            weights_list = []
            DNNId_list = []
            cat_list = []
            # looping over all keys each time is rather slow
            # but constructing keys yourself is tricky since there can be multiple jobs with different numbers
            # so now I loop over possible keys for each dataset and append the correct arrays
            # merging different lepton channels together according to self.channel
            for lep in self.channel:
                if shift in self.config_inst.get_aux("systematic_shifts"):
                    np_dict = self.input()["systematic_shifts" + "_" + lep]  # ["collection"].targets[0]
                    for p in proc_list:
                        for ind in inverse_np_dict[p]:
                            key = cat + "_" + p + "_" + str(ind)
                            # get weights as well for each process
                            weights_list.append(np_dict[key]["systematic_shifts"][shift].load())
                            cat_list.append(np_dict[key]["array"].load())
                            DNNId_list.append(np_dict[key]["DNNId"].load())
                else:  # should be TotalUp Down
                    np_dict = self.input()[shift + "_" + lep]
                    for p in proc_list:
                        for ind in inverse_np_dict[p]:
                            key = cat + "_" + p + "_" + str(ind)
                            # get weights as well for each process
                            weights_list.append(np_dict[key]["weights"].load())
                            cat_list.append(np_dict[key]["array"].load())
                            DNNId_list.append(np_dict[key]["DNNId"].load())

            print(shift)
            # if shift == "PreFireWeightDown":
            full_arr = np.concatenate(cat_list)  # , dtype=np.float16
            self.output()[cat + "_" + dat + "_" + shift]["array"].dump(full_arr)
            weights_arr = np.concatenate(weights_list)
            self.output()[cat + "_" + dat + "_" + shift]["weights"].dump(weights_arr)
            DNNId_arr = np.concatenate(DNNId_list)
            self.output()[cat + "_" + dat + "_" + shift]["DNNId"].dump(DNNId_arr)
