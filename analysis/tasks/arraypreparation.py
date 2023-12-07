# coding: utf-8

import luigi
import numpy as np
import sklearn.model_selection as skm

# other modules
from tasks.coffea import CoffeaTask
from tasks.grouping import MergeArrays


class ArrayNormalisation(CoffeaTask):

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    channel = luigi.Parameter(default="N0b_CR", description="channel to prepare")
    data = luigi.BoolParameter(default=False, description="save data")

    def requires(self):
        # if self.debug:
        # return CoffeaProcessor.req(self, debug=True, workflow="local")
        return MergeArrays.req(self, datasets_to_process=self.datasets_to_process)

    def path(self):
        return "/nfs/dust/cms/group/susy-desy/Susy1Lepton/0b/Run2_pp_13TeV_2016/CoffeaProcessor/testDNN/0b/ArrayExporter"

    # def output(self):
    # return self.local_target("hists")

    def output(self):
        out = {  # "norm_values": self.local_target("norm_values.npy"),
            "one_hot_labels": self.local_target("one_hot_labels.npy"),
            "MC_compl": self.local_target("MC_compl.npy"),
            "means_stds": self.local_target("means_stds.npy"),
            "X_train": self.local_target("X_train.npy"),
            "y_train": self.local_target("y_train.npy"),
            "weight_train": self.local_target("weight_train.npy"),
            "X_val": self.local_target("X_val.npy"),
            "y_val": self.local_target("y_val.npy"),
            "weight_val": self.local_target("weight_val.npy"),
            "X_test": self.local_target("X_test.npy"),
            "y_test": self.local_target("y_test.npy"),
            "weight_test": self.local_target("weight_test.npy"),
        }
        # cat + "_" + proc: self.local_target("normed_" + cat + "_" + proc + ".npy")
        # for proc in self.config_inst.processes.names()
        # for cat in self.config_inst.categories.names()
        # if self.channel in cat and not "data" in proc
        # }
        if self.data:
            out.update({"data": self.local_target("data.npy")})
        return out

    def normalise(self, array):
        return ((array - array.mean()) / array.std(), array.mean(), array.std())

    def calc_norm_parameter(self, data):
        # return values to shift distribution to normal

        dat = np.swapaxes(data, 0, 1)
        means, stds = [], []
        for var in dat:
            means.append(var.mean())
            stds.append(var.std())

        return np.array(means), np.array(stds)

    def run(self):
        self.output()["one_hot_labels"].parent.touch()

        # load inputs from MergedArrays
        # target_dict = self.input()["merged"]
        # debug_dict = self.input()["debug"]["collection"].targets
        # make regular dict out of ordered dict

        proc_dict = {}
        one_hot_labels = []
        data_list = []
        weight_dict = {}

        # loop through datasets and sort according to aux template
        # only N0b for now
        for cat in self.config_inst.categories.names()[:1]:
            for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[cat].keys()):
                # fill MC classes
                proc_list = []
                weight_list = []
                for subproc in self.config_inst.get_aux("DNN_process_template")[cat][key]:
                    proc_list.append(self.input()[cat + "_" + subproc]["array"].load())
                    weight_list.append(self.input()[cat + "_" + subproc]["weights"].load())
                    print("\n", subproc, np.max(self.input()[cat + "_" + subproc]["array"].load()))
                    argmax = np.argmax(self.input()[cat + "_" + subproc]["array"].load(), axis=0)
                    arr = self.input()[cat + "_" + subproc]["array"].load()
                    var_names = self.config_inst.variables.names()
                    # for j in range(len(argmax)):
                    #     print(var_names[j] , arr[argmax[j]][j])#, arr[argmax[i]])
                proc_dict.update({key: np.concatenate(proc_list)})
                weight_dict.update({key + "_weights": np.concatenate(weight_list)})

                # build labels for classification
                output_nodes = len(self.config_inst.get_aux("DNN_process_template")[cat].keys())
                labels = np.zeros((len(np.concatenate(proc_list)), output_nodes))
                labels[:, i] = 1
                print(i, key)
                one_hot_labels.append(labels)
            # fill data
            for dat in self.datasets_to_process:
                if self.config_inst.get_process(dat).aux["isData"]:
                    data_list.append(self.input()[cat + "_" + dat]["array"].load())

        # merge all processes
        MC_compl = np.concatenate(list(proc_dict.values()))
        one_hot_labels = np.concatenate(one_hot_labels)
        weight_compl = np.concatenate(list(weight_dict.values()))
        if self.data:
            data = np.concatenate(data_list)

        # split up test set 9:1
        X_train, X_test, y_train, y_test, weight_train, weight_test = skm.train_test_split(MC_compl, one_hot_labels, weight_compl, test_size=0.10, random_state=1)

        # train and validation set 80:20 FIXME
        X_train, X_val, y_train, y_val, weight_train, weight_val = skm.train_test_split(X_train, y_train, weight_train, test_size=0.2, random_state=2)

        # define means and stds for each variable
        means, stds = self.calc_norm_parameter(MC_compl)
        means_stds = np.vstack((means, stds))
        # save all arrays away, using the fact that keys have the variable name
        # Not the best way to do it, but very short
        print("\nmeans", means)
        print("\nstds", stds)
        for key in self.output().keys():
            self.output()[key].dump(eval(key))


class CrossValidationPrep(CoffeaTask):
    kfold = luigi.IntParameter(default=2)

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    def requires(self):
        return MergeArrays.req(self)

    # def output(self):
    # return self.local_target("hists")

    def output(self):
        out = {
            "cross_val_{}".format(i): {
                "cross_val_X_train_{}".format(i): self.local_target("cross_val_X_train_{}.npy".format(i)),
                "cross_val_y_train_{}".format(i): self.local_target("cross_val_y_train_{}.npy".format(i)),
                "cross_val_weight_train_{}".format(i): self.local_target("cross_val_weight_train_{}.npy".format(i)),
                "cross_val_X_val_{}".format(i): self.local_target("cross_val_X_val_{}.npy".format(i)),
                "cross_val_y_val_{}".format(i): self.local_target("cross_val_y_val_{}.npy".format(i)),
                "cross_val_weight_val_{}".format(i): self.local_target("cross_val_weight_val_{}.npy".format(i)),
            }
            for i in range(self.kfold)
        }

        out.update({"data": self.local_target("data.npy")})
        return out

    def normalise(self, array):
        return ((array - array.mean()) / array.std(), array.mean(), array.std())

    def calc_norm_parameter(self, data):
        # return values to shift distribution to normal

        dat = np.swapaxes(data, 0, 1)
        means, stds = [], []
        for var in dat:
            means.append(var.mean())
            stds.append(var.std())

        return np.array(means), np.array(stds)

    def run(self):
        proc_dict = {}
        weight_dict = {}
        one_hot_labels = []
        data_list = []
        DNNId_dict = {}

        # loop through datasets and sort according to aux template
        # for now only 1st category

        for cat in self.config_inst.categories.names()[:1]:
            for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[cat].keys()):
                proc_list = []
                weight_list = []
                DNNId_list = []
                print("node", key)
                for subproc in self.config_inst.get_aux("DNN_process_template")[cat][key]:
                    proc_list.append(self.input()[cat + "_" + subproc]["array"].load())
                    weight_list.append(self.input()[cat + "_" + subproc]["weights"].load())
                    DNNId_list.append(self.input()[cat + "_" + subproc]["DNNId"].load())
                    print(subproc, "len, weightsum", len(self.input()[cat + "_" + subproc]["array"].load()), np.sum(self.input()[cat + "_" + subproc]["weights"].load()))

                # print(proc_list)
                proc_dict.update({key: np.concatenate(proc_list)})
                weight_dict.update({key + "_weights": np.concatenate(weight_list)})
                DNNId_dict.update({key + "_DNNId": np.concatenate(DNNId_list)})
                # build labels for classification
                output_nodes = len(self.config_inst.get_aux("DNN_process_template")[cat].keys())
                labels = np.zeros((len(np.concatenate(proc_list)), output_nodes))
                labels[:, i] = 1
                print(key, i)
                one_hot_labels.append(labels)
            for dat in self.config_inst.aux["data"]:
                data_list.append(self.input()[cat + "_" + dat]["array"].load())

        # merge all processes
        MC_compl = np.concatenate(list(proc_dict.values()))
        weight_compl = np.concatenate(list(weight_dict.values()))
        DNNId_compl = np.concatenate(list(DNNId_dict.values()))
        one_hot_labels = np.concatenate(one_hot_labels)
        self.output()["data"].dump(np.concatenate(data_list))

        kfold = skm.KFold(n_splits=self.kfold, shuffle=True, random_state=42)
        # for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # kfold returns generator, loop over generated indices
        # for each kfold, dump the respective data and labels
        # for i, idx in enumerate(kfold.split(data_compl)):
        for i, idx in enumerate([-1, 1]):  # iterating over previous ids from coffea
            print("fold", i)
            # spltting everything in half, using remainding 90% for training, and 10% vor validation
            X_train, X_val, y_train, y_val, weight_train, weight_val = skm.train_test_split(MC_compl[DNNId_compl == idx], one_hot_labels[DNNId_compl == idx], weight_compl[DNNId_compl == idx], test_size=0.1, random_state=42)

            self.output()["cross_val_{}".format(i)]["cross_val_X_train_{}".format(i)].dump(X_train)
            self.output()["cross_val_{}".format(i)]["cross_val_y_train_{}".format(i)].dump(y_train)
            self.output()["cross_val_{}".format(i)]["cross_val_weight_train_{}".format(i)].dump(weight_train)
            self.output()["cross_val_{}".format(i)]["cross_val_X_val_{}".format(i)].dump(X_val)
            self.output()["cross_val_{}".format(i)]["cross_val_y_val_{}".format(i)].dump(y_val)
            self.output()["cross_val_{}".format(i)]["cross_val_weight_val_{}".format(i)].dump(weight_val)
            for j in range(len(y_train[0])):
                print("len weight for node:", j, np.sum(np.argmax(y_train, axis=-1) == j), np.sum(weight_train[np.argmax(y_train, axis=-1) == j]))
