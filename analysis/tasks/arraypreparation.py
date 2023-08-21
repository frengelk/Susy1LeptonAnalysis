# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import operator

import sklearn.model_selection as skm


# other modules
from tasks.base import ConfigTask
from tasks.coffea import CoffeaTask, CoffeaProcessor
from tasks.grouping import MergeArrays


class ArrayNormalisation(CoffeaTask):

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    channel = luigi.Parameter(default="N0b_CR", description="channel to prepare")

    def requires(self):
        # if self.debug:
        # return CoffeaProcessor.req(self, debug=True, workflow="local")

        # else:
        print("\nMergeArrays", MergeArrays.req(self))
        return MergeArrays.req(self)

    def path(self):
        return "/nfs/dust/cms/group/susy-desy/Susy1Lepton/0b/Run2_pp_13TeV_2016/CoffeaProcessor/testDNN/0b/ArrayExporter"

    # def output(self):
    # return self.local_target("hists")

    def output(self):
        # In [9]: for k in self.config_inst.get_aux("DNN_process_template").keys():
        out = {}
        # cat + "_" + proc: self.local_target("normed_" + cat + "_" + proc + ".npy")
        # for proc in self.config_inst.processes.names()
        # for cat in self.config_inst.categories.names()
        # if self.channel in cat and not "data" in proc
        # }
        out.update(
            {  # "norm_values": self.local_target("norm_values.npy"),
                "one_hot_labels": self.local_target("one_hot_labels.npy"),
                "data_compl": self.local_target("data_compl.npy"),
                "means_stds": self.local_target("means_stds.npy"),
                "X_train": self.local_target("X_train.npy"),
                "y_train": self.local_target("y_train.npy"),
                "X_val": self.local_target("X_val.npy"),
                "y_val": self.local_target("y_val.npy"),
                "X_test": self.local_target("X_test.npy"),
                "y_test": self.local_target("y_test.npy"),
            }
        )
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

        # loop through datasets and sort according to aux template

        for cat in self.config_inst.categories.names()[:1]:
            print(1, cat)
            proc_list = []
            for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[cat].keys()):
                for subproc in self.config_inst.get_aux("DNN_process_template")[cat][key]:
                    print(2, subproc)
                    proc_list.append(self.input()[cat + "_" + subproc]["array"].load())

                # print(proc_list)
                proc_dict.update({key: np.concatenate(proc_list)})
                # build labels for classification
                output_nodes = len(self.config_inst.get_aux("DNN_process_template")[cat].keys())
                labels = np.zeros((len(np.concatenate(proc_list)), output_nodes))
                labels[:, i] = 1
                one_hot_labels.append(labels)

        # merge all processes
        data_compl = np.concatenate(list(proc_dict.values()))
        one_hot_labels = np.concatenate(one_hot_labels)

        # split up test set 9:1
        X_train, X_test, y_train, y_test = skm.train_test_split(data_compl, one_hot_labels, test_size=0.10, random_state=1)

        # train and validation set 80:20 FIXME
        X_train, X_val, y_train, y_val = skm.train_test_split(X_train, y_train, test_size=0.5, random_state=2)

        # define means and stds for each variable
        means, stds = self.calc_norm_parameter(data_compl)
        means_stds = np.vstack((means, stds))

        # save all arrays away, using the fact that keys have the variable name
        for key in self.output().keys():
            self.output()[key].dump(eval(key))


class CrossValidationPrep(CoffeaTask):
    kfold = luigi.IntParameter(default=5)

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    channel = luigi.Parameter(default="N0b_CR", description="channel to prepare")

    def requires(self):
        return MergeArrays.req(self)

    # def output(self):
    # return self.local_target("hists")

    def output(self):
        return {
            "cross_val_{}".format(i): {
                "cross_val_X_train_{}".format(i): self.local_target("cross_val_X_train_{}.npy".format(i)),
                "cross_val_y_train_{}".format(i): self.local_target("cross_val_y_train_{}.npy".format(i)),
                "cross_val_X_val_{}".format(i): self.local_target("cross_val_X_val_{}.npy".format(i)),
                "cross_val_y_val_{}".format(i): self.local_target("cross_val_y_val_{}.npy".format(i)),
            }
            for i in range(self.kfold)
        }

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
        one_hot_labels = []

        # loop through datasets and sort according to aux template

        for cat in self.config_inst.categories.names()[:1]:
            print(1, cat)
            proc_list = []
            for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[cat].keys()):
                for subproc in self.config_inst.get_aux("DNN_process_template")[cat][key]:
                    print(2, subproc)
                    proc_list.append(self.input()[cat + "_" + subproc]["array"].load())

                # print(proc_list)
                proc_dict.update({key: np.concatenate(proc_list)})
                # build labels for classification
                output_nodes = len(self.config_inst.get_aux("DNN_process_template")[cat].keys())
                labels = np.zeros((len(np.concatenate(proc_list)), output_nodes))
                labels[:, i] = 1
                one_hot_labels.append(labels)

        # merge all processes
        data_compl = np.concatenate(list(proc_dict.values()))
        one_hot_labels = np.concatenate(one_hot_labels)

        kfold = skm.KFold(n_splits=self.kfold, shuffle=True, random_state=42)
        # for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        # kfold returns generator, loop over generated indices
        # for each kfold, dump the respective data and labels
        for i, idx in enumerate(kfold.split(data_compl)):
            # unpack tuple
            train_idx, val_idx = idx

            self.output()["cross_val_{}".format(i)]["cross_val_X_train_{}".format(i)].dump(data_compl[train_idx])
            self.output()["cross_val_{}".format(i)]["cross_val_y_train_{}".format(i)].dump(one_hot_labels[train_idx])
            self.output()["cross_val_{}".format(i)]["cross_val_X_val_{}".format(i)].dump(data_compl[val_idx])
            self.output()["cross_val_{}".format(i)]["cross_val_y_val_{}".format(i)].dump(one_hot_labels[val_idx])
