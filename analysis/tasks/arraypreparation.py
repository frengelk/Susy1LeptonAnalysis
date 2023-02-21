# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import operator

import sklearn.model_selection as skm


# other modules
from tasks.basetasks import ConfigTask
from tasks.coffea import CoffeaProcessor, CollectCoffeaOutput
from utils.SignalRegions import signal_regions_0b


class ArrayNormalisation(ConfigTask):

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    channel = luigi.Parameter(default="N1b_CR", description="channel to prepare")

    def requires(self):
        return {
            "debug": CoffeaProcessor.req(
                self, processor="ArrayExporter", debug=True, workflow="local"
            ),
            # comment out following line if just interested in debug
            "complete": CoffeaProcessor.req(self, processor="ArrayExporter"),
        }

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

        # load inputs from ArrayExporter
        target_dict = self.input()["complete"]["collection"].targets
        # debug_dict = self.input()["debug"]["collection"].targets

        # make regular dict out of ordered dict
        file_dict = {}
        for i in target_dict.keys():
            file_dict.update(target_dict[i])

        proc_dict = {}
        one_hot_labels = []

        # loop through datasets and sort according to aux template
        datasets = list(file_dict.values())
        # os.listdir(self.path())
        for i, key in enumerate(
            self.config_inst.get_aux("DNN_process_template").keys()
        ):
            proc_list = []
            for cat in self.config_inst.categories.names():
                # print(1,cat)
                for subproc in self.config_inst.get_aux("DNN_process_template")[key]:
                    # print(2,subproc)
                    # catch bottom level processes that have no childs
                    for dat in datasets:
                        if cat in dat.path and subproc in dat.path:
                            # self.config_inst.datasets.names():
                            proc_list.append(dat.load())

                        for subsubproc in self.config_inst.get_process(
                            subproc
                        ).walk_processes():
                            # print(3,dat, cat, subsubproc)
                            if cat in dat.path and subsubproc[0].name in dat.path:
                                # load input
                                # print(4,dat, subsubproc[0].name)
                                proc_list.append(dat.load())

            # from IPython import embed;embed()

            # print(proc_list)
            proc_dict.update({key: np.concatenate(proc_list)})
            # 3 processes
            labels = np.zeros((len(np.concatenate(proc_list)), 3))
            labels[:, i] = 1
            one_hot_labels.append(labels)

        # merge all processes
        data_compl = np.concatenate(list(proc_dict.values()))
        one_hot_labels = np.concatenate(one_hot_labels)

        # split up test set 9:1
        X_train, X_test, y_train, y_test = skm.train_test_split(
            data_compl, one_hot_labels, test_size=0.10, random_state=1
        )

        # train and validation set 80:20 FIXME
        X_train, X_val, y_train, y_val = skm.train_test_split(
            X_train, y_train, test_size=0.5, random_state=2
        )

        # define means and stds for each variable
        means, stds = self.calc_norm_parameter(data_compl)
        means_stds = np.vstack((means, stds))

        # save all arrays away, using the fact that keys have the variable name
        for key in self.output().keys():
            self.output()[key].dump(eval(key))


class CrossValidationPrep(ConfigTask):

    kfold = luigi.IntParameter(default=5)

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    channel = luigi.Parameter(default="N1b_CR", description="channel to prepare")

    def requires(self):
        return CoffeaProcessor.req(self, processor="ArrayExporter")

    # def output(self):
    # return self.local_target("hists")

    def output(self):
        out = {}
        out.update(
            {
                "cross_val_{}".format(i): {
                    "cross_val_X_train_{}".format(i): self.local_target(
                        "cross_val_X_train_{}.npy".format(i)
                    ),
                    "cross_val_y_train_{}".format(i): self.local_target(
                        "cross_val_y_train_{}.npy".format(i)
                    ),
                    "cross_val_X_val_{}".format(i): self.local_target(
                        "cross_val_X_val_{}.npy".format(i)
                    ),
                    "cross_val_y_val_{}".format(i): self.local_target(
                        "cross_val_y_val_{}.npy".format(i)
                    ),
                }
                for i in range(self.kfold)
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

        # load inputs from ArrayExporter
        target_dict = self.input()["collection"].targets

        # make regular dict out of ordered dict
        file_dict = {}
        for i in target_dict.keys():
            file_dict.update(target_dict[i])

        proc_dict = {}
        one_hot_labels = []

        # loop through datasets and sort according to aux template
        datasets = list(file_dict.values())

        # at some point, we have to include k fold cross validation
        # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
        # from sklearn.model_selection import KFold

        for i, key in enumerate(
            self.config_inst.get_aux("DNN_process_template").keys()
        ):
            proc_list = []
            for cat in self.config_inst.categories.names():
                # print(1,cat)
                for subproc in self.config_inst.get_aux("DNN_process_template")[key]:
                    # print(2,subproc)
                    # catch bottom level processes that have no childs
                    for dat in datasets:
                        if cat in dat.path and subproc in dat.path:
                            # self.config_inst.datasets.names():
                            proc_list.append(dat.load())

                        for subsubproc in self.config_inst.get_process(
                            subproc
                        ).walk_processes():
                            # print(3,dat, cat, subsubproc)
                            if cat in dat.path and subsubproc[0].name in dat.path:
                                # load input
                                # print(4,dat, subsubproc[0].name)
                                proc_list.append(dat.load())

            # from IPython import embed;embed()

            # print(proc_list)
            proc_dict.update({key: np.concatenate(proc_list)})
            # 3 processes
            labels = np.zeros((len(np.concatenate(proc_list)), 3))
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

            self.output()["cross_val_{}".format(i)][
                "cross_val_X_train_{}".format(i)
            ].dump(data_compl[train_idx])
            self.output()["cross_val_{}".format(i)][
                "cross_val_y_train_{}".format(i)
            ].dump(one_hot_labels[train_idx])
            self.output()["cross_val_{}".format(i)][
                "cross_val_X_val_{}".format(i)
            ].dump(data_compl[val_idx])
            self.output()["cross_val_{}".format(i)][
                "cross_val_y_val_{}".format(i)
            ].dump(one_hot_labels[val_idx])


class WriteSignalTable(ConfigTask):
    def requires(self):
        return CollectCoffeaOutput.req(self, processor="ArrayExporter")

    # def output(self):
    # return self.local_target("hists")

    def output(self):
        return self.local_target("SignalTable.txt")

    def run(self):
        inp = self.input()["signal_bin_counts"].load()
        self.output().parent.touch()
        with open(self.output().path, "w") as text_file:
            text_file.write(r"\begin{table}" + "\n")
            text_file.write(r"\begin{tabular}{|l|l|c|}" + " \hline \n")
            text_file.write("Bin & Cut & Yield \\\ \hline \n")
            for key in signal_regions_0b.keys():
                text_file.write(
                    "{} & {} & {} \\\ \hline \n".format(
                        key,
                        " ".join(signal_regions_0b[key]).replace("&", " "),
                        inp[key],
                    )
                )
            text_file.write("\end{tabular} \n \end{table}")
