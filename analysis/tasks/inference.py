# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.backends.backend_pdf import PdfPages
import boost_histogram as bh
import mplhep as hep
import coffea
import torch
import sklearn as sk
from tqdm.auto import tqdm
from functools import total_ordering

# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask
from tasks.makefiles import CollectInputData
from tasks.grouping import GroupCoffea, MergeArrays  # , SumGenWeights
from tasks.arraypreparation import ArrayNormalisation, CrossValidationPrep
from tasks.multiclass import PytorchMulticlass, PredictDNNScores, PytorchCrossVal
from tasks.base import HTCondorWorkflow, DNNTask

import utils.pytorch_base as util


class DatacardWriting(DNNTask):
    category = luigi.Parameter(default="N0b", description="set it for now, can be dynamical later")
    kfold = luigi.IntParameter(default=2)

    def requires(self):
        return {
            "data_predicted": PredictDNNScores.req(self),
            "inputs": CrossValidationPrep.req(self, kfold=self.kfold),
            # "model":PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes,    dropout=self.dropout, kfold = self.kfold),
        }

    def output(self):
        # from IPython import embed; embed()
        return self.local_target("Datacard.txt")

    """
    def store_parts(self):
        # make plots for each use case
        parts = tuple()
        if self.unblinded:
            parts += ("unblinded",)
        if self.density:
            parts += ("density",)
        return super(DatacardWriting, self).store_parts() + parts + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)
    """

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        all_processes = list(self.config_inst.get_aux("DNN_process_template")["N" + self.channel].keys())  # ["data"] +
        n_processes = len(all_processes)

        self.output().parent.touch()

        # producing rates for node 0:
        proc_dict = {}
        node = 0
        data_scores = self.input()["data_predicted"]["data"].load()
        data_scores = self.input()["data_predicted"]["data"].load()
        MC_scores = self.input()["data_predicted"]["scores"].load()
        labels = self.input()["data_predicted"]["labels"].load()
        weights = self.input()["data_predicted"]["weights"].load()

        with open(self.output().path, "w") as datacard:
            datacard.write("## Datacard for (signal %s)\n" % (all_processes[-1]))
            datacard.write("imax {}  number of channels \n".format(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())))
            datacard.write("jmax %i  number of processes -1 \n" % (len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - 1))
            datacard.write("kmax * number of nuisance parameters (sources of systematical uncertainties) \n")  # .format(1)
            datacard.write("##---\n")

            # shapes *           hww0j_8TeV  hwwof_0j.input_8TeV.root histo_$PROCESS histo_$PROCESS_$SYSTEMATIC
            # shapes data_obs    hww0j_8TeV  hwwof_0j.input_8TeV.root histo_Data
            # writing data separatly
            bins, data_obs = [], []
            for j, bin in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                bins.append("DNN_Score_Node_" + bin)
                data_mask = np.argmax(data_scores, axis=1) == j
                data_obs.append(str(np.sum(data_mask)))

            datacard.write("bin " + " ".join(bins) + "\n")
            datacard.write("data_obs: " + " ".join(data_obs) + "\n")
            datacard.write("##---\n")

            # writing MC for every bin
            # double loop over node names since we have the labels and the predicted region
            bin_names, process_names, process_numbers, rates = [], [], [], []
            for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                for i, proc in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                    find_proc = labels[:, i] == 1
                    MC_mask = np.argmax(MC_scores[find_proc], axis=1) == node
                    # proc_dict.update({key: np.sum(weights[find_proc][MC_mask])})

                    bin_names.append("DNN_Score_Node_" + key)
                    process_names.append(proc)
                    # signal is last process in proc_dict, so we want it to be 0
                    process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - (i + 1)))
                    rates.append(str(np.sum(weights[find_proc][MC_mask])))

                    # klen = max(len(samp) for samp in sampNames)
                    # catDict = {"SR_MB" : "S", "CR_MB" : "C", "SR_SB" : "S2", "CR_SB" : "C2", "SR_SB_NB1i" : "S3", "CR_SB_NB1i" : "C3"}
                    # np.random.seed(42)

            datacard.write("bin " + " ".join(bin_names) + "\n")
            datacard.write("process " + " ".join(process_names) + "\n")
            datacard.write("process " + " ".join(process_numbers) + "\n")
            datacard.write("rate " + " ".join(rates) + "\n")

            # datacard.write('bin '+ "DNN_Score_Node_0 "*n_processes +"\n")
            # datacard.write("process " + " ".join([str(i ) for i in range(n_processes - 1, -1, -1)]) +"\n") # - n_processes+1
            # datacard.write("process " + " ".join([proc for proc in all_processes]) +"\n")
            # datacard.write("rate " + " ".join([str(val) for val in proc_dict.values()]) +"\n")

            # systematics
            datacard.write("##---\n")
            datacard.write("lumi lnN " + "1.11 " * n_processes**2 + "\n")  # each process has a score in each process node
        # from IPython import embed; embed()
