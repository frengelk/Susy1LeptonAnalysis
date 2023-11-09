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


class ConstructInferenceBins(DNNTask):
    category = luigi.Parameter(default="N0b", description="set it for now, can be dynamical later")
    kfold = luigi.IntParameter(default=2)

    def requires(self):
        return {
            "data_predicted": PredictDNNScores.req(self),
            "inputs": CrossValidationPrep.req(self, kfold=self.kfold),
            # "model":PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes,    dropout=self.dropout, kfold = self.kfold),
        }

    def output(self):
        return self.local_target("inference_bins.json")

    def store_parts(self):
        return super(ConstructInferenceBins, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        print("if problem persists, replace + by _")
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

        binning = self.config_inst.get_aux("signal_binning")
        # writing data separatly
        data_bins, data_obs = [], []
        for j, bin in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            if bin == self.config_inst.get_aux("signal_process").replace("V", "W"):
                data_in_node = data_scores[data_mask][:, j]
                hist = np.histogram(data_in_node, bins=binning)
                for k, edge in enumerate(binning[:-1]):
                    data_bins.append("DNN_Score_Node_" + bin + "_" + str(edge))
                    data_obs.append(str(hist[0][k]))

            else:
                data_bins.append("DNN_Score_Node_" + bin)
                data_mask = np.argmax(data_scores, axis=1) == j
                data_obs.append(str(np.sum(data_mask)))

        # writing MC for every bin
        # double loop over node names since we have the labels and the predicted region
        bin_names, process_names, process_numbers, rates = [], [], [], []
        for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            # leave CR as one bin, do a binning in SR to see high signal count
            if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                for i, proc in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                    find_proc = labels[:, i] == 1
                    MC_mask = np.argmax(MC_scores[find_proc], axis=1) == node
                    scores_in_node = MC_scores[find_proc][MC_mask][:, node]
                    hist = np.histogram(scores_in_node, bins=binning)
                    for j, edge in enumerate(binning[:-1]):
                        bin_names.append("DNN_Score_Node_" + key + "_" + str(edge))
                        process_names.append(proc)
                        process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - (i + 1)))
                        rates.append(str(hist[0][j]))
            else:
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
        print(bin_names, process_names, process_numbers, rates)
        inference_bins = {"data_bins": data_bins, "data_obs": data_obs, "bin_names": bin_names, "process_names": process_names, "process_numbers": process_numbers, "rates": rates}
        self.output().dump(inference_bins)


class DatacardWriting(DNNTask):
    category = luigi.Parameter(default="N0b", description="set it for now, can be dynamical later")
    kfold = luigi.IntParameter(default=2)

    def requires(self):
        return ConstructInferenceBins.req(self)

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
        print("if problem persists, replace + by _")
        proc_dict = self.config_inst.get_aux("DNN_process_template")["N" + self.channel]
        all_processes = list(proc_dict.keys())
        n_processes = len(all_processes)
        inp = self.input().load()
        n_bins = len(inp["data_bins"])

        # calculating xsec uncertainty, signal is tricky
        scinums = {}
        for p in list(self.config_inst.get_aux("DNN_process_template")[self.category].keys())[:-1]:
            real_procs = self.config_inst.get_aux("DNN_process_template")[self.category][p]
            a = 0
            for pr in real_procs:
                conf_p = self.config_inst.get_process(pr)
                for child in conf_p.walk_processes():
                    a += child[0].xsecs[13]
            scinums[p] = a
        scinums["T5qqqqWW"] = self.config_inst.get_process("T5qqqqWW").xsecs[13]

        process_unc = {}
        for key in scinums.keys():
            unc = []
            for name in inp["process_names"]:
                if name == key:
                    num = scinums[key]
                    # absolute unc / nominal
                    unc.append(str(1 + num.u()[0] / num.n))
                else:
                    unc.append("-")
            process_unc[key] = unc

        self.output().parent.touch()
        with open(self.output().path, "w") as datacard:
            datacard.write("## Datacard for (signal %s)\n" % (all_processes[-1]))
            datacard.write("imax {}  number of channels \n".format(n_bins))
            datacard.write("jmax %i  number of processes -1 \n" % (len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - 1))
            datacard.write("kmax * number of nuisance parameters (sources of systematical uncertainties) \n")  # .format(1)
            datacard.write("---\n")

            # shapes *           hww0j_8TeV  hwwof_0j.input_8TeV.root histo_$PROCESS histo_$PROCESS_$SYSTEMATIC
            # shapes data_obs    hww0j_8TeV  hwwof_0j.input_8TeV.root histo_Data

            datacard.write("bin " + " ".join(inp["data_bins"]) + "\n")
            datacard.write("observation " + " ".join(inp["data_obs"]) + "\n")
            datacard.write("---\n")

            datacard.write("bin " + " ".join(inp["bin_names"]) + "\n")
            datacard.write("process " + " ".join(inp["process_names"]) + "\n")
            datacard.write("process " + " ".join(inp["process_numbers"]) + "\n")
            datacard.write("rate " + " ".join(inp["rates"]) + "\n")

            # datacard.write('bin '+ "DNN_Score_Node_0 "*n_processes +"\n")
            # datacard.write("process " + " ".join([str(i ) for i in range(n_processes - 1, -1, -1)]) +"\n") # - n_processes+1
            # datacard.write("process " + " ".join([proc for proc in all_processes]) +"\n")
            # datacard.write("rate " + " ".join([str(val) for val in proc_dict.values()]) +"\n")

            # systematics
            datacard.write("---\n")
            datacard.write("lumi lnN " + "1.023 " * len(inp["bin_names"]) + "\n")  # each process has a score in each process node
            # missing xsec uncertainties
            for key in process_unc.keys():
                datacard.write("Xsec_{} lnN ".format(key) + " ".join(process_unc[key]) + "\n")

            # doing background estimation
            datacard.write("---\n")
            # format name rateParam channel process initial value
            datacard.write("alpha rateParam * {} 1 \n".format(all_processes[0]))
            datacard.write("beta rateParam * {} 1 \n".format(all_processes[1]))
