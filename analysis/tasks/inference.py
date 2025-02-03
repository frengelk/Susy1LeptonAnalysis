# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.backends.backend_pdf import PdfPages
import boost_histogram as bh
import mplhep as hep
import coffea
from coffea import processor  # , hist
import torch
import sklearn as sk
from tqdm.auto import tqdm
from functools import total_ordering
import json
import time
import pandas as pd

# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask
from tasks.makefiles import WriteDatasetPathDict, CollectInputData, CalcBTagSF, CollectMasspoints
from tasks.grouping import GroupCoffea, MergeArrays, MergeShiftArrays
from tasks.arraypreparation import ArrayNormalisation, CrossValidationPrep
from tasks.multiclass import PytorchMulticlass, PredictDNNScores, PytorchCrossVal, CalcNormFactors
from tasks.base import HTCondorWorkflow, DNNTask
from tasks.bgestimation import EstimateQCDinSR

import utils.pytorch_base as util
from utils.coffea_base import ArrayExporter, ArrayAccumulator


class ConstructInferenceBins(DNNTask):
    # category = luigi.Parameter(default="N0b", description="set it for now, can be dynamical later")
    kfold = luigi.IntParameter(default=2)
    QCD_est = luigi.BoolParameter(default=False)
    apply_norm = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "data_predicted": PredictDNNScores.req(self, workflow="local"),
            "inputs": CrossValidationPrep.req(self, kfold=self.kfold),
            "QCD_pred": EstimateQCDinSR.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "QCD", "Rare", "DY", "MET", "SingleMuon", "SingleElectron"]),
            "Norm_factors": CalcNormFactors.req(self, QCD_estimate=self.QCD_est),  # norm factors should reflect QCD yields
            # "model":PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes,    dropout=self.dropout, kfold = self.kfold),
        }

    def output(self):
        return self.local_target("inference_bins.json")

    def store_parts(self):
        parts = tuple()
        if self.apply_norm:
            parts += ("apply_norm",)
        return super(ConstructInferenceBins, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + (self.gamma,) + parts

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        print("if problem persists, replace + by _")
        all_processes = list(self.config_inst.get_aux("DNN_process_template")[self.category].keys())  # ["data"] +
        n_processes = len(all_processes)

        self.output().parent.touch()

        # opening all needed inputs
        proc_dict = {}
        data_scores = self.input()["data_predicted"]["collection"].targets[0]["data"].load()
        MC_scores = self.input()["data_predicted"]["collection"].targets[0]["scores"].load()
        labels = self.input()["data_predicted"]["collection"].targets[0]["labels"].load()
        weights = self.input()["data_predicted"]["collection"].targets[0]["weights"].load()
        QCD_scores = self.input()["data_predicted"]["collection"].targets[0]["QCD_scores"].load()
        QCD_weights = self.input()["data_predicted"]["collection"].targets[0]["QCD_weights"].load()
        # in SR, we take data-driven QCD prediction
        QCD_estimate = self.input()["QCD_pred"].load()["SR_prediction"]
        QCD_est_err = self.input()["QCD_pred"].load()["SR_err"]
        QCD_bg_est = self.input()["QCD_pred"].load()["diff_data_EWK"]
        QCD_bg_est_err = self.input()["QCD_pred"].load()["diff_err"]
        F_Sel_Anti = self.input()["QCD_pred"].load()["F_Sel_Anti"]
        print("Importing QCD estimate:", QCD_estimate)

        binning = self.config_inst.get_aux("signal_binning")
        norm_factors = self.input()["Norm_factors"].load()
        proc_factor_dict = {"ttjets": "alpha", "Wjets": "beta"}
        # writing data separatly

        data_bins, data_obs = [], []
        for j, bin in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            if bin == self.config_inst.get_aux("signal_process").replace("V", "W"):
                data_mask = np.argmax(data_scores, axis=1) == j
                data_in_node = data_scores[data_mask][:, j]
                hist = np.histogram(data_in_node, bins=binning)
                for k, edge in enumerate(binning[:-1]):
                    data_bins.append("DNN_Score_Node_" + bin + "_" + str(edge))
                    data_obs.append(str(hist[0][k]))

            else:
                data_bins.append("DNN_Score_Node_" + bin)
                data_mask = np.argmax(data_scores, axis=1) == j
                data_obs.append(str(np.sum(data_mask)))
        print("observed data", data_obs)
        # writing MC for every bin
        # double loop over node names since we have the labels and the predicted region
        # we count up, T5:0, ttjets:1, WJets:2, QCD:3
        bin_names, process_names, process_numbers, rates, rates_err = [], [], [], [], []
        for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            # leave CR as one bin, do a binning in SR to resolve high signal count regions better
            if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                # for j, edge in enumerate(binning[:-1]):
                for i, proc in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                    # signal is done in YieldsPerMasspoint, QCD estimate is done before, so skip MC
                    if proc == "T5qqqqWW" or (proc == "QCD" and self.QCD_est):
                        continue
                    find_proc = labels[:, i] == 1
                    MC_mask = np.argmax(MC_scores[find_proc], axis=1) == node
                    scores_in_node = MC_scores[find_proc][MC_mask][:, node]

                    # applying calculated norm factor on bg yields
                    factor = 1.0
                    if self.apply_norm:
                        if proc in proc_factor_dict.keys():
                            factor = norm_factors[proc_factor_dict[proc]]

                    # doing counts per binning via boost for errors
                    new_hist_sig = bh.Histogram(bh.axis.Variable(binning), storage=bh.storage.Weight())
                    sig_weights = weights[find_proc][MC_mask] * factor
                    new_hist_sig.fill(scores_in_node, weight=sig_weights)

                    errors_sig = np.sqrt(new_hist_sig.variances())
                    counts = new_hist_sig.counts()
                    # we need to find weights in each bin
                    # events_in_bin = (scores_in_node > binning[j]) & (scores_in_node < binning[j + 1])
                    # weights_in_bin = weights[find_proc][MC_mask][events_in_bin]
                    bin_names.extend(["DNN_Score_Node_" + key + "_" + str(edge) for edge in binning[:-1]])
                    process_names.extend([proc] * len(counts))
                    process_numbers.extend([str(i + 1)] * len(counts))
                    # rates.append(str(factor *np.sum(weights_in_bin))) # FIXME factor * in datacard
                    rates.extend([str(c) for c in counts])
                    rates_err.extend([str(1 + c) for c in errors_sig / counts])
                    # doing debugging
                    new_hist_debug = bh.Histogram(bh.axis.Variable(binning))
                    new_hist_debug.fill(scores_in_node)
                    print(proc, " in signal bins with rates&errors&raw yields:\n", counts, "\n", errors_sig / counts, "\n", new_hist_debug.counts(), "\n")
                if self.QCD_est:
                    # QCD on top, append QCD on its own, QCD estimate in SR was calculated beforehand
                    bin_names.extend(["DNN_Score_Node_" + key + "_" + str(edge) for edge in binning[:-1]])
                    process_names.extend(["QCD"] * len(counts))
                    process_numbers.extend([str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - 1)] * len(counts))
                    rates.extend([str(QCD) for QCD in QCD_estimate])
                    # rates_err.extend(["QCD_unc"] * len(counts))
                    rates_err.extend(["QCD_unc " + str(err / nom) for err, nom in zip(QCD_est_err, QCD_estimate)])

            else:
                for i, proc in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                    if proc == "T5qqqqWW":
                        continue
                    find_proc = labels[:, i] == 1
                    MC_mask = np.argmax(MC_scores[find_proc], axis=1) == node
                    # proc_dict.update({key: np.sum(weights[find_proc][MC_mask])})

                    bin_names.append("DNN_Score_Node_" + key)
                    process_names.append(proc)
                    # # signal is last process in proc_dict, so we want it to be 0
                    # process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - (i + 1)))
                    process_numbers.append(str(i + 1))
                    # applying calculated norm factor on bg yields
                    factor = 1.0
                    if self.apply_norm:
                        if proc in proc_factor_dict.keys():
                            factor = norm_factors[proc_factor_dict[proc]]

                    # doing error estimation with histograming, saving first entry since only one bin
                    new_hist_bkg = bh.Histogram((1, 0, 1), storage=bh.storage.Weight())
                    bkg_weights = factor * weights[find_proc][MC_mask]
                    new_hist_bkg.fill(MC_scores[:, node][find_proc][MC_mask], weight=bkg_weights)
                    errors_bkg = np.sqrt(new_hist_bkg.variances())

                    # also doing data driven QCD in bg nodes
                    if proc == "QCD" and self.QCD_est:
                        rates.append(str(QCD_bg_est[node] * F_Sel_Anti * factor))
                        rates_err.append("QCD_unc " + str(QCD_bg_est_err[node] / QCD_bg_est[node]))
                        continue

                    # rates.append(str(factor *np.sum(weights[find_proc][MC_mask]))) # FIXME factor * in datacard
                    rates.append(str(new_hist_bkg.counts()[0]))
                    rates_err.append(str(1 + errors_bkg[0] / new_hist_bkg.counts()[0]))
                # # append QCD on its own
                # argmax_mask = np.argmax(QCD_scores, axis=1) == node

                # new_hist_bkg=bh.Histogram((1,0,1),storage=bh.storage.Weight())
                # new_hist_bkg.fill(QCD_scores[:,node][argmax_mask], weight = QCD_weights[argmax_mask])
                # errors_bkg = np.sqrt(new_hist_bkg.variances())

                # bin_names.append("DNN_Score_Node_" + key)
                # process_names.append("QCD")
                # print("QCD unten")
                # # just put QCD on top of all other processes
                # process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())-1))
                # rates.append(str(new_hist_bkg.counts()[0]))
                # rates_err.append(str(1 + errors_bkg[0] / new_hist_bkg.counts()[0]))

        print(bin_names, process_names, process_numbers, rates, rates_err)

        nominal_dict = {}
        for i, rate in enumerate(rates):
            nominal_dict["{}_{}".format(bin_names[i], process_names[i])] = rate

        # blinding last 5 signal bins, replace them by data_count FIXME!!!
        print("\nBlinding is turned off\n")
        # sum_up_bg = {id: [] for id in np.unique(process_numbers)}
        # for id, numb in enumerate(process_numbers):
        #     sum_up_bg[numb].append(float(rates[id]))
        # summed_bg = np.sum(list(sum_up_bg.values()), 0)
        # # blinding last data bins
        # for iobs in range(len(data_obs)):
        #     if iobs > (len(data_obs) - 6):
        #         print(data_obs[iobs])
        #         data_obs[iobs] = str(int(summed_bg[iobs] + 0.5))
        # put together output dict to write datacards
        inference_bins = {"data_bins": data_bins, "data_obs": data_obs, "bin_names": bin_names, "process_names": process_names, "process_numbers": process_numbers, "rates": rates, "rates_err": rates_err, "nominal_dict": nominal_dict}

        # optimising binning?
        all_MC = MC_scores  # np.append(MC_scores, QCD_scores, axis=0)
        all_weights = weights  # np.append(weights, QCD_weights, axis=0)
        all_labels = labels  # np.append(labels, np.resize([0.5, 0.5, 0], (len(QCD_weights), 3)), axis=0)
        not_signal = all_labels[:, -1] == 0
        not_QCD = ~(all_labels[:, 2] == 1)  # we want to skip QCD from MC as well
        MC_scores_bg_in_signode = all_MC[not_signal & not_QCD][:, -1]
        weights_bg = all_weights[not_signal & not_QCD]
        for step in np.arange(0.9, 1, 0.001):
            print(np.round(step, 4), ": ", np.sum(weights_bg[MC_scores_bg_in_signode > step]))
        # print("data max signal sore: ", np.max(data_scores[:, -1]), " and second highest: ", np.max(data_scores[:, -1][data_scores[:, -1] < np.max(data_scores[:, -1])]))
        print("Max signal score:", np.max(MC_scores[:, -1]))
        # finish task
        for i, p in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            print("SR prediction in node ", p, ":", self.input()["QCD_pred"].load()["diff_data_EWK"][i] * self.input()["QCD_pred"].load()["F_Sel_Anti"])
            QCD_in_node = np.argmax(QCD_scores, axis=-1) == i
            print("MC sum in same node:", np.sum(QCD_weights[QCD_in_node]))
        self.output().dump(inference_bins)


class GetShiftedYields(CoffeaTask, DNNTask):  # jobs should be really small
    shifts = luigi.ListParameter(default=["PreFireWeightUp"])
    channel = luigi.ListParameter(default=["LeptonIncl"])
    apply_norm = luigi.BoolParameter(default=False)

    def requires(self):
        # everything requires self, specify by arguments
        return {
            "nominal_bins": ConstructInferenceBins.req(self),
            "shifted_arrays": MergeShiftArrays.req(self),  # , workflow="local"
            # "samples": CrossValidationPrep.req(self),
            # "predictions": PredictDNNScores.req(self),
            "model": PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout, category="All_Lep"),  #
            "Norm_factors": CalcNormFactors.req(self),
        }

    def output(self):
        return self.local_target("shift_unc.json")

    def store_parts(self):
        parts = tuple()
        if self.apply_norm:
            parts += ("apply_norm",)
        return super(GetShiftedYields, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + (self.gamma,) + parts

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        models = inp["model"]["collection"].targets[0]
        # loading all models once
        print("loading models")
        models_loaded = {fold: torch.load(models["fold_" + str(fold)]["model"].path) for fold in range(self.kfold)}
        nominal = inp["nominal_bins"].load()
        nominal_dict = nominal["nominal_dict"]

        # correction factors
        norm_factors = self.input()["Norm_factors"].load()
        proc_factor_dict = {"ttjets": "alpha", "Wjets": "beta", "QCD": "delta"}

        # processes to loop over
        nodes = list(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        # signal binning
        binning = self.config_inst.get_aux("signal_binning")

        # collecting results per shift
        out_dict = {}
        # evaluating weighted sum of events sorted into respective node by DNN
        shifts_long = self.unpack_shifts()
        for shift in tqdm(shifts_long):
            out_dict[shift] = {}
            shifted_yields = {}
            # only want to loop over backgrounds here
            for key in nodes[:-1]:
                scores, labels, weights = [], [], []
                real_procs = self.config_inst.get_aux("DNN_process_template")[self.category][key]
                for pr in real_procs:
                    if pr not in self.datasets_to_process:
                        continue
                    shift_dict = inp["shifted_arrays"]["collection"].targets[0]["{}_{}_{}".format(self.category, pr, shift)]  #
                    shift_ids = shift_dict["DNNId"].load()
                    shift_arr = shift_dict["array"].load()
                    shift_weight = shift_dict["weights"].load()
                    for fold, DNNId in enumerate(self.config_inst.get_aux("DNNId")):
                        # to get respective switched id per fold
                        j = -1 * DNNId
                        # each model should now predict labels for the validation data
                        model = models_loaded[fold]
                        X_test = shift_arr[shift_ids == j]
                        # we know the process
                        # y_test = np.array([[0, 0, 0]] * len(X_test))
                        # y_test[:, node] = 1
                        weight_test = shift_weight[shift_ids == j]

                        # pred_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(weight_test).float())
                        # pred_loader = torch.utils.data.DataLoader(dataset=pred_dataset, batch_size=len(X_test))
                        with torch.no_grad():
                            model.eval()
                            # for X_pred_batch, y_pred_batch, weight_pred_batch in pred_loader:
                            X_scores = model(torch.from_numpy(X_test)).softmax(dim=1)

                            scores.append(X_scores.numpy())
                            # labels.append(y_test)
                            weights.append(weight_test)

                # merge predictions labels, np.concatenate(labels),
                scores, weights = np.concatenate(scores), np.concatenate(weights)
                # in_node process shift yield
                for ii, node in enumerate(nodes):
                    if node == self.config_inst.get_aux("signal_process").replace("V", "W"):
                        mask = np.argmax(scores, axis=-1) == ii
                        for j, edge in enumerate(binning[:-1]):
                            scores_in_node = scores[mask][:, ii]
                            events_in_bin = (scores_in_node > binning[j]) & (scores_in_node < binning[j + 1])
                            weight_sum = np.sum(weights[mask][events_in_bin])
                            # print(shift, node, edge, len(weights[mask][events_in_bin]))
                            if self.apply_norm:
                                weight_sum *= norm_factors[proc_factor_dict[key]]
                            shifted_yields["DNN_Score_Node_{}_{}_proc_{}".format(node + "_" + str(edge), shift, key)] = weight_sum
                    else:
                        weight_sum = np.sum(weights[np.argmax(scores, axis=-1) == ii])
                        # normalisation factors to correctly model yields FIXME now done in datacard
                        if self.apply_norm:
                            weight_sum *= norm_factors[proc_factor_dict[key]]
                        # FIXME doing right assignment of yields per node?
                        # for now results look as expected
                        shifted_yields["DNN_Score_Node_{}_{}_proc_{}".format(node, shift, key)] = weight_sum
            for key2, rate in shifted_yields.items():
                nominal_key = key2.replace(shift + "_proc_", "")
                nominal_rate = float(nominal_dict[nominal_key])
                # print(nominal_key, nominal_rate, " shifted to ",key2, rate )
                # impact is shift relative to previous yields, then 1+impact
                # print(abs(rate - nominal_rate) / nominal_rate)
                # floats so it can dumped to json
                out_dict[shift][key2] = {"new_yields": float(shifted_yields[key2]), "nominal_yields": float(nominal_rate), "factor": float(abs(rate - nominal_rate) / nominal_rate)}

        # only wanting to use the average by Up/Down, saving as 1+average
        averaged_shifts = {}
        for key in out_dict.keys():
            # since we average out, only need to do it once for up, skip down
            if not "Up" in key or "PileUpDown" in key:
                continue
            averaged_shifts[key.replace("Up", "")] = dict()
            # catch the PileUp -> PileDown stuff
            get_down = out_dict[key.replace("Up", "Down").replace("PileDown", "PileUp")]
            get_up = out_dict[key]
            for yiel in get_up.keys():
                # skip QCD in signal nodes
                if self.config_inst.get_aux("signal_process").replace("V", "W") in yiel and "QCD" in yiel:
                    continue
                if get_up[yiel]["nominal_yields"] != 0.0:
                    average = (abs(get_up[yiel]["new_yields"] - get_up[yiel]["nominal_yields"]) + abs(get_down[yiel.replace("Up", "Down").replace("PileDown", "PileUp")]["new_yields"] - get_up[yiel]["nominal_yields"])) / 2 / get_up[yiel]["nominal_yields"]
                else:
                    average = 0.0
                averaged_shifts[key.replace("Up", "")][yiel.replace("Up", "")] = 1 + average
                print(yiel.replace("Up", ""), 1 + average)
        print(self.output().path)
        print(averaged_shifts.keys())
        self.output().dump(averaged_shifts)


class YieldPerMasspoint(CoffeaTask, DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    # jobs should be really small
    RAM = 5000
    hours = 3
    # to do systs here as well

    def requires(self):
        inp = {"files": WriteDatasetPathDict.req(self), "weights": CollectInputData.req(self), "masspoints": CollectMasspoints.req(self), "btagSF": CalcBTagSF.req(self, debug=False), "model": PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout, category="All_Lep", debug=False)}  #
        return inp

    def output(self):
        _, masspoints = self.load_masspoints()
        if self.shift == "systematic_shifts":
            # out = {}
            # for mp in masspoints:
            #     for shift in self.config_inst.get_aux("systematic_shifts"):
            #         out[shift+"_".join(mp)]={shift+"_scores": self.local_target(shift+"{}_{}_scores.npy".format(mp[0], mp[1])), shift+"_weights": self.local_target(shift+"{}_{}_weights.npy".format(mp[0], mp[1]))}
            # out={shift+"_".join(mp): {shift+"_scores": self.local_target(shift+"{}_{}_scores.npy".format(mp[0], mp[1])), shift+"_weights": self.local_target(shift+"{}_{}_weights.npy".format(mp[0], mp[1]))}
            # for mp in masspoints
            # for shift in self.config_inst.get_aux("systematic_shifts")
            # }
            out = {self.shift + "_".join(mp): self.local_target("{}_{}_score_weights.pkl".format(mp[0], mp[1])) for mp in masspoints}

        else:
            out = {self.shift + "_".join(mp): {"scores": self.local_target("{}_{}_scores.npy".format(mp[0], mp[1])), "weights": self.local_target("{}_{}_weights.npy".format(mp[0], mp[1]))} for mp in masspoints}
        return out

    def store_parts(self):
        parts = (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + (self.shift,) + (self.gamma,)
        if self.debug:
            parts = ("debug",) + parts
        return super(YieldPerMasspoint, self).store_parts() + parts

    def create_branch_map(self):
        # define job number according to number of files of the dataset that you want to process
        masspoints, _ = self.load_masspoints()
        job_number = len(masspoints)
        return list(range(job_number))

    def load_masspoints(self):
        if self.debug:
            # so we can generate one datacard quickly
            return [[800, 100]], [["800", "100"]]
        with open(self.config_inst.get_aux("masspoints")) as f:
            masspoints = json.load(f)
        floats = masspoints["masspoints"]
        strs = [[str(int(x)) for x in lis] for lis in floats]
        return floats, strs

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        masspoints, str_masspoints = self.load_masspoints()
        mp = masspoints[self.branch]  # self.create_branch_map()[]
        mp_str = str_masspoints[self.branch]
        data_dict = self.input()["files"]["dataset_dict"].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["files"]["dataset_path"].load()
        sum_gen_weights_dict = self.input()["weights"]["sum_gen_weights"].load()
        # declare processor
        processor_inst = ArrayExporter(self, Lepton=self.lepton_selection)
        # building together the respective strings to use for the coffea call
        files, job_number, job_number_dict, process_dict = self.load_job_dict()
        treename = self.lepton_selection
        out_list = []
        # we are spawning 800 jobs, one per masspoints, so each job will have to process all files
        # for i, file in range(job_number_dict.values()):
        # first file should have same contents as every signal file
        subset = job_number_dict[0]
        dataset = process_dict[0]
        proc = self.config_inst.get_process(dataset)
        # key for masspoint
        dataset_mp = "_".join([dataset] + mp_str)
        with up.open(data_path + "/" + subset) as file:
            primaryDataset = file["MetaData"]["primaryDataset"].array()[0]
            isData = file["MetaData"]["IsData"].array()[0]
            isFastSim = file["MetaData"]["IsFastSim"].array()[0]
            isSignal = proc.get_aux("isSignal")
            # assert all events with the same Xsec in scope with float precision
            xSec = proc.xsecs[13].nominal
            lumi = file["MetaData"]["Luminosity"].array()[0]
            # find the calculated btag SFs per file and save path
            # subsub = subset.split("/")[1]
            # btagSF = self.input()["btagSF"][treename + "_" + subsub].path
        # make sure all paths and shifts are defined
        if self.shift != "systematic_shifts":
            self.output()[self.shift + "_".join(mp_str)]["scores"].parent.touch()
        else:
            self.output()[self.shift + "_".join(mp_str)].parent.touch()
        # define paths to merged files same as done in CalcBtagSF
        all_path = self.input()["btagSF"][list(self.input()["btagSF"].keys())[0]]["weights"].parent.path + self.config_inst.get_aux("all_btag_SF")
        all_path_up = self.input()["btagSF"][list(self.input()["btagSF"].keys())[0]]["weights"].parent.path + self.config_inst.get_aux("all_btag_SF").replace("_T5qqqqWW", "_up_T5qqqqWW")
        all_path_down = self.input()["btagSF"][list(self.input()["btagSF"].keys())[0]]["weights"].parent.path + self.config_inst.get_aux("all_btag_SF").replace("_T5qqqqWW", "_down_T5qqqqWW")

        all_files = []
        for filename in sorted(job_number_dict.values()):
            # so we don't use files needed for extra statistics
            if "extra" in filename:
                continue
            all_files.append(data_path + "/" + filename)

        fileset = {
            dataset: {
                "files": all_files,
                "metadata": {"PD": primaryDataset, "isData": isData, "isFastSim": isFastSim, "isSignal": isSignal, "xSec": xSec, "Luminosity": lumi, "sumGenWeight": sum_gen_weights_dict[dataset_mp], "btagSF": all_path, "btagSF_up": all_path_up, "btagSF_down": all_path_down, "shift": self.shift, "masspoint": mp, "category": self.category},
            }
        }
        # call imported processor, magic happens here
        out = processor.run_uproot_job(
            fileset,
            treename=treename,
            processor_instance=processor_inst,
            # pre_executor=processor.futures_executor,
            # pre_args=dict(workers=32),
            executor=processor.iterative_executor,
            executor_args=dict(status=False),  # desc="", unit="Trolling"), # , desc="Trolling"
            # metadata_cache = 'MetaData',
            # schema=BaseSchema,),
            chunksize=10000,
        )

        # nominal stuff, only weights will alter for systematic shifts
        out_variables = out["arrays"][self.category + "_" + dataset]["hl"].value
        out_DNNIds = out["arrays"][self.category + "_" + dataset]["DNNId"].value
        out_weights = out["arrays"][self.category + "_" + dataset]["weights"].value
        scores, labels, weights = [], [], []

        # loading all models once
        print("loading models")
        models_loaded = {fold: torch.load(self.input()["model"]["fold_" + str(fold)]["model"].path) for fold in range(self.kfold)}

        # now using the produced output to predict events per masspoint
        for fold, Id in enumerate(self.config_inst.get_aux("DNNId")):
            # to get respective switched id per fold
            j = -1 * Id
            # each model should now predict labels for the validation data
            model = models_loaded[fold]
            X_test = out_variables[out_DNNIds == j]
            # we know the process
            # we only have signal here
            # y_test = np.array([[0, 0, 1]] * len(X_test))
            weights.append(out_weights[out_DNNIds == j])

            # pred_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(weight_test).float())
            # pred_loader = torch.utils.data.DataLoader(dataset=pred_dataset, batch_size=len(X_test))
            with torch.no_grad():
                model.eval()
                # for X_pred_batch, y_pred_batch, weight_pred_batch in pred_loader:
                X_scores = model(torch.from_numpy(X_test)).softmax(dim=1)
                scores.append(X_scores.numpy())

        scores = np.concatenate(scores)

        if self.shift == "systematic_shifts":
            df = pd.DataFrame()
            for shift in self.config_inst.get_aux("systematic_shifts") + self.config_inst.get_aux("systematic_shifts_signal"):
                shifted_weights = out["arrays"][self.category + "_" + dataset][shift].value
                # we need to rearrange weight array with DNNId so it matches the scores per event
                shifted_weights_DNNid = []
                for fold, Id in enumerate(self.config_inst.get_aux("DNNId")):
                    j = -1 * Id
                    shifted_weights_DNNid.append(shifted_weights[out_DNNIds == j])

                shifted_weights = np.concatenate(shifted_weights_DNNid)
                df[shift + "_weights"] = pd.Series(shifted_weights.tolist())
                # self.output()[shift+"_".join(mp_str)][shift+"_scores"].dump(np.concatenate(scores))
                # self.output()[shift+"_".join(mp_str)][shift+"_weights"].dump(np.concatenate(shifted_weights_DNNid))
            # scores are the same for each shift, then save pd dataframe
            df["scores"] = pd.Series(scores.tolist())
            df.to_pickle(self.output()[self.shift + "_".join(mp_str)].path)
        else:
            self.output()[self.shift + "_".join(mp_str)]["scores"].dump(scores)
            self.output()[self.shift + "_".join(mp_str)]["weights"].dump(np.concatenate(weights))


class DatacardPerMasspoint(CoffeaTask, DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    shifts = luigi.ListParameter(default=["systematic_shifts"])
    channel = luigi.ListParameter(default=["LeptonIncl"])
    kfold = luigi.IntParameter(default=2)
    QCD_unc = luigi.FloatParameter(default=1.0)
    signal_shifts = ["TotalUp", "TotalDown", "systematic_shifts", "JERUp", "JERDown"]
    apply_norm = luigi.BoolParameter(default=False)

    # base requirements for one spawned worker
    RAM = 5000
    hours = 1

    def requires(self):
        # FIXME apply norm
        inp = {"Signal_yields": YieldPerMasspoint.req(self, datasets_to_process=["T5qqqqWW"], shift="nominal", lepton_selection="LeptonIncl", workflow="local"), "Fixed_yields": ConstructInferenceBins.req(self, QCD_est=True, apply_norm=self.apply_norm), "Norm_factors": CalcNormFactors.req(self), "Shifted_yields": GetShiftedYields.req(self)}  #
        inp.update({"signal_" + shift: YieldPerMasspoint.req(self, datasets_to_process=["T5qqqqWW"], shift=shift, lepton_selection="LeptonIncl", workflow="local") for shift in self.shifts if shift in self.signal_shifts})
        return inp

    def output(self):
        _, masspoints = self.load_masspoints()
        out = {"_".join(mp): self.local_target("{}_{}_datacard.txt".format(mp[0], mp[1])) for mp in masspoints}
        return out

    def store_parts(self):
        parts = ("{}_shifts".format(len(self.shifts)),) + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + (self.gamma,)
        if self.debug:
            parts = ("debug",) + parts
        if self.QCD_unc != 1.0:
            parts = (str(self.QCD_unc),) + parts
        if self.apply_norm:
            parts = ("apply_norm",) + parts
        return super(DatacardPerMasspoint, self).store_parts() + parts

    def load_masspoints(self):
        if self.debug:
            # so we can generate one datacard quickly
            return [[800, 100]], [["800", "100"]]
        with open(self.config_inst.get_aux("masspoints")) as f:
            masspoints = json.load(f)
        floats = masspoints["masspoints"]
        strs = [[str(int(x)) for x in lis] for lis in floats]
        return floats, strs

    def get_susy_xsec(self):
        susy_xsec = os.path.expandvars("$ANALYSIS_BASE/config/SUSYCrossSections13TeVgluglu.json")
        with open(susy_xsec) as f:
            susy_xsec_unc = json.load(f)
        return susy_xsec_unc

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        from copy import deepcopy

        # ignoring warning from boost
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        print("\nstarting")
        signal_yields = self.input()["Signal_yields"]  # ["collection"].targets[0]
        # if self.debug:
        #     signal_yields = self.input()["Signal_yields"]["collection"].targets[0]
        fixed_yields = self.input()["Fixed_yields"]
        masspoints, str_masspoints = self.load_masspoints()
        # make dir for all datacards
        self.output()["_".join(str_masspoints[0])].parent.touch()
        proc_dict = self.config_inst.get_aux("DNN_process_template")[self.category]
        all_processes = list(proc_dict.keys())
        n_processes = len(all_processes)
        binning = self.config_inst.get_aux("signal_binning")
        # these values are the same for every datacard
        fixed_yields = self.input()["Fixed_yields"].load()
        fixed_bin_names = fixed_yields["bin_names"]
        process_names = fixed_yields["process_names"]
        process_numbers = fixed_yields["process_numbers"]
        fixed_rates = fixed_yields["rates"]
        fixed_rates_err = fixed_yields["rates_err"]
        data_bins = fixed_yields["data_bins"]
        data_obs = fixed_yields["data_obs"]
        shifted_yields = self.input()["Shifted_yields"].load()
        norm_factors = self.input()["Norm_factors"].load()
        lumi_unc = str(self.config_inst.get_aux("lumi_unc_per_year")[self.year])
        # ### kicking out everything but signal bins
        # datbins = len(data_bins)
        # for i in range(datbins-1, -1, -1):
        #     if not "T5qqqqWW" in data_bins[i]:
        #         del data_bins[i]
        #         del data_obs[i]
        # ###

        # bins = len(fixed_bin_names)
        # for i in range(bins-1, -1, -1):
        #     if not "T5qqqqWW" in fixed_bin_names[i]:
        #         del process_names[i]
        #         del fixed_bin_names[i]
        #         del fixed_rates[i]
        #         del process_numbers[i]
        # ###

        # uncertainties backgrounds
        xsec_dic = self.get_susy_xsec()
        # calculating xsec uncertainty
        scinums = {}
        for proc in list(self.config_inst.get_aux("DNN_process_template")[self.category].keys())[:-1]:
            real_procs = self.config_inst.get_aux("DNN_process_template")[self.category][proc]
            val = 0
            for pr in real_procs:
                conf_p = self.config_inst.get_process(pr)
                for child in conf_p.walk_processes():
                    val += child[0].xsecs[13]
            scinums[proc] = val
        process_unc = {}
        for key in scinums.keys():
            # data driven, not MC
            if key == "QCD":
                continue
            unc = []
            for i, name in enumerate(process_names):
                # QCD in signal gets predicted
                if name == key:  # and not ("QCD" in name): # and self.config_inst.get_aux("signal_process").replace("V", "W") in fixed_bin_names[i]):
                    num = scinums[key]
                    # absolute unc / nominal
                    unc.append(str(1 + num.u()[0] / num.n))
                else:
                    unc.append("-")
            process_unc["Xsec_" + key] = unc
        # for ib, bin_name in enumerate(fixed_bin_names):
        for ish, shift in enumerate(shifted_yields.keys()):
            if not shift + "Up" in self.config_inst.get_aux("systematic_shifts") and not shift.replace("Pile", "PileUp") + "Up" in self.shifts:
                continue
            process_unc.setdefault(shift, [])  # ["-"] * len(fixed_bin_names)
            # process_unc[shift] = []
            # build corresponding factors per bin and process
            for ib, bin_name in enumerate(fixed_bin_names):
                new_key = "{}_{}_proc_{}".format(bin_name, shift, process_names[ib])
                # we only want shift uncertainty for MC background, not QCD data-driven prediction
                if new_key in shifted_yields[shift].keys() and not "QCD_unc" in fixed_rates_err[ib]:
                    process_unc[shift].append(str(shifted_yields[shift][new_key]))
                else:
                    process_unc[shift].append("-")
        # Total only for signal, for background we use the singular sources
        # process_unc['Total'] = ["-"] * len(process_unc['Total'])
        # fully_correlated = process_unc[self.config_inst.get_aux("JEC_correlations")["fully"][0]]
        # for i in range(len(fully_correlated)):
        #     combined = 0
        #     for shift in self.config_inst.get_aux("JEC_correlations")["fully"]:
        #         if process_unc[shift][i] == "-":
        #             continue
        #         combined += float(process_unc[shift][i])**2
        #     fully_correlated[i] = str(np.sqrt(combined)).replace("0.0", "-")
        # for shift in self.config_inst.get_aux("JEC_correlations")["fully"]:
        #     process_unc.pop(shift)
        # process_unc["fully_correlated_JEC"] = fully_correlated
        # from IPython import embed; embed()
        # flat QCD unc since estimate is rather rough, position was saved in fixed_rate err
        # # process_unc["QCD_estimate"] = ["2." if "QCD_unc" == name else "-" for name in fixed_rates_err]
        # # fixed_rates_err = [err.replace('QCD_unc', "-") for err   in fixed_rates_err]
        max_scores = []
        # [("600", "100"),("1500","1000"), ("1500","1200"), ("1600","1100"), ("1700","1200"), ("1800","1300"), ("1900","100"), ("1900","800"), ("1900","1000"), ("2200","100"), ("2200","800"), ("1500", "500")]:
        for mp in tqdm(str_masspoints):
            rates, rates_err, bin_names = [], [], []
            mp_scores = signal_yields["nominal" + "_".join(mp)]["scores"].load()
            mp_weights = signal_yields["nominal" + "_".join(mp)]["weights"].load()
            # # so large cross sections don't mess up limit plottinh
            # if int(mp[0]) < 1410:
            #     mp_weights /= 100
            for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                mask = np.argmax(mp_scores, axis=1) == node
                scores_in_node = mp_scores[mask][:, node]
                weights_in_node = mp_weights[mask]
                if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                    hist_sig_region = bh.Histogram(bh.axis.Variable(binning), storage=bh.storage.Weight())
                    hist_sig_region.fill(scores_in_node, weight=weights_in_node)
                    errors_sig = np.sqrt(hist_sig_region.variances())
                    counts_sig = hist_sig_region.counts()
                    # for j, edge in enumerate(binning[:-1]):
                    #     # we need to find weights in each bin
                    #     events_in_bin = (scores_in_node > binning[j]) & (scores_in_node < binning[j + 1])
                    #     weights_in_bin = mp_weights[mask][events_in_bin]
                    rates.extend([str(c) for c in counts_sig])
                    # no nans for 0 counts
                    rates_err.extend([str(1 + np.nan_to_num(c, 1.0)) for c in errors_sig / counts_sig])
                    bin_names.extend(["DNN_Score_Node_" + key + "_" + str(edge) for edge in binning[:-1]])
                else:
                    # only one bin
                    hist_bkg_region = bh.Histogram((1, 0, 1), storage=bh.storage.Weight())
                    hist_bkg_region.fill(scores_in_node, weight=weights_in_node)
                    errors_bkg = np.sqrt(hist_bkg_region.variances())
                    rates.append(str(hist_bkg_region.counts()[0]))
                    # if we see no signal misclassified, do this
                    if hist_bkg_region.counts()[0] != 0:
                        rates_err.append(str(1 + errors_bkg[0] / hist_bkg_region.counts()[0]))
                    else:
                        rates_err.append(str(1.0))
                    bin_names.append("DNN_Score_Node_" + key)

            # print("Max signal score: ", mp, " ", np.max(mp_scores[:,-1]))
            max_scores.append(np.max(mp_scores[:, -1]))
            # construct uncertainty rows
            # deepcopy dict in each iteration so we don't print all masspoints in last datacard
            cp_process_unc = deepcopy(process_unc)
            for key in cp_process_unc.keys():
                cp_process_unc[key] = ["-"] * len(bin_names) + cp_process_unc[key]
            # add fast sim SF as well, will only contain signal
            for fast in self.config_inst.get_aux("systematic_shifts_signal"):
                if not "Up" in fast:
                    continue
                cp_process_unc[fast.replace("Up", "")] = ["-"] * len(cp_process_unc[key])
            # to account for missing SF files, assign maximum of 10% uncertainty to account for missing signal
            if self.year == "2016":
                cp_process_unc["missing_lepSF"] = ["1.1"] * len(bin_names) + ["-"] * len(fixed_bin_names)
            # This is just flat unc for signal cp_process_unc["Flat_T5qqqqWW_Unc"] = ["1.20"] * len(bin_names) + ["-"] * len(fixed_bin_names)
            # FIXME old analysis done by Uncertainty (NLO + NLL) [%]
            cp_process_unc["Xsec_T5qqqqWW_" + mp[0]] = [str(1 + xsec_dic["Uncertainty (NNLOapprox + NNLL) [%]"][mp[0]] / 100)] * len(bin_names) + ["-"] * len(fixed_bin_names)
            # doing signal systs now per shift
            signal_bin_shifts = {}
            for shift in self.shifts:
                if shift not in self.signal_shifts:
                    continue
                shift_inp = self.input()["signal_" + shift]  # ["collection"].targets[0]
                # if self.debug:
                #     shift_inp = self.input()['signal_'+shift]["collection"].targets[0]
                if shift == "systematic_shifts":
                    df_pickle = pd.read_pickle(shift_inp[shift + "_".join(mp)].path)
                    signal_shift_scores = np.array(df_pickle["scores"].values.tolist())
                    for sub_shift in self.config_inst.get_aux("systematic_shifts") + self.config_inst.get_aux("systematic_shifts_signal"):
                        # signal_shift_scores = shift_inp[sub_shift+"_".join(mp)][sub_shift+"_scores"].load()
                        signal_shift_weights = np.array(df_pickle[sub_shift + "_weights"].values.tolist())
                        # shift_inp[sub_shift+"_".join(mp)][sub_shift+"_weights"].load()
                        for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                            mask = np.argmax(signal_shift_scores, axis=1) == node
                            scores_in_node = signal_shift_scores[mask][:, node]
                            weights_in_node = signal_shift_weights[mask]
                            if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                                for j, edge in enumerate(binning[:-1]):
                                    events_in_bin = (scores_in_node > binning[j]) & (scores_in_node < binning[j + 1])
                                    weights_in_bin = signal_shift_weights[mask][events_in_bin]
                                    signal_bin_shifts["DNN_Score_Node_" + key + "_" + str(edge) + "_" + sub_shift] = np.sum(weights_in_bin)
                            else:
                                signal_bin_shifts["DNN_Score_Node_" + key + "_" + sub_shift] = np.sum(signal_shift_weights[mask])
                else:
                    signal_shift_scores = shift_inp[shift + "_".join(mp)]["scores"].load()
                    signal_shift_weights = shift_inp[shift + "_".join(mp)]["weights"].load()
                    for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                        mask = np.argmax(signal_shift_scores, axis=1) == node
                        scores_in_node = signal_shift_scores[mask][:, node]
                        weights_in_node = signal_shift_weights[mask]
                        if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                            for j, edge in enumerate(binning[:-1]):
                                events_in_bin = (scores_in_node > binning[j]) & (scores_in_node < binning[j + 1])
                                weights_in_bin = signal_shift_weights[mask][events_in_bin]
                                signal_bin_shifts["DNN_Score_Node_" + key + "_" + str(edge) + "_" + shift] = np.sum(weights_in_bin)
                                # print(shift, key, edge, np.sum(weights_in_bin))
                        else:
                            signal_bin_shifts["DNN_Score_Node_" + key + "_" + shift] = np.sum(signal_shift_weights[mask])

            # looping over our signal bin names
            for ib, name in enumerate(bin_names):
                for shift in self.shifts:
                    # so we don't calc shifts for variations not defined in signal fast sim samples
                    if shift not in self.signal_shifts:
                        continue
                    if shift == "systematic_shifts":
                        for sub_shift in self.config_inst.get_aux("systematic_shifts") + self.config_inst.get_aux("systematic_shifts_signal"):
                            # since we average out, only need to do it once for up, skip down, and Prefire is not defined for signal
                            if not "Up" in sub_shift or "PreFireWeight" in sub_shift:
                                continue
                            else:
                                get_up = signal_bin_shifts[name + "_" + sub_shift]
                                get_down = signal_bin_shifts[name + "_" + sub_shift.replace("Up", "Down")]
                                get_nominal = float(rates[ib])
                                syst_in_bin = (abs(get_up - get_nominal) + abs(get_down - get_nominal)) / 2 / get_nominal
                                # catching nans if there is no signal in certain bins
                                cp_process_unc[sub_shift.replace("Up", "")][ib] = str(1 + np.nan_to_num(syst_in_bin, 0))
                    else:
                        # since we average out, only need to do it once for up, skip down, and catch PileUpDown
                        if not "Up" in shift or "PileUpDown" in shift:
                            continue
                        else:
                            get_up = signal_bin_shifts[name + "_" + shift]
                            get_down = signal_bin_shifts[name + "_" + shift.replace("Up", "Down").replace("PileDown", "PileUp")]
                            get_nominal = float(rates[ib])
                            syst_in_bin = (abs(get_up - get_nominal) + abs(get_down - get_nominal)) / 2 / get_nominal
                            # catching nans if there is no signal in certain bins
                            cp_process_unc[shift.replace("Up", "")][ib] = str(1 + np.nan_to_num(syst_in_bin, 0))  # .replace("Pile", "PileUp")
                            # print(name, shift, syst_in_bin)
                # writing signal stat unc per bin
                entry = ["-"] * (len(bin_names) + len(fixed_bin_names))
                entry[ib] = rates_err[ib]
                cp_process_unc["Stat_unc_{}_T5qqqqWW_{}_{}".format(name, mp[0], mp[1])] = entry
            # one nuissance paramter per process statistical uncertainty
            # we want to do a syst per proc per bin
            # QCD estimation has a different prediction
            QCD_estimate_unc = ["-"] * (len(bin_names) + len(fixed_bin_names))
            fixed_lumi_bins = []
            for i, bin in enumerate(fixed_bin_names):
                entry = ["-"] * (len(bin_names) + len(fixed_bin_names))
                if "QCD_unc" not in fixed_rates_err[i]:
                    entry[len(bin_names) + i] = fixed_rates_err[i]
                    cp_process_unc["Stat_unc_{}_{}".format(bin, process_names[i])] = entry
                    fixed_lumi_bins.append(lumi_unc)
                else:
                    # QCD_estimate_unc[len(bin_names) + i] = "{}".format(1 + self.QCD_unc)
                    QCD_estimate_unc[len(bin_names) + i] = "{}".format(1 + float(fixed_rates_err[i].replace("QCD_unc ", "")))
                    fixed_lumi_bins.append("-")
            cp_process_unc["QCD_estimate_unc"] = QCD_estimate_unc
            bins_total = bin_names + fixed_bin_names
            # toggle off if the contribution of signal events is unwanted
            # bin_mask = ["T5qqqqWW_0." not in b for b in bins_total]
            bin_mask = [True for b in bins_total]
            rates_total = rates + fixed_rates
            process_total = ["T5qqqqWW_{}_{}".format(mp[0], mp[1])] * len(bin_names) + process_names
            numbers_total = ["0"] * len(bin_names) + process_numbers
            lumi_total = [lumi_unc] * len(rates_err) + fixed_lumi_bins
            masked_rates = [rat for i, rat in enumerate(rates_total) if bin_mask[i]]
            # dedicated datacard per masspoint
            with open(self.output()["_".join(mp)].path, "w") as datacard:
                datacard.write("## Datacard for (signal mGlu {} mNeu {})\n".format(mp[0], mp[1]))
                datacard.write("imax {}  number of channels \n".format(len(bin_names)))
                # datacard.write("imax {}  number of channels \n".format(4))
                datacard.write("jmax %i  number of processes -1 \n" % (n_processes - 1))  # +1 QCD extra
                datacard.write("kmax * number of nuisance parameters (sources of systematical uncertainties) \n")  # .format(1)
                datacard.write("---\n")

                # write data observed
                # datacard.write("bin " + " ".join(data_bins) + "\n")
                # datacard.write("observation " + " ".join(data_obs) + "\n")
                datacard.write("bin " + " ".join([rat for i, rat in enumerate(data_bins) if bin_mask[i]]) + "\n")
                datacard.write("observation " + " ".join([rat for i, rat in enumerate(data_obs) if bin_mask[i]]) + "\n")
                datacard.write("---\n")

                # MC
                # datacard.write("bin " + " ".join(bin_names) + " " + " ".join(fixed_bin_names) + "\n")
                # datacard.write("process " + "T5qqqqWW_{}_{} ".format(mp[0], mp[1]) * len(bin_names) + " ".join(process_names) + "\n")
                # datacard.write("process " + "0 " * len(bin_names) + " ".join(process_numbers) + "\n")
                # datacard.write("rate " + " ".join(rates) + " " + " ".join(fixed_rates) + "\n")

                datacard.write("bin " + " ".join([rat for i, rat in enumerate(bins_total) if bin_mask[i]]) + "\n")
                datacard.write("process " + " ".join([rat for i, rat in enumerate(process_total) if bin_mask[i]]) + "\n")
                datacard.write("process " + " ".join([rat for i, rat in enumerate(numbers_total) if bin_mask[i]]) + "\n")
                datacard.write("rate " + " ".join([rat for i, rat in enumerate(rates_total) if bin_mask[i]]) + "\n")

                # systematics
                datacard.write("---\n")
                datacard.write("lumi_{} lnN ".format(self.year) + " ".join([rat for i, rat in enumerate(lumi_total) if bin_mask[i]]) + "\n")
                # datacard.write("lumi lnN " + "1.023 " * len(rates_err) + " ".join(fixed_lumi_bins) + "\n")
                # datacard.write("Stat_Unc " + " ".join(fixed_rates_err) + "\n")
                # each process has xsec uncertainties, dict filled beforehand and just written here
                for key in cp_process_unc.keys():
                    datacard.write("{}_{} lnN ".format(key.replace("Total", "Total_JEC"), self.year) + " ".join([rat for i, rat in enumerate(cp_process_unc[key]) if bin_mask[i]]) + "\n")
                # Uncertainty signal to account for missing files in 2016
                if self.year == "2016":
                    datacard.write("FastSFfiles2016 lnN " + " ".join(["1.10" if num == "0" else "-" for num in numbers_total]))

                # doing background estimation in fit, not beforehand
                if not self.apply_norm:
                    datacard.write("---\n")
                    # format name rateParam channel process initial value
                    datacard.write("alpha_{} rateParam * {} {} [0,5] \n".format(self.year, all_processes[0], str(norm_factors["alpha"])))  #   [0.01,100]
                    datacard.write("beta_{} rateParam * {} {} [0,5] \n".format(self.year, all_processes[1], str(norm_factors["beta"])))  #   [0.01,100]
                    datacard.write("delta_{} rateParam * {} {} [0,5] \n".format(self.year, all_processes[2], str(norm_factors["delta"])))  #   [0.01,100]
                    # datacard.write("alpha_{} rateParam * {} 1. [0,5] \n".format(self.year, all_processes[0]))  #   [0.01,100]
                    # datacard.write("beta_{} rateParam * {} 1. [0,5] \n".format(self.year, all_processes[1]))  #   [0.01,100]
                    # datacard.write("delta_{} rateParam * {} 1. [0,5] \n".format(self.year, all_processes[2]))  #   [0.01,100]
                # doing stat unc per command
                # datacard.write("* autoMCStats 2 0")
        print("least signal score:", np.min(np.array(max_scores)), masspoints[np.argmin(np.array(max_scores))])
        if self.apply_norm:
            print("\nAtttention, no rate Params applied\n")

        # print only backgrounds
        for k in process_unc.keys():  # cp_
            mask = np.array(process_unc[k]) != "-"
            arr = np.asarray(np.array(process_unc[k])[mask], dtype=float)
            print(k, "&", np.round(np.min(arr), 4), "&", np.round(np.max(arr), 4), "&", np.round(np.median(arr), 4), "\\\\")

        # all
        for k in cp_process_unc.keys():
            mask = np.array(cp_process_unc[k]) != "-"
            arr = np.asarray(np.array(cp_process_unc[k])[mask], dtype=float)
            print(k, "&", np.round(np.min(arr), 4), "&", np.round(np.max(arr), 4), "&", np.round(np.median(arr), 4), "\\\\")


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
        proc_dict = self.config_inst.get_aux("DNN_process_template")[self.category]
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
            datacard.write("imax {}  number of channels \n".format(n_bins + 1))  # QCD now
            datacard.write("jmax %i  number of processes -1 \n" % (len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - 1 + 1))  # QCD extra
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
            datacard.write("lumi lnN " + "1.223 " * len(inp["bin_names"]) + "\n")  # each process has a score in each process node
            # FIXME lumi is 1.023. all other systematics are commented out
            # for key in process_unc.keys():
            #    datacard.write("Xsec_{} lnN ".format(key) + " ".join(process_unc[key]) + "\n")

            # doing background estimation
            # datacard.write("---\n")
            # format name rateParam channel process initial value FIXME
            # datacard.write("alpha rateParam * {} 1 \n".format(all_processes[0]))
            # datacard.write("beta rateParam * {} 1 \n".format(all_processes[1]))
            # maybe Calcnorm as initial values
