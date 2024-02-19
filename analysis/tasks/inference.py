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
from coffea import processor, hist
import torch
import sklearn as sk
from tqdm.auto import tqdm
from functools import total_ordering
import json

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

    def requires(self):
        return {
            "data_predicted": PredictDNNScores.req(self, workflow="local"),
            "inputs": CrossValidationPrep.req(self, kfold=self.kfold),
            "QCD_pred": EstimateQCDinSR.req(self),
            "Norm_factors": CalcNormFactors.req(self),
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

        binning = self.config_inst.get_aux("signal_binning")
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
        bin_names, process_names, process_numbers, rates = [], [], [], []
        for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            # leave CR as one bin, do a binning in SR to resolve high signal count regions better
            if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                for i, proc in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                    if proc == "T5qqqqWW":
                        continue
                    find_proc = labels[:, i] == 1
                    MC_mask = np.argmax(MC_scores[find_proc], axis=1) == node
                    scores_in_node = MC_scores[find_proc][MC_mask][:, node]
                    for j, edge in enumerate(binning[:-1]):
                        # we need to find weights in each bin
                        events_in_bin = (scores_in_node > binning[j]) & (scores_in_node < binning[j + 1])
                        weights_in_bin = weights[find_proc][MC_mask][events_in_bin]
                        bin_names.append("DNN_Score_Node_" + key + "_" + str(edge))
                        process_names.append(proc)
                        process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - (i + 1)))
                        rates.append(str(np.sum(weights_in_bin)))
                        # QCD on top, append QCD on its own
                        bin_names.append("DNN_Score_Node_" + key + "_" + str(edge))
                        process_names.append("QCD")
                        process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())))
                        rates.append(str(QCD_estimate[i]))

            else:
                for i, proc in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                    if proc == "T5qqqqWW":
                        continue
                    find_proc = labels[:, i] == 1
                    MC_mask = np.argmax(MC_scores[find_proc], axis=1) == node
                    # proc_dict.update({key: np.sum(weights[find_proc][MC_mask])})

                    bin_names.append("DNN_Score_Node_" + key)
                    process_names.append(proc)
                    # signal is last process in proc_dict, so we want it to be 0
                    process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys()) - (i + 1)))
                    rates.append(str(np.sum(weights[find_proc][MC_mask])))
                # append QCD on its own
                QCD_mask = np.argmax(QCD_scores, axis=1) == j
                QCD_counts_bin = np.sum(QCD_weights[QCD_mask])
                bin_names.append("DNN_Score_Node_" + key)
                process_names.append("QCD")
                # just put QCD on top of all other processes
                process_numbers.append(str(len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())))
                rates.append(str(QCD_counts_bin))

                # klen = max(len(samp) for samp in sampNames)
                # catDict = {"SR_MB" : "S", "CR_MB" : "C", "SR_SB" : "S2", "CR_SB" : "C2", "SR_SB_NB1i" : "S3", "CR_SB_NB1i" : "C3"}
                # np.random.seed(42)
        print(bin_names, process_names, process_numbers, rates)

        nominal_dict = {}
        for i, rate in enumerate(rates):
            nominal_dict["{}_{}".format(bin_names[i], process_names[i])] = rate

        inference_bins = {"data_bins": data_bins, "data_obs": data_obs, "bin_names": bin_names, "process_names": process_names, "process_numbers": process_numbers, "rates": rates, "nominal_dict": nominal_dict}

        # optimising binning?
        not_signal = labels[:, -1] == 0
        MC_scores_bg_in_signode = MC_scores[not_signal][:, -1]
        for step in np.arange(0.99, 1, 0.0001):
            print(np.round(step, 4), ": ", np.sum(MC_scores_bg_in_signode[MC_scores_bg_in_signode > step]))
        print("data max signal sore: ", np.max(data_scores[:, -1]), " and second highest: ", np.max(data_scores[:, -1][data_scores[:, -1] < np.max(data_scores[:, -1])]))
        print("Max signal score:", np.max(MC_scores[:, -1]))
        # finish task
        self.output().dump(inference_bins)


class GetShiftedYields(CoffeaTask, DNNTask):
    shifts = luigi.ListParameter(default=["PreFireWeightUp"])
    channel = luigi.ListParameter(default=["Muon", "Electron"])

    def requires(self):
        # everything requires self, specify by arguments
        return {
            "nominal_bins": ConstructInferenceBins.req(self),
            "shifted_arrays": MergeShiftArrays.req(self),  # , workflow="local"
            "samples": CrossValidationPrep.req(self),
            "predictions": PredictDNNScores.req(self),
            "model": PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout),
        }

    def output(self):
        return self.local_target("shift_unc.json")

    def store_parts(self):
        return super(GetShiftedYields, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        models = inp["model"]["collection"].targets[0]

        nominal = inp["nominal_bins"].load()
        nominal_dict = nominal["nominal_dict"]

        # omitting signal
        nodes = list(self.config_inst.get_aux("DNN_process_template")[self.category].keys())[:-1]

        # collecting results per shift
        out_dict = {}
        # evaluating weighted sum of events sorted into respective node by DNN
        shifts_long = self.unpack_shifts()
        for shift in tqdm(shifts_long):
            out_dict[shift] = {}
            shifted_yields = {}
            for node, key in enumerate(nodes):
                scores, labels, weights = [], [], []
                real_procs = self.config_inst.get_aux("DNN_process_template")[self.category][key]
                for pr in real_procs:
                    if pr not in self.datasets_to_process:
                        continue
                    shift_dict = inp["shifted_arrays"]["collection"].targets[0]["{}_{}_{}".format(self.category, pr, shift)]
                    shift_ids = shift_dict["DNNId"].load()
                    shift_arr = shift_dict["array"].load()
                    shift_weight = shift_dict["weights"].load()
                    for fold, DNNId in enumerate(self.config_inst.get_aux("DNNId")):
                        # to get respective switched id per fold
                        j = -1 * DNNId
                        # each model should now predict labels for the validation data
                        model = torch.load(models["fold_" + str(fold)]["model"].path)
                        X_test = shift_arr[shift_ids == j]
                        # we know the process
                        y_test = np.array([[0, 0, 0]] * len(X_test))
                        y_test[:, node] = 1
                        weight_test = shift_weight[shift_ids == j]

                        pred_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(weight_test).float())
                        pred_loader = torch.utils.data.DataLoader(dataset=pred_dataset, batch_size=len(X_test))

                        with torch.no_grad():
                            model.eval()
                            for X_pred_batch, y_pred_batch, weight_pred_batch in pred_loader:
                                X_scores = model(X_pred_batch)

                                scores.append(X_scores.numpy())
                                labels.append(y_pred_batch)
                                weights.append(weight_pred_batch)

                # merge predictions
                scores, labels, weights = np.concatenate(scores), np.concatenate(labels), np.concatenate(weights)

                # in_node process shift yield
                for ii, node in enumerate(nodes):
                    weight_sum = np.sum(weights[np.argmax(scores, axis=-1) == ii])
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
            if not "Up" in key:
                continue
            else:
                averaged_shifts[key.replace("Up", "")] = dict()
                get_down = out_dict[key.replace("Up", "Down")]
                get_up = out_dict[key]
                for yiel in get_up.keys():
                    average = (abs(get_up[yiel]["new_yields"] - get_up[yiel]["nominal_yields"]) + abs(get_down[yiel.replace("Up", "Down")]["new_yields"] - get_up[yiel]["nominal_yields"])) / 2 / get_up[yiel]["nominal_yields"]
                    averaged_shifts[key.replace("Up", "")][yiel.replace("Up", "")] = 1 + average
                    print(yiel.replace("Up", ""), 1 + average)

        self.output().dump(averaged_shifts)


class YieldPerMasspoint(CoffeaTask, DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    # jobs should be really small
    RAM = 500
    hours = 1

    def requires(self):
        inp = {"files": WriteDatasetPathDict.req(self), "weights": CollectInputData.req(self), "masspoints": CollectMasspoints.req(self), "btagSF": CalcBTagSF.req(self), "model": PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout)}
        return inp

    def output(self):
        _, masspoints = self.load_masspoints()
        out = {"_".join(mp): {"scores": self.local_target("{}_{}_scores.npy".format(mp[0], mp[1])), "weights": self.local_target("{}_{}_weights.npy".format(mp[0], mp[1]))} for mp in masspoints}
        return out

    def store_parts(self):
        return super(YieldPerMasspoint, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    def create_branch_map(self):
        # define job number according to number of files of the dataset that you want to process
        masspoints, _ = self.load_masspoints()
        job_number = len(masspoints)
        if self.debug:
            job_number = 1
        return list(range(job_number))

    def load_masspoints(self):
        with open(self.config_inst.get_aux("masspoints")) as f:
            masspoints = json.load(f)
        floats = masspoints["masspoints"]
        strs = [[str(int(x)) for x in lis] for lis in floats]
        return floats, strs

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        masspoints, str_masspoints = self.load_masspoints()
        mp = masspoints[self.branch]
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

        all_files, all_btagSF = [], []
        for filename in job_number_dict.values():
            all_files.append(data_path + "/" + filename)
            all_btagSF.append(self.input()["btagSF"][treename + "_" + filename.split("/")[1]].load())
        # get joined btagSF file for skimming all signal at once
        all_btagSF = np.concatenate(all_btagSF)
        self.output()["_".join(mp_str)]["scores"].parent.touch()
        all_path = self.output()["_".join(mp_str)]["scores"].parent.path + "/all_btagSF_T5qqqqWW.npy"
        if not os.path.isfile(all_path):
            np.save(all_path, all_btagSF)
        fileset = {
            dataset: {
                "files": all_files,
                "metadata": {"PD": primaryDataset, "isData": isData, "isFastSim": isFastSim, "isSignal": isSignal, "xSec": xSec, "Luminosity": lumi, "sumGenWeight": sum_gen_weights_dict[dataset], "btagSF": all_path, "shift": self.shift, "masspoint": mp, "category": self.category},
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

        out_variables = out["arrays"][self.category + "_" + dataset]["hl"].value
        out_DNNIds = out["arrays"][self.category + "_" + dataset]["DNNId"].value
        out_weights = out["arrays"][self.category + "_" + dataset]["weights"].value

        scores, labels, weights = [], [], []
        # now using the produced output to predict events per masspoint
        for fold, Id in enumerate(self.config_inst.get_aux("DNNId")):
            # to get respective switched id per fold
            j = -1 * Id
            # each model should now predict labels for the validation data
            model = torch.load(self.input()["model"]["fold_" + str(fold)]["model"].path)
            X_test = out_variables[out_DNNIds == j]
            # we know the process
            # we only have signal here
            y_test = np.array([[0, 0, 1]] * len(X_test))
            weight_test = out_weights[out_DNNIds == j]

            pred_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(weight_test).float())
            pred_loader = torch.utils.data.DataLoader(dataset=pred_dataset, batch_size=len(X_test))

            with torch.no_grad():
                model.eval()
                for X_pred_batch, y_pred_batch, weight_pred_batch in pred_loader:
                    X_scores = model(X_pred_batch)

                    scores.append(X_scores.numpy())
                    labels.append(y_pred_batch)
                    weights.append(weight_pred_batch)

        self.output()["_".join(mp_str)]["scores"].dump(np.concatenate(scores))
        self.output()["_".join(mp_str)]["weights"].dump(np.concatenate(weights))


class DatacardPerMasspoint(CoffeaTask, DNNTask):
    shifts = luigi.ListParameter(default=["PreFireWeightUp"])
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    kfold = luigi.IntParameter(default=2)

    def requires(self):
        return {"Signal_yields": YieldPerMasspoint.req(self, datasets_to_process=["T5qqqqWW"]), "Fixed_yields": ConstructInferenceBins.req(self), "Shifted_yields": GetShiftedYields.req(self)}

    def output(self):
        _, masspoints = self.load_masspoints()
        out = {"_".join(mp): self.local_target("{}_{}_datacard.txt".format(mp[0], mp[1])) for mp in masspoints}
        return out

    def store_parts(self):
        return super(DatacardPerMasspoint, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    def load_masspoints(self):
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

        print("\nstarting")
        signal_yields = self.input()["Signal_yields"]["collection"].targets[0]
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
        data_bins = fixed_yields["data_bins"]
        data_obs = fixed_yields["data_obs"]
        shifted_yields = self.input()["Shifted_yields"].load()
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
            unc = []
            for name in process_names:
                if name == key:
                    num = scinums[key]
                    # absolute unc / nominal
                    unc.append(str(1 + num.u()[0] / num.n))
                else:
                    unc.append("-")
            process_unc["Xsec_" + key] = unc
        # FIXME what values to put into syst datacard rows, 1+ shift, +- shift??
        # for ib, bin_name in enumerate(fixed_bin_names):
        for ish, shift in enumerate(shifted_yields.keys()):
            process_unc.setdefault(shift, [])  # ["-"] * len(fixed_bin_names)
            # process_unc[shift] = []
            # build corresponding factors per bin and process
            for ib, bin_name in enumerate(fixed_bin_names):
                new_key = "{}_{}_proc_{}".format(bin_name, shift, process_names[ib])
                if new_key in shifted_yields[shift].keys():
                    process_unc[shift].append(str(shifted_yields[shift][new_key]))
                else:
                    process_unc[shift].append("-")
        max_scores = []
        for mp in tqdm(str_masspoints):
            rates, bin_names = [], []
            mp_scores = signal_yields["_".join(mp)]["scores"].load()
            mp_weights = signal_yields["_".join(mp)]["weights"].load()
            # so large cross sections don't mess up limit plottinh
            if int(mp[0]) < 1410:
                mp_weights /= 100
            for node, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
                mask = np.argmax(mp_scores, axis=1) == node
                scores_in_node = mp_scores[mask][:, node]
                weights_in_node = mp_weights[mask]
                if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                    for j, edge in enumerate(binning[:-1]):
                        # we need to find weights in each bin
                        events_in_bin = (scores_in_node > binning[j]) & (scores_in_node < binning[j + 1])
                        weights_in_bin = mp_weights[mask][events_in_bin]
                        bin_names.append("DNN_Score_Node_" + key + "_" + str(edge))
                        rates.append(str(np.sum(weights_in_bin)))
                else:
                    rates.append(str(np.sum(mp_weights[mask])))
                    bin_names.append("DNN_Score_Node_" + key)

            # print("Max signal score: ", mp, " ", np.max(mp_scores[:,-1]))
            max_scores.append(np.max(mp_scores[:, -1]))
            # construct uncertainty rows
            # deepcopy dict in each iteration so we don't print all masspoints in last datacard
            cp_process_unc = deepcopy(process_unc)
            for key in cp_process_unc.keys():
                cp_process_unc[key] = ["-"] * len(bin_names) + cp_process_unc[key]
            # FIXME old analysis done by Uncertainty (NLO + NLL) [%]
            # cp_process_unc["Xsec_T5qqqqWW_" + mp[0]] = [str(1 + xsec_dic["Uncertainty (NNLOapprox + NNLL) [%]"][mp[0]] / 100)] * len(bin_names) + ["-"] * len(fixed_bin_names)
            cp_process_unc["Flat_T5qqqqWW_Unc"] = ["1.20"] * len(bin_names) + ["-"] * len(fixed_bin_names)
            with open(self.output()["_".join(mp)].path, "w") as datacard:
                datacard.write("## Datacard for (signal mGlu {} mNeu {})\n".format(mp[0], mp[1]))
                datacard.write("imax {}  number of channels \n".format(len(bin_names)))
                datacard.write("jmax %i  number of processes -1 \n" % (n_processes - 1 + 1))  # QCD extra
                datacard.write("kmax * number of nuisance parameters (sources of systematical uncertainties) \n")  # .format(1)
                datacard.write("---\n")

                # write data observed
                datacard.write("bin " + " ".join(data_bins) + "\n")
                datacard.write("observation " + " ".join(data_obs) + "\n")
                datacard.write("---\n")

                # MC
                datacard.write("bin " + " ".join(bin_names) + " " + " ".join(fixed_bin_names) + "\n")
                datacard.write("process " + "T5qqqqWW_{}_{} ".format(mp[0], mp[1]) * len(bin_names) + " ".join(process_names) + "\n")
                datacard.write("process " + "0 " * len(bin_names) + " ".join(process_numbers) + "\n")
                datacard.write("rate " + " ".join(rates) + " " + " ".join(fixed_rates) + "\n")

                # systematics
                datacard.write("---\n")
                datacard.write("lumi lnN " + "1.023 " * (len(bin_names) + len(fixed_bin_names)) + "\n")  # each process has
                # xsec uncertainties, dict filled beforehand and just written here
                for key in cp_process_unc.keys():
                    datacard.write("{} lnN ".format(key.replace("Total", "Total_JEC")) + " ".join(cp_process_unc[key]) + "\n")

                # doing background estimation
                datacard.write("---\n")
                # format name rateParam channel process initial value
                datacard.write("alpha rateParam * {} 1 \n".format(all_processes[0]))
                datacard.write("beta rateParam * {} 1 \n".format(all_processes[1]))
        print("least signal score:", np.min(np.array(max_scores)), masspoints[np.argmin(np.array(max_scores))])


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
            datacard.write("lumi lnN " + "1.023 " * len(inp["bin_names"]) + "\n")  # each process has a score in each process node
            for key in process_unc.keys():
                datacard.write("Xsec_{} lnN ".format(key) + " ".join(process_unc[key]) + "\n")

            # doing background estimation
            datacard.write("---\n")
            # format name rateParam channel process initial value
            datacard.write("alpha rateParam * {} 1 \n".format(all_processes[0]))
            datacard.write("beta rateParam * {} 1 \n".format(all_processes[1]))
            # maybe Calcnorm as initial values
