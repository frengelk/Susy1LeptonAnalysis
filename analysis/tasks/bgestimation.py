# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import uproot as up
import coffea
from tqdm import tqdm
import torch

# going to plot
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.backends.backend_pdf import PdfPages
import boost_histogram as bh
import mplhep as hep


# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask, AntiProcessor
from tasks.makefiles import WriteDatasetPathDict, WriteDatasets
from tasks.base import HTCondorWorkflow, DNNTask
from tasks.grouping import MergeArrays
from tasks.multiclass import PytorchMulticlass, PredictDNNScores, PytorchCrossVal, CalcNormFactors


class FitQCDContribution(CoffeaTask):
    channel = luigi.ListParameter(default=["LeptonIncl"])

    def requires(self):
        inp = {
            cat: MergeArrays.req(
                self,
                lepton_selection=self.channel,
                datasets_to_process=self.datasets_to_process,
                category=cat,
            )
            for cat in ["Anti_cuts", "SB_cuts"]
        }
        return inp

    def output(self):
        return {"plot": self.local_target("LP.png"), "factors": self.local_target("factors.json")}

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        SB = inp["SB_cuts"]
        Anti = inp["Anti_cuts"]
        var_names = self.config_inst.variables.names()
        Anti_dict, SB_dict = {}, {}
        # "data": bh.Histogram(bh.axis.Regular(binning[0], binning[1], binning[2]))
        for dat in self.datasets_to_process:
            SB_arr = SB["SB_cuts_" + dat]["array"].load()
            Anti_arr = Anti["Anti_cuts_" + dat]["array"].load()
            SB_weights = SB["SB_cuts_" + dat]["weights"].load()
            Anti_weights = Anti["Anti_cuts_" + dat]["weights"].load()

            SB_dict[dat] = {"hist": SB_arr[:, var_names.index("LP")], "weights": SB_weights}
            Anti_dict[dat] = {"hist": Anti_arr[:, var_names.index("LP")], "weights": Anti_weights}

        hist_dict = {}
        for dic, name in [(SB_dict, "SB"), (Anti_dict, "Anti")]:
            MC_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]))
            data_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]))
            QCD_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]))
            for key in dic.keys():
                if key in self.config_inst.get_aux("data"):
                    data_hist.fill(dic[key]["hist"], weight=dic[key]["weights"])
                elif key == "QCD":
                    QCD_hist.fill(dic[key]["hist"], weight=dic[key]["weights"])
                else:
                    MC_hist.fill(dic[key]["hist"], weight=dic[key]["weights"])
            hist_dict[name] = {"data": data_hist, "QCD": QCD_hist, "MC": MC_hist}

        fig, ax = plt.subplots(figsize=(12, 10))
        for name, col in [("SB", "red"), ("Anti", "blue")]:
            hep.histplot(hist_dict[name]["data"], histtype="errorbar", label="data " + name, ax=ax, flow="none", color=col)
            hep.histplot(hist_dict[name]["QCD"], histtype="step", label="QCD " + name, ax=ax, linewidth=1, flow="none", color=col)
            hep.histplot(hist_dict[name]["MC"], histtype="step", label="MC " + name, ax=ax, linewidth=1, flow="none", linestyle="dotted", color=col)

        plt.legend(fontsize=16)
        ax.set_xlabel(self.config_inst.get_variable("LP").get_full_x_title(), fontsize=18)
        ax.set_ylabel(self.config_inst.get_variable("LP").y_title, fontsize=18)
        self.output()["plot"].parent.touch()
        plt.savefig(self.output()["plot"].path, bbox_inches="tight")
        ax.set_yscale("log")
        plt.savefig(self.output()["plot"].path.replace(".png", "_log.png"), bbox_inches="tight")
        ax.figure.clf()

        factors = {}
        for name in ["SB", "Anti"]:
            print(name)
            # doing a template fit to get QCD factor, after loop Anti-selected is filled
            contrib = np.column_stack([hist_dict[name]["QCD"].values(), hist_dict[name]["MC"].values()])
            # FIXME doing real template fit like pyRoot or minute
            # https://scikit-hep.org/iminuit/
            output = np.linalg.lstsq(contrib, hist_dict[name]["data"].values(), rcond=None)
            # new plot
            fig, ax = plt.subplots(figsize=(12, 10))
            hep.style.use("CMS")
            hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
            hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)

            hep.histplot(hist_dict[name]["data"], histtype="errorbar", label="data", ax=ax, flow="none", color=col)
            hep.histplot(hist_dict[name]["QCD"] * output[0][0], histtype="step", label="QCD * " + str(output[0][0]), ax=ax, linewidth=1, flow="none", color=col)
            hep.histplot(hist_dict[name]["MC"] * output[0][1], histtype="step", label="MC * " + str(output[0][1]), ax=ax, linewidth=1, flow="none", linestyle="dotted", color=col)
            plt.legend(fontsize=16)
            ax.set_xlabel(self.config_inst.get_variable("LP").get_full_x_title(), fontsize=18)
            ax.set_ylabel(self.config_inst.get_variable("LP").y_title + " Fitted Anti selected contributions", fontsize=18)
            # ax.set_title("Fitted Anti selected contributions", fontsize=18)
            plt.savefig(self.output()["plot"].path.replace(".png", "_{}_fitted.png".format(name)), bbox_inches="tight")
            ax.set_yscale("log")
            plt.savefig(self.output()["plot"].path.replace(".png", "_{}_fitted_log.png".format(name)), bbox_inches="tight")
            ax.figure.clf()

            factors[name] = {"QCD": output[0][0], "EWK": output[0][1]}
        self.output()["factors"].dump(factors)


class TransferQCD(CoffeaTask, DNNTask):
    channel = luigi.ListParameter(default=["LeptonIncl"])

    def requires(self):
        inp = {
            cat: MergeArrays.req(
                self,
                lepton_selection=self.channel[0],
                datasets_to_process=self.datasets_to_process,
                category=cat,
            )
            for cat in ["Anti_cuts", "SB_cuts"]
        }
        inp["model"] = PytorchCrossVal.req(
            self,
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            dropout=self.dropout,
            kfold=self.kfold,
        )
        inp["factors"] = FitQCDContribution.req(self, category="Anti_cuts")
        return inp

    def output(self):
        return self.local_target("out.png")
        # return {"factors": self.local_target("factors.json")}

    def store_parts(self):
        # make plots for each use case
        return super(TransferQCD, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.safe_output
    def run(self):
        # only keep arrays, so we can loop cuts dynamically
        inp = self.input()
        inp.pop("model")
        inp.pop("factors")
        factors = self.input()["factors"]["factors"].load()

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

        SR_scores = {}
        for cat in inp.keys():
            hists = {"MC": {}, "signal": {}, "data": {}}
            fig, ax = plt.subplots(figsize=(12, 10))
            for dat in tqdm(self.datasets_to_process):
                arr = inp[cat][cat + "_" + dat]["array"].load()
                weights = inp[cat][cat + "_" + dat]["weights"].load()
                # don't need to care for ID, since network never saw sidebands DNNId = inp[cat][cat+"_"+dat]["DNNId"].load()

                SR_scores[cat + "_" + dat] = {}
                # SR_scores[cat+"_"+dat]["weights"]=weights
                for i in range(self.kfold):
                    # accessing the input and unpacking the condor submission structure
                    path = self.input()["model"]["collection"].targets[0]["fold_" + str(i)]["model"].path

                    # load complete model
                    reconstructed_model = torch.load(path)

                    # load all the prepared data thingies
                    X_test = torch.tensor(arr)

                    with torch.no_grad():
                        reconstructed_model.eval()
                        y_predictions = reconstructed_model(X_test).numpy()
                    SR_scores[cat + "_" + dat]["scores_" + str(i)] = y_predictions

                scores_average = (SR_scores[cat + "_" + dat]["scores_0"] + SR_scores[cat + "_" + dat]["scores_1"]) / 2
                mask = np.argmax(scores_average, axis=-1) == (n_processes - 1)
                proc = self.config_inst.get_process(dat)
                boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                boost_hist.fill(scores_average[mask][:, -1], weight=weights[mask])

                if proc.aux["isData"]:  # and self.unblinded:
                    hists["data"][dat] = boost_hist
                elif proc.aux["isSignal"]:  # and self.signal:
                    hists["signal"][dat] = {
                        "hist": boost_hist,
                        "label": proc.label,
                        "color": proc.color,
                    }
                elif not proc.aux["isData"] and not proc.aux["isSignal"]:
                    # MC categorisation by exclusion, apply norm factors here, if we want
                    factor = 1.0
                    # if self.norm:
                    #     for factor_key in self.config_inst.get_aux("DNN_process_template")[self.category].keys():
                    #         if dat in self.config_inst.get_aux("DNN_process_template")[self.category][factor_key]:
                    #             factor= norm_factors[proc_factor_dict[factor_key]]
                    hists["MC"][dat] = {
                        "hist": boost_hist * factor,
                        "label": proc.label,
                        "color": proc.color,
                    }

            # one histplot together, ordered by integral
            # can't stack seperate histplot calls, so we have do it like that
            hist_list, label_list, color_list = [], [], []
            for key in hists["MC"].keys():
                hist_list.append(hists["MC"][key]["hist"])
                label_list.append(hists["MC"][key]["label"])
                color_list.append(hists["MC"][key]["color"])
            # Always plot MC
            order = np.argsort(np.sum(hist_list, axis=-1))
            hep.histplot(np.array(hist_list)[order], bins=hist_list[0].axes[0].edges, histtype="fill", stack=True, label=np.array(label_list)[order], color=np.array(color_list)[order], ax=ax)
            hep.histplot(sum(list(hists["data"].values())), histtype="errorbar", label="data", color="black", ax=ax)
            # if self.unblinded:

            #     data_hist = sum(list(hists["data"].values()))
            #     MC_hist = sum(hist_list)
            #     ratio = data_hist / MC_hist
            #     stat_unc = np.sqrt(ratio * (ratio / MC_hist + ratio / data_hist))
            #     rax.axhline(1.0, color="black", linestyle="--")
            #     rax.fill_between(ratio.axes[0].centers, 1 - 0.023, 1 + 0.023, alpha=0.3, facecolor="black")
            #     hep.histplot(ratio, color="black", histtype="errorbar", stack=False, yerr=stat_unc, ax=rax)
            #     rax.set_xlabel("DNN Scores in Signal node", fontsize=24)
            #     rax.set_xlim(0, 1)
            #     rax.set_ylabel("Data/MC", fontsize=24)
            #     rax.set_ylim(0.5, 1.5)
            #     rax.tick_params(axis="both", which="major", labelsize=18)
            # else:
            ax.set_xlabel("DNN Scores in Signal node", fontsize=24)
            ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0, prop={"size": 18})
            # if self.signal:
            # for key in hists["signal"].keys():
            #     hep.histplot(hists["signal"][key]["hist"], histtype="step", label=hists["signal"][key]["label"], color=hists["signal"][key]["color"], ax=ax)
            self.output().parent.touch()
            plt.savefig(self.output().path.replace(".png", "_{}.png".format(cat)), bbox_inches="tight")


class IterativeQCDFitting(CoffeaTask, DNNTask):
    channel = luigi.ListParameter(default=["LeptonIncl"])

    def requires(self):
        inp = {
            cat: MergeArrays.req(
                self,
                lepton_selection=self.channel[0],
                datasets_to_process=self.datasets_to_process,
                category=cat,
            )
            for cat in ["Anti_cuts", "SB_cuts"]
        }
        inp["model"] = PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout, kfold=self.kfold, category="SR0b")
        return inp

    def output(self):
        return {"factors": self.local_target("alpha_beta_delta.json"), "F_Sel_Anti": self.local_target("F_Sel_Anti.json")}

    def store_parts(self):
        # make plots for each use case
        return super(IterativeQCDFitting, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        inp.pop("model")

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

        # initial delta
        delta = 1.0

        # we need to get these scores once
        SR_scores = {}
        for cat in ["SB_cuts"]:  # inp.keys():
            hists = {"MC": {}, "signal": {}, "data": {}}
            for dat in tqdm(self.datasets_to_process):
                proc = self.config_inst.get_process(dat)
                arr = inp[cat][cat + "_" + dat]["array"].load()
                weights = inp[cat][cat + "_" + dat]["weights"].load()
                # don't need to care for ID, since network never saw

                SR_scores[cat + "_" + dat] = {}
                # SR_scores[cat+"_"+dat]["weights"]=weights
                for i in range(self.kfold):
                    # accessing the input and unpacking the condor submission structure
                    path = self.input()["model"]["collection"].targets[0]["fold_" + str(i)]["model"].path

                    # load complete model
                    reconstructed_model = torch.load(path)

                    # load all the prepared data thingies
                    X_test = torch.tensor(arr)

                    with torch.no_grad():
                        reconstructed_model.eval()
                        y_predictions = reconstructed_model(X_test).numpy()
                    SR_scores[cat + "_" + dat]["scores_" + str(i)] = y_predictions
                SR_scores[cat + "_" + dat]["weights"] = weights

        sort_for_factors = {}
        for key in self.config_inst.get_aux("DNN_process_template")["SB_cuts"].keys():
            subprocs, weights = [], []
            for subproc in self.config_inst.get_aux("DNN_process_template")["SB_cuts"][key]:
                scores_average = (SR_scores["SB_cuts" + "_" + subproc]["scores_0"] + SR_scores["SB_cuts" + "_" + subproc]["scores_1"]) / 2
                subprocs.append(scores_average)
                weights.append(SR_scores["SB_cuts" + "_" + subproc]["weights"])
            sort_for_factors[key] = np.concatenate(subprocs)
            sort_for_factors[key + "_weights"] = np.concatenate(weights)

        for i in range(10):
            # assigning nodes, redoing with new delta
            tt_node_0 = np.sum(sort_for_factors["ttjets_weights"][np.argmax(sort_for_factors["ttjets"], axis=-1) == 0])
            Wjets_node_0 = np.sum(sort_for_factors["Wjets_weights"][np.argmax(sort_for_factors["Wjets"], axis=-1) == 0])
            data_node_0 = np.sum(sort_for_factors["data_weights"][np.argmax(sort_for_factors["data"], axis=-1) == 0]) - delta * np.sum(sort_for_factors["QCD_weights"][np.argmax(sort_for_factors["QCD"], axis=-1) == 0])

            tt_node_1 = np.sum(sort_for_factors["ttjets_weights"][np.argmax(sort_for_factors["ttjets"], axis=-1) == 1])
            Wjets_node_1 = np.sum(sort_for_factors["Wjets_weights"][np.argmax(sort_for_factors["Wjets"], axis=-1) == 1])
            data_node_1 = np.sum(sort_for_factors["data_weights"][np.argmax(sort_for_factors["data"], axis=-1) == 1]) - delta * np.sum(sort_for_factors["QCD_weights"][np.argmax(sort_for_factors["QCD"], axis=-1) == 1])

            # constructing and solving linear equation system
            left_side = np.array([[tt_node_0, Wjets_node_0], [tt_node_1, Wjets_node_1]])
            right_side = np.array([data_node_0, data_node_1])
            factors = np.linalg.solve(left_side, right_side)

            # alpha * tt + beta * Wjets
            factor_dict = {"ttjets": factors[0], "Wjets": factors[1]}
            print(factor_dict)

            SB = inp["SB_cuts"]
            Anti = inp["Anti_cuts"]
            var_names = self.config_inst.variables.names()

            Anti_dict, SB_dict = {}, {}
            # "data": bh.Histogram(bh.axis.Regular(binning[0], binning[1], binning[2]))
            for dat in self.datasets_to_process:
                SB_arr = SB["SB_cuts_" + dat]["array"].load()
                Anti_arr = Anti["Anti_cuts_" + dat]["array"].load()
                SB_weights = SB["SB_cuts_" + dat]["weights"].load()
                Anti_weights = Anti["Anti_cuts_" + dat]["weights"].load()

                SB_dict[dat] = {"hist": SB_arr[:, var_names.index("LP")], "weights": SB_weights}
                Anti_dict[dat] = {"hist": Anti_arr[:, var_names.index("LP")], "weights": Anti_weights}

            hist_dict = {}
            for dic, name in [(SB_dict, "SB"), (Anti_dict, "Anti")]:
                MC_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]))
                data_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]))
                QCD_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]))
                for key in dic.keys():
                    if key in self.config_inst.get_aux("data"):
                        data_hist.fill(dic[key]["hist"], weight=dic[key]["weights"])
                    elif key == "QCD":
                        QCD_hist.fill(dic[key]["hist"], weight=dic[key]["weights"])
                    else:
                        factor = 1.0
                        for proc in self.config_inst.get_aux("DNN_process_template")["SB_cuts"].keys():
                            if key in self.config_inst.get_aux("DNN_process_template")["SB_cuts"][proc]:
                                factor = factor_dict[proc]
                        MC_hist.fill(dic[key]["hist"], weight=dic[key]["weights"] * factor)
                hist_dict[name] = {"data": data_hist, "QCD": QCD_hist, "MC": MC_hist}

            factors = {}
            for name in ["SB"]:  # , "Anti"]:
                # doing a template fit to get QCD factor, after loop Anti-selected is filled
                contrib = np.column_stack([hist_dict[name]["QCD"].values(), hist_dict[name]["MC"].values()])
                output = np.linalg.lstsq(contrib, hist_dict[name]["data"].values(), rcond=None)
                print(name, " QCD ", " MC ", output[0])

            delta = output[0][0]

            print("delta:", delta, " alpha beta: ", factor_dict, "\n")
        self.output()["factors"].parent.path
        self.output()["factors"].dump({"delta": delta, "alpha": factor_dict["ttjets"], "beta": factor_dict["Wjets"]})
        F_Sel_Anti = (hist_dict["SB"]["QCD"] * delta) / hist_dict["Anti"]["QCD"]
        print("F_Sel_Anti:", F_Sel_Anti, "\n")
        self.output()["F_Sel_Anti"].dump({"F_Sel_Anti": sum(F_Sel_Anti)})


class EstimateQCDinSR(CoffeaTask, DNNTask):
    def requires(self):
        channels = {"SR0b": ["Muon", "Electron"], "SR_Anti": ["LeptonIncl"]}
        inp = {
            cat: MergeArrays.req(
                self,
                channel=channels[cat],
                datasets_to_process=self.datasets_to_process,
                category=cat,
            )
            for cat in channels.keys()
        }
        inp["model"] = PytorchCrossVal.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout, kfold=self.kfold, category="SR0b")
        inp["F_Sel_Anti"] = IterativeQCDFitting.req(
            self,
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            dropout=self.dropout,
            kfold=self.kfold,
        )
        return inp

    def output(self):
        return self.local_target("SR_prediction.json")

    def store_parts(self):
        return super(EstimateQCDinSR, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

        # we need to get these scores once
        Anti_scores = {}
        for cat in ["SR_Anti"]:  # inp.keys():
            hists = {"MC": {}, "signal": {}, "data": {}}
            for dat in tqdm(self.datasets_to_process):
                proc = self.config_inst.get_process(dat)
                arr = inp[cat][cat + "_" + dat]["array"].load()
                weights = inp[cat][cat + "_" + dat]["weights"].load()
                # don't need to care for ID, since network never saw

                Anti_scores[cat + "_" + dat] = {}
                for i in range(self.kfold):
                    # accessing the input and unpacking the condor submission structure
                    path = self.input()["model"]["collection"].targets[0]["fold_" + str(i)]["model"].path

                    # load complete model
                    reconstructed_model = torch.load(path)

                    # load all the prepared data thingies
                    X_test = torch.tensor(arr)

                    with torch.no_grad():
                        reconstructed_model.eval()
                        y_predictions = reconstructed_model(X_test).numpy()
                    Anti_scores[cat + "_" + dat]["scores_" + str(i)] = y_predictions
                Anti_scores[cat + "_" + dat]["weights"] = weights

        sort_for_factors = {}
        for key in self.config_inst.get_aux("DNN_process_template")["SR_Anti"].keys():
            subprocs, weights = [], []
            for subproc in self.config_inst.get_aux("DNN_process_template")["SR_Anti"][key]:
                scores_average = (Anti_scores["SR_Anti" + "_" + subproc]["scores_0"] + Anti_scores["SR_Anti" + "_" + subproc]["scores_1"]) / 2
                subprocs.append(scores_average)
                weights.append(Anti_scores["SR_Anti" + "_" + subproc]["weights"])
            sort_for_factors[key] = np.concatenate(subprocs)
            sort_for_factors[key + "_weights"] = np.concatenate(weights)

        # plotting and constructing SR bins for Anti-selection
        fig, ax = plt.subplots(figsize=(12, 10))
        hep.style.use("CMS")
        hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
        hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)
        hist_list, label_list = [], []
        for contrib in sort_for_factors.keys():
            if "weight" in contrib:
                continue
            # only argmax
            mask = np.argmax(sort_for_factors[contrib], axis=-1) == (n_processes - 1)
            if contrib == "data":
                data_boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                data_boost_hist.fill(sort_for_factors[contrib][mask][:, -1], weight=sort_for_factors[contrib + "_weights"][mask])
                hep.histplot(boost_hist, histtype="errorbar", label=contrib, color="black", ax=ax)
            else:
                boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                boost_hist.fill(sort_for_factors[contrib][mask][:, -1], weight=sort_for_factors[contrib + "_weights"][mask])
                hist_list.append(boost_hist)
                label_list.append(contrib)

        hep.histplot(hist_list, histtype="fill", stack=True, label=label_list, ax=ax)  # bins=hist_list[0].axes[0].edges,
        ax.set_xlabel("DNN Scores in Signal node", fontsize=24)
        ax.set_ylabel("Counts", fontsize=24)
        ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0, prop={"size": 18})
        self.output().parent.touch()
        plt.savefig(self.output().parent.path + "/Anti_SR.png", bbox_inches="tight")
        ax.set_yscale("log")
        plt.savefig(self.output().parent.path + "/Anti_SR_log.png", bbox_inches="tight")

        # predicting QCD in SR by removing QCD and substracting Data-EWK
        del hist_list[label_list.index("QCD")]
        EWK = sum(hist_list)
        QCD_new = data_boost_hist - EWK
        F_Sel_Anti = self.input()["F_Sel_Anti"]["F_Sel_Anti"].load()["F_Sel_Anti"]
        # FIXME abs now is for numeric stability, maybe try QCD MC for an estimate?
        SR_prediction = {"SR_prediction": (abs(QCD_new.values()) * F_Sel_Anti).tolist()}
        self.output().dump(SR_prediction)

        """
        from IPython import embed; embed()
        SR0b_scores = {}
        for dat in tqdm(self.datasets_to_process):
            # accessing the input and unpacking the condor submission structure
            cat = "SR0b"
            sumOfHists = []
            arr = inp[cat][cat+"_"+dat]["array"].load()
            weights = inp[cat][cat+"_"+dat]["weights"].load()
            DNNId = inp[cat][cat+"_"+dat]["DNNId"].load()
            SR_scores, SR_weights = [], []
            for i in range(self.kfold):
                # switch around in 2 point k fold
                j = abs(i - 1)
                path = self.input()["model"]["collection"].targets[0]["fold_" + str(i)]["model"].path

                # load complete model
                reconstructed_model = torch.load(path)

                # load all the prepared data thingies
                X_test = torch.tensor(arr[DNNId == j])
                weight_test = weights[DNNId == j]

                with torch.no_grad():
                    reconstructed_model.eval()
                    y_predictions = reconstructed_model(X_test).numpy()
                mask = np.argmax(y_predictions, axis=-1) == (n_processes - 1)
                SR_scores.append(y_predictions[mask][:,-1])
                SR_weights.append(weight_test[mask])
            SR0b_scores[dat] = np.concatenate(SR_scores)
            SR0b_scores[dat+"_weight"]=np.concatenate(SR_weights)
        """
