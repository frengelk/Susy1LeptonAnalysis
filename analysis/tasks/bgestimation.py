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

# from iminuit import Minuit
# from iminuit.cost import LeastSquares, Template
from scipy.optimize import minimize

# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask  # , AntiProcessor
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
            plt.legend(fontsize=16, title=name)
            ax.set_xlabel(self.config_inst.get_variable("LP").get_full_x_title(), fontsize=18)
            ax.set_ylabel(self.config_inst.get_variable("LP").y_title + " Fitted contributions", fontsize=18)
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
            category="All_Lep",
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

        # loading all models
        models = self.input()["model"]["collection"].targets[0]
        print("loading models")
        models_loaded = {fold: torch.load(models["fold_" + str(fold)]["model"].path) for fold in range(self.kfold)}

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

        DNN_scores = {}
        for cat in inp.keys():
            hists = {"MC": {}, "signal": {}, "data": {}}
            fig, ax = plt.subplots(figsize=(12, 10))
            for dat in tqdm(self.datasets_to_process):
                arr = inp[cat][cat + "_" + dat]["array"].load()
                weights = inp[cat][cat + "_" + dat]["weights"].load()
                # FIXME don't need to care for ID, since network never saw sidebands
                # DNNId = inp[cat][cat+"_"+dat]["DNNId"].load()

                DNN_scores[cat + "_" + dat] = {}
                # DNN_scores[cat+"_"+dat]["weights"]=weights
                for fold in range(self.kfold):
                    # load complete model
                    reconstructed_model = models_loaded[fold]
                    # load all the prepared data thingies
                    X_test = torch.tensor(arr)

                    with torch.no_grad():
                        reconstructed_model.eval()
                        y_predictions = reconstructed_model(X_test).softmax(dim=1).numpy()
                    DNN_scores[cat + "_" + dat]["scores_" + str(fold)] = y_predictions

                # hardcoded for two-fold, FIXME
                scores_average = (DNN_scores[cat + "_" + dat]["scores_0"] + DNN_scores[cat + "_" + dat]["scores_1"]) / 2
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
            ax.set_xlabel("{}: DNN Scores in Signal node".format(cat), fontsize=24)
            ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0, prop={"size": 18})
            # if self.signal:
            # for key in hists["signal"].keys():
            #     hep.histplot(hists["signal"][key]["hist"], histtype="step", label=hists["signal"][key]["label"], color=hists["signal"][key]["color"], ax=ax)
            self.output().parent.touch()
            plt.savefig(self.output().path.replace(".png", "_{}.png".format(cat)), bbox_inches="tight")
            ax.set_yscale("log")
            plt.savefig(self.output().path.replace(".png", "_{}_log.png".format(cat)), bbox_inches="tight")


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
        inp["model"] = PytorchCrossVal.req(
            self,
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            dropout=self.dropout,
            kfold=self.kfold,
            category="All_Lep",
        )  # "SR0b"
        return inp

    def output(self):
        return {"factors": self.local_target("alpha_beta_delta.json"), "F_Sel_Anti": self.local_target("F_Sel_Anti.json")}

    def store_parts(self):
        # make plots for each use case
        return super(IterativeQCDFitting, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + (self.gamma,)

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    # Define the fit function
    def fit_function(self, params, template1, template2):
        x, y = params
        return x * template1 + y * template2

    def chi_square(self, params, data, data_errors, template1, template2, template1_errors, template2_errors):
        model = self.fit_function(params, template1, template2)
        model_errors = np.sqrt((params[0] * template1_errors) ** 2 + (params[1] * template2_errors) ** 2)
        chi2 = np.sum(((data - model) / np.sqrt(data_errors**2 + model_errors**2)) ** 2)
        return chi2

    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        inp.pop("model")

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

        # initial delta
        delta = 1.0

        # loading all models
        models = self.input()["model"]["collection"].targets[0]
        print("loading models")
        models_loaded = {fold: torch.load(models["fold_" + str(fold)]["model"].path) for fold in range(self.kfold)}

        # we need to get these scores once
        DNN_scores = {}
        for cat in ["SB_cuts", "Anti_cuts"]:  # inp.keys():
            hists = {"MC": {}, "signal": {}, "data": {}}
            for dat in tqdm(self.datasets_to_process):
                proc = self.config_inst.get_process(dat)
                arr = inp[cat][cat + "_" + dat]["array"].load()
                weights = inp[cat][cat + "_" + dat]["weights"].load()
                DNNId = inp[cat][cat + "_" + dat]["DNNId"].load()
                # don't need to care for ID, since network never saw
                DNN_scores[cat + "_" + dat] = {}
                # DNN_scores[cat+"_"+dat]["weights"]=weights
                for fold in range(self.kfold):
                    # load complete model
                    reconstructed_model = models_loaded[fold]
                    # fold Id to predict on unseen events from the other fold
                    foldId = 1 - 2 * fold

                    # load all the prepared data thingies
                    X_test = torch.tensor(arr[DNNId == foldId])
                    weights_test = weights[DNNId == foldId]

                    with torch.no_grad():
                        reconstructed_model.eval()
                        y_predictions = reconstructed_model(X_test).softmax(dim=1).numpy()
                    DNN_scores[cat + "_" + dat]["scores_" + str(fold)] = y_predictions
                    DNN_scores[cat + "_" + dat]["weights_" + str(fold)] = weights_test

        sort_for_factors = {}
        for cat in ["SB_cuts", "Anti_cuts"]:
            sort_for_factors[cat] = {}
            for key in self.config_inst.get_aux("DNN_process_template")[cat].keys():
                subprocs, weights = [], []
                for subproc in self.config_inst.get_aux("DNN_process_template")[cat][key]:
                    scores_combined = np.concatenate((DNN_scores[cat + "_" + subproc]["scores_0"], DNN_scores[cat + "_" + subproc]["scores_1"]))
                    weights_combined = np.concatenate((DNN_scores[cat + "_" + subproc]["weights_0"], DNN_scores[cat + "_" + subproc]["weights_1"]))
                    subprocs.append(scores_combined)
                    weights.append(weights_combined)
                sort_for_factors[cat][key] = np.concatenate(subprocs)
                sort_for_factors[cat][key + "_weights"] = np.concatenate(weights)

        LP_binning = (self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2])

        for i in range(10):
            # assigning nodes, redoing with new delta
            tt_node_0 = np.sum(sort_for_factors["SB_cuts"]["ttjets_weights"][np.argmax(sort_for_factors["SB_cuts"]["ttjets"], axis=-1) == 0])
            Wjets_node_0 = np.sum(sort_for_factors["SB_cuts"]["Wjets_weights"][np.argmax(sort_for_factors["SB_cuts"]["Wjets"], axis=-1) == 0])
            data_node_0 = np.sum(sort_for_factors["SB_cuts"]["data_weights"][np.argmax(sort_for_factors["SB_cuts"]["data"], axis=-1) == 0]) - delta * np.sum(sort_for_factors["SB_cuts"]["QCD_weights"][np.argmax(sort_for_factors["SB_cuts"]["QCD"], axis=-1) == 0])

            tt_node_1 = np.sum(sort_for_factors["SB_cuts"]["ttjets_weights"][np.argmax(sort_for_factors["SB_cuts"]["ttjets"], axis=-1) == 1])
            Wjets_node_1 = np.sum(sort_for_factors["SB_cuts"]["Wjets_weights"][np.argmax(sort_for_factors["SB_cuts"]["Wjets"], axis=-1) == 1])
            data_node_1 = np.sum(sort_for_factors["SB_cuts"]["data_weights"][np.argmax(sort_for_factors["SB_cuts"]["data"], axis=-1) == 1]) - delta * np.sum(sort_for_factors["SB_cuts"]["QCD_weights"][np.argmax(sort_for_factors["SB_cuts"]["QCD"], axis=-1) == 1])

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
                MC_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]), storage=bh.storage.Weight())
                data_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]), storage=bh.storage.Weight())
                QCD_hist = bh.Histogram(bh.axis.Regular(self.config_inst.get_variable("LP").binning[0], self.config_inst.get_variable("LP").binning[1], self.config_inst.get_variable("LP").binning[2]), storage=bh.storage.Weight())
                for key in dic.keys():
                    if key in self.config_inst.get_aux("data"):
                        data_hist.fill(dic[key]["hist"])  # , weight=dic[key]["weights"])
                    elif key == "QCD":
                        QCD_hist.fill(dic[key]["hist"], weight=dic[key]["weights"])
                    else:
                        factor = 1.0
                        for proc in self.config_inst.get_aux("DNN_process_template")["SB_cuts"].keys():
                            if key in self.config_inst.get_aux("DNN_process_template")["SB_cuts"][proc]:
                                factor = factor_dict[proc]
                        MC_hist.fill(dic[key]["hist"], weight=dic[key]["weights"] * factor)
                hist_dict[name] = {"data": data_hist, "QCD": QCD_hist, "MC": MC_hist}

            print("delta from previous fit", delta, " alpha beta: ", factor_dict, "\n")

            # print(QCD_hist, MC_hist, data_hist)

            # # Define the fit function, to find a and b for (MC, QCD) as best fit to data
            # def fit_function(sim_data, a, delta ):
            #     return sim_data[0] * a + sim_data[1] * delta

            # # Create the minimizer using iminuit
            # # sum_error = np.sqrt((MC_hist + QCD_hist).variances())
            # least_squares = LeastSquares(data_hist.values(), (MC_hist.values(), QCD_hist.values()), (np.sqrt(MC_hist.variances()), np.sqrt(QCD_hist.variances())), fit_function)
            # minimizer = Minuit(least_squares, a=output[0][1], delta=output[0][0])

            # # Perform the minimization
            # from IPython import embed; embed()
            # minimizer.migrad()

            # # overwriting delta as better estimate for fit
            # # Retrieve the fit results
            # x = minimizer.values['a']
            # delta = minimizer.values['delta']
            # x_error = minimizer.errors['a']
            # delta_error = minimizer.errors['delta']
            # print(f"Fit results: x * MC = {x} ± {x_error}, y * QCD = {delta} ± {delta_error}")

            # combined = QCD_hist.values()+MC_hist.values() # np.array([list(QCD_hist.values()), list(MC_hist.values())])
            # bins_LP = np.linspace(LP_binning[1], LP_binning[2], LP_binning[0] + 1)
            # templ = Template(data_hist.values(), bins_LP, combined, method="da")
            # m3 = Minuit(templ, *output[0])
            # m3.limits = (0, None)
            # m3.migrad()
            # m3.hesse()

            # print(m3)
            # Example data
            data = hist_dict["SB"]["data"].values()  # data_hist.values()
            data_errors = np.sqrt(hist_dict["SB"]["data"].variances())
            combined = np.array([list(hist_dict["SB"]["QCD"].values()), list(hist_dict["SB"]["MC"].values())])
            combined_errors = np.array([list(np.sqrt(hist_dict["SB"]["QCD"].variances())), list(np.sqrt(hist_dict["SB"]["MC"].variances()))])

            # Normalize the templates
            combined_norm = combined / combined.sum(axis=1, keepdims=True)
            data_norm = data / data.sum()

            # Initial guess for the parameters (more realistic based on expected range)
            initial_guess = [1.0, 1.0]

            # Perform the minimization
            # result = minimize(self.chi_square, initial_guess, args=(data_norm, combined_norm[0], combined_norm[1]), method='BFGS')
            result = minimize(self.chi_square, initial_guess, args=(data, data_errors, combined[0], combined[1], combined_errors[0], combined_errors[1]), method="BFGS")

            # Retrieve the fit results
            x_fit, y_fit = result.x

            try:
                # Approximate errors from the inverse Hessian if available
                x_error, y_error = np.sqrt(np.diag(result.hess_inv))
            except AttributeError:
                x_error, y_error = None, None

            print(f"Fit results: x = {x_fit} ± {x_error}, y = {y_fit} ± {y_error}")
            delta = x_fit

        # normalizing in Anti_cuts as well
        tt_node_0 = np.sum(sort_for_factors["Anti_cuts"]["ttjets_weights"][np.argmax(sort_for_factors["Anti_cuts"]["ttjets"], axis=-1) == 0])
        tt_node_1 = np.sum(sort_for_factors["Anti_cuts"]["ttjets_weights"][np.argmax(sort_for_factors["Anti_cuts"]["ttjets"], axis=-1) == 1])
        tt_node_2 = np.sum(sort_for_factors["Anti_cuts"]["ttjets_weights"][np.argmax(sort_for_factors["Anti_cuts"]["ttjets"], axis=-1) == 2])

        Wjets_node_0 = np.sum(sort_for_factors["Anti_cuts"]["Wjets_weights"][np.argmax(sort_for_factors["Anti_cuts"]["Wjets"], axis=-1) == 0])
        Wjets_node_1 = np.sum(sort_for_factors["Anti_cuts"]["Wjets_weights"][np.argmax(sort_for_factors["Anti_cuts"]["Wjets"], axis=-1) == 1])
        Wjets_node_2 = np.sum(sort_for_factors["Anti_cuts"]["Wjets_weights"][np.argmax(sort_for_factors["Anti_cuts"]["Wjets"], axis=-1) == 2])

        QCD_node_0 = np.sum(sort_for_factors["Anti_cuts"]["QCD_weights"][np.argmax(sort_for_factors["Anti_cuts"]["QCD"], axis=-1) == 0])
        QCD_node_1 = np.sum(sort_for_factors["Anti_cuts"]["QCD_weights"][np.argmax(sort_for_factors["Anti_cuts"]["QCD"], axis=-1) == 1])
        QCD_node_2 = np.sum(sort_for_factors["Anti_cuts"]["QCD_weights"][np.argmax(sort_for_factors["Anti_cuts"]["QCD"], axis=-1) == 2])

        data_node_0 = np.sum(sort_for_factors["Anti_cuts"]["data_weights"][np.argmax(sort_for_factors["Anti_cuts"]["data"], axis=-1) == 0])
        data_node_1 = np.sum(sort_for_factors["Anti_cuts"]["data_weights"][np.argmax(sort_for_factors["Anti_cuts"]["data"], axis=-1) == 1])
        data_node_2 = np.sum(sort_for_factors["Anti_cuts"]["data_weights"][np.argmax(sort_for_factors["Anti_cuts"]["data"], axis=-1) == 2])

        # solving linear equation system
        left_side = np.array([[tt_node_0, Wjets_node_0, QCD_node_0], [tt_node_1, Wjets_node_1, QCD_node_1], [tt_node_2, Wjets_node_2, QCD_node_2]])
        right_side = np.array([data_node_0, data_node_1, data_node_2])
        output = np.linalg.solve(left_side, right_side)
        print("normalization in Anti cuts", output)

        # Plot the results without template fit
        plt.figure(figsize=(10, 6))
        hep.style.use("CMS")
        hep.cms.text("Private work (CMS simulation)", loc=0)
        hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$")
        bins_LP = np.linspace(LP_binning[1], LP_binning[2], LP_binning[0] + 1)
        bin_centers = 0.5 * (bins_LP[:-1] + bins_LP[1:])

        plt.errorbar(bin_centers, data, yerr=np.sqrt(data), fmt="o", label="Data", color="black")
        plt.step(bin_centers, combined[0], where="mid", label="QCD MC", linestyle="--", color="blue")
        plt.step(bin_centers, combined[1], where="mid", label="Other MC", linestyle="--", color="yellow")
        plt.step(bin_centers, combined[0] + combined[1], where="mid", label="QCD + Other MC", linestyle=":", color="red")
        # plt.step(bin_centers, self.fit_function([output[0][0], output[0][1]], combined[0], combined[1]), where='mid', label='Least Sq Solution',  linestyle='-.')

        plt.xlabel(self.config_inst.get_variable("LP").get_full_x_title(), fontsize=18)
        plt.ylabel(self.config_inst.get_variable("LP").y_title, fontsize=18)
        plt.tick_params(axis="both", which="major", labelsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        self.output()["factors"].parent.touch()
        plt.savefig(self.output()["factors"].path.replace(".json", "_before.png"))
        plt.savefig(self.output()["factors"].path.replace(".json", "_before.pdf"))
        # Plot the results after iterating
        plt.figure(figsize=(10, 6))
        hep.style.use("CMS")
        hep.cms.text("Private work (CMS simulation)", loc=0)
        hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$")
        bins_LP = np.linspace(LP_binning[1], LP_binning[2], LP_binning[0] + 1)
        bin_centers = 0.5 * (bins_LP[:-1] + bins_LP[1:])

        plt.errorbar(bin_centers, data, yerr=np.sqrt(data), fmt="o", label="Data", color="black")
        plt.step(bin_centers, self.fit_function([x_fit, y_fit], combined[0], combined[1]), where="mid", label="Fit", linestyle=":", color="red")
        plt.step(bin_centers, x_fit * combined[0], where="mid", label="QCD MC template", linestyle="--", color="blue")
        plt.step(bin_centers, y_fit * combined[1], where="mid", label="Other MC template", linestyle="--", color="yellow")
        # plt.step(bin_centers, self.fit_function([output[0][0], output[0][1]], combined[0], combined[1]), where='mid', label='Least Sq Solution',  linestyle='-.')

        plt.xlabel(self.config_inst.get_variable("LP").get_full_x_title(), fontsize=18)
        plt.ylabel(self.config_inst.get_variable("LP").y_title, fontsize=18)
        plt.tick_params(axis="both", which="major", labelsize=16)
        plt.legend(fontsize=16)
        # plt.title("Template Fit QCD/MC in Side band")
        plt.grid()
        self.output()["factors"].parent.touch()
        plt.savefig(self.output()["factors"].path.replace("json", "png"))
        plt.savefig(self.output()["factors"].path.replace("json", "pdf"))
        # do the fraction directly on the sum so we are not binning dependent
        delta_anti = output[2]
        F_Sel_Anti = sum(hist_dict["SB"]["QCD"].values() * delta) / sum(hist_dict["Anti"]["QCD"].values() * delta_anti)

        print("F_Sel_Anti:", F_Sel_Anti, "\n")
        rel_errors = np.sqrt(hist_dict["SB"]["QCD"].variances()) / hist_dict["SB"]["QCD"].values()
        rel_errors_MC = np.sqrt(hist_dict["SB"]["MC"].variances()) / hist_dict["SB"]["MC"].values()
        print("QCD mean:", np.mean(np.nan_to_num(rel_errors)), "MC mean:", np.mean(np.nan_to_num(rel_errors_MC)))
        # print("F_Sel_Anti:", sum(F_Sel_Anti), "\n")
        self.output()["factors"].dump({"delta": delta, "alpha": factor_dict["ttjets"], "beta": factor_dict["Wjets"], "error_delta": x_error})
        self.output()["F_Sel_Anti"].dump({"F_Sel_Anti": F_Sel_Anti})


class EstimateQCDinSR(CoffeaTask, DNNTask):
    def requires(self):
        # channels = {"SR0b": ["Muon", "Electron"], "SR_Anti": ["LeptonIncl"]}
        # channels = ["LeptonIncl"]
        inp = {
            cat: MergeArrays.req(
                self,
                channel=["LeptonIncl"],  # channels
                category="SR_Anti",
                datasets_to_process=self.datasets_to_process,
                lepton_selection="LeptonIncl",
            )
            for cat in ["SR_Anti"]  # channels.keys()
        }
        inp["model"] = PytorchCrossVal.req(
            self,
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            dropout=self.dropout,
            kfold=self.kfold,
            category="All_Lep",
        )  # , category="SR0b"
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
        return super(EstimateQCDinSR, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + (self.gamma,)

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

        # loading all models
        models = self.input()["model"]["collection"].targets[0]
        print("loading models")
        models_loaded = {fold: torch.load(models["fold_" + str(fold)]["model"].path) for fold in range(self.kfold)}
        # factors= self.input()["F_Sel_Anti"]["factors"].load()

        # we need to get these scores once
        Anti_scores = {}
        for cat in ["SR_Anti"]:  # inp.keys():
            hists = {"MC": {}, "signal": {}, "data": {}}
            for dat in tqdm(self.datasets_to_process):
                proc = self.config_inst.get_process(dat)
                arr = inp[cat][cat + "_" + dat]["array"].load()
                weights = inp[cat][cat + "_" + dat]["weights"].load()
                DNNId = inp[cat][cat + "_" + dat]["DNNId"].load()
                # don't need to care for ID, since network never saw

                Anti_scores[cat + "_" + dat] = {}
                for fold in range(self.kfold):
                    # load complete model
                    reconstructed_model = models_loaded[fold]
                    # fold Id to predict on unseen events from the other fold
                    foldId = 1 - 2 * fold

                    # load all the prepared data thingies
                    X_test = torch.tensor(arr[DNNId == foldId])
                    weights_test = weights[DNNId == foldId]

                    with torch.no_grad():
                        reconstructed_model.eval()
                        y_predictions = reconstructed_model(X_test).softmax(dim=1).numpy()
                    Anti_scores[cat + "_" + dat]["scores_" + str(fold)] = y_predictions
                    Anti_scores[cat + "_" + dat]["weights_" + str(fold)] = weights_test

        sort_for_factors = {}
        for key in self.config_inst.get_aux("DNN_process_template")["SR_Anti"].keys():
            subprocs, weights = [], []
            for subproc in self.config_inst.get_aux("DNN_process_template")["SR_Anti"][key]:
                scores_combined = np.concatenate((Anti_scores["SR_Anti" + "_" + subproc]["scores_0"], Anti_scores["SR_Anti" + "_" + subproc]["scores_1"]))
                weights_combined = np.concatenate((Anti_scores["SR_Anti" + "_" + subproc]["weights_0"], Anti_scores["SR_Anti" + "_" + subproc]["weights_1"]))
                subprocs.append(scores_combined)
                weights.append(weights_combined)
            sort_for_factors[key] = np.concatenate(subprocs)
            sort_for_factors[key + "_weights"] = np.concatenate(weights)

        # calculating alpha beta here as well
        tt_node_0 = np.sum(sort_for_factors["ttjets_weights"][np.argmax(sort_for_factors["ttjets"], axis=-1) == 0])
        Wjets_node_0 = np.sum(sort_for_factors["Wjets_weights"][np.argmax(sort_for_factors["Wjets"], axis=-1) == 0])
        data_node_0 = np.sum(sort_for_factors["data_weights"][np.argmax(sort_for_factors["data"], axis=-1) == 0])
        QCD_node_0 = np.sum(sort_for_factors["QCD_weights"][np.argmax(sort_for_factors["QCD"], axis=-1) == 0])

        tt_node_1 = np.sum(sort_for_factors["ttjets_weights"][np.argmax(sort_for_factors["ttjets"], axis=-1) == 1])
        Wjets_node_1 = np.sum(sort_for_factors["Wjets_weights"][np.argmax(sort_for_factors["Wjets"], axis=-1) == 1])
        data_node_1 = np.sum(sort_for_factors["data_weights"][np.argmax(sort_for_factors["data"], axis=-1) == 1])
        QCD_node_1 = np.sum(sort_for_factors["QCD_weights"][np.argmax(sort_for_factors["QCD"], axis=-1) == 1])

        tt_node_2 = np.sum(sort_for_factors["ttjets_weights"][np.argmax(sort_for_factors["ttjets"], axis=-1) == 2])
        Wjets_node_2 = np.sum(sort_for_factors["Wjets_weights"][np.argmax(sort_for_factors["Wjets"], axis=-1) == 2])
        data_node_2 = np.sum(sort_for_factors["data_weights"][np.argmax(sort_for_factors["data"], axis=-1) == 2])
        QCD_node_2 = np.sum(sort_for_factors["QCD_weights"][np.argmax(sort_for_factors["QCD"], axis=-1) == 2])

        # constructing and solving linear equation system
        left_side = np.array([[tt_node_0, Wjets_node_0, QCD_node_0], [tt_node_1, Wjets_node_1, QCD_node_1], [tt_node_2, Wjets_node_2, QCD_node_2]])
        right_side = np.array([data_node_0, data_node_1, data_node_2])
        factors = np.linalg.solve(left_side, right_side)

        print("left right factors", left_side, right_side, factors)

        # norm factors as cross check, but fo the difference, un-altered yields are wanted
        factors = [1.0, 1.0, 1.0]

        # plotting and constructing SR bins for Anti-selection
        fig, ax = plt.subplots(figsize=(12, 10))
        hep.style.use("CMS")
        hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
        hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)

        pred_dict = {}
        for i, proc in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            hist_list, label_list = [], []
            for contrib in sort_for_factors.keys():
                if "weight" in contrib:
                    continue
                # only argmax
                mask = np.argmax(sort_for_factors[contrib], axis=-1) == i
                if contrib == "data":
                    data_boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False), storage=bh.storage.Weight())  # , storage=bh.storage.Weight())
                    data_boost_hist.fill(sort_for_factors[contrib][mask][:, i], weight=sort_for_factors[contrib + "_weights"][mask])
                elif contrib == "T5qqqqWW":
                    sig_boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False), storage=bh.storage.Weight())  # , storage=bh.storage.Weight())
                    sig_boost_hist.fill(sort_for_factors[contrib][mask][:, i], weight=sort_for_factors[contrib + "_weights"][mask])
                    sig_proc = self.config_inst.get_process(self.config_inst.get_aux("DNN_process_template")["SR_Anti"][contrib][0])

                else:
                    factor = 1.0
                    if contrib == "ttjets":
                        factor = factors[0]
                    if contrib == "Wjets":
                        factor = factors[1]
                    if contrib == "QCD":
                        factor = factors[2]
                    boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False), storage=bh.storage.Weight())  # , storage=bh.storage.Weight())
                    boost_hist.fill(sort_for_factors[contrib][mask][:, i], weight=sort_for_factors[contrib + "_weights"][mask])  # -1
                    hist_list.append(boost_hist * factor)
                    label_list.append(contrib)

            pred_dict[proc] = {"hist": hist_list, "label": label_list, "data": data_boost_hist}

        hep.histplot(data_boost_hist, histtype="errorbar", label=contrib, color="black", ax=ax)
        hep.histplot(hist_list, histtype="fill", stack=True, label=label_list, ax=ax)  # bins=hist_list[0].axes[0].edges,
        hep.histplot(sig_boost_hist, histtype=sig_proc.aux["histtype"], label=sig_proc.label, color=sig_proc.color, ax=ax)
        ax.set_xlabel("DNN Scores in Signal node", fontsize=24)
        ax.set_ylabel("Counts", fontsize=24)
        ax.set_ylim(10 ** (-3), 10**5)
        ax.set_xlim(0.2, 1)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0, prop={"size": 18})
        self.output().parent.touch()
        plt.savefig(self.output().parent.path + "/Anti_SR.png", bbox_inches="tight")
        plt.savefig(self.output().parent.path + "/Anti_SR.pdf", bbox_inches="tight")
        ax.set_yscale("log")
        plt.savefig(self.output().parent.path + "/Anti_SR_log.png", bbox_inches="tight")
        plt.savefig(self.output().parent.path + "/Anti_SR_log.pdf", bbox_inches="tight")
        # that still works, since signal is last entry, but is very ugly
        # predicting QCD in SR by removing QCD and substracting Data-EWK
        QCD_old = hist_list[label_list.index("QCD")]
        del hist_list[label_list.index("QCD")]
        EWK = sum(hist_list)

        F_Sel_Anti = self.input()["F_Sel_Anti"]["F_Sel_Anti"].load()["F_Sel_Anti"]
        delta_err = self.input()["F_Sel_Anti"]["factors"].load()["error_delta"]
        print("loaded F_Sel_Anti:", F_Sel_Anti)

        QCD_new_values = data_boost_hist.counts() - EWK.counts()
        # quadratic sum of errors on QCD yields with fit error
        QCD_new_errors = np.sqrt(EWK.variances() + data_boost_hist.variances() + delta_err) * F_Sel_Anti
        # set to 0.5 FIXME
        # F_Sel_Anti = 0.5
        # FIXME if bkg > MC, take QCD MC as an better estimate
        # QCD_new_values = QCD_new.values()
        print("\nQCD_new_values", QCD_new_values)
        QCD_new_values[QCD_new_values < 0] = QCD_old.values()[QCD_new_values < 0]
        # Placeholder to catch empty bins
        QCD_new_values[QCD_new_values == 0] = 10 ** (-3)
        print("With MC QCD correction and * F_Sel_Anti", QCD_new_values * F_Sel_Anti)

        diff_data_EWK = []
        diff_err = []

        for key in pred_dict.keys():
            EWK_num = pred_dict[key]["hist"][0].sum().value + pred_dict[key]["hist"][1].sum().value
            data = pred_dict[key]["data"].sum().value
            diff_data_EWK.append(data - EWK_num)
            diff_err.append(np.sqrt(sum(pred_dict[key]["hist"][0].variances()) + sum(pred_dict[key]["hist"][1].variances()) + sum(pred_dict[key]["data"].variances())))

        SR_prediction = {"SR_prediction": (QCD_new_values * F_Sel_Anti).tolist(), "SR_err": (QCD_new_errors * F_Sel_Anti).tolist(), "diff_data_EWK": diff_data_EWK, "diff_err": diff_err, "F_Sel_Anti": F_Sel_Anti}
        for key in sort_for_factors.keys():
            if not "weights" in key:
                continue
            print(key, np.sum(sort_for_factors[key]))
        from IPython import embed

        embed()
        self.output().dump(SR_prediction)

        """
        SR0b_scores = {}
        for dat in tqdm(self.datasets_to_process):
            # accessing the input and unpacking the condor submission structure
            cat = "SR0b"
            sumOfHists = []
            arr = inp[cat][cat+"_"+dat]["array"].load()
            weights = inp[cat][cat+"_"+dat]["weights"].load()
            DNNId = inp[cat][cat+"_"+dat]["DNNId"].load()
            DNN_scores, SR_weights = [], []
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
                DNN_scores.append(y_predictions[mask][:,-1])
                SR_weights.append(weight_test[mask])
            SR0b_scores[dat] = np.concatenate(DNN_scores)
            SR0b_scores[dat+"_weight"]=np.concatenate(SR_weights)
        """
