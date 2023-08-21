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
from tasks.arraypreparation import ArrayNormalisation
from tasks.multiclass import PytorchMulticlass
from tasks.base import HTCondorWorkflow, DNNTask

import utils.pytorch_base as util


class ArrayPlotting(CoffeaTask):  # , HTCondorWorkflow, law.LocalWorkflow):
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    formats = luigi.ListParameter(default=["png", "pdf"])
    density = luigi.BoolParameter(default=False)
    unblinded = luigi.BoolParameter(default=False)
    signal = luigi.BoolParameter(default=False)
    divide_by_binwidth = luigi.BoolParameter(default=False)
    debug = luigi.BoolParameter(default=False)
    merged = luigi.BoolParameter(default=False)

    def requires(self):
        if self.debug:
            return {sel: CoffeaProcessor.req(self, debug=True, workflow="local") for sel in self.channel}

        if self.merged:
            return {"merged": MergeArrays.req(self)}

        return {
            sel: CoffeaProcessor.req(
                self,
                lepton_selection=sel,
                # workflow="local",
            )
            for sel in self.channel
        }

    def output(self):
        if self.merged:
            return {
                var
                + cat
                + ending: {
                    "nominal": self.local_target(cat + "/" + "density/" * self.density + var + "." + ending),
                    "log": self.local_target(cat + "/" + "density/" * self.density + "/log/" + var + "." + ending),
                }
                for var in self.config_inst.variables.names()
                for cat in self.config_inst.categories.names()
                for ending in self.formats
            }
        return {
            var
            + cat
            + lep
            + ending: {
                "nominal": self.local_target(cat + "/" + lep + "/" + "density/" * self.density + var + "." + ending),
                "log": self.local_target(cat + "/" + lep + "/" + "density/" * self.density + "/log/" + var + "." + ending),
            }
            for var in self.config_inst.variables.names()
            for cat in self.config_inst.categories.names()
            for lep in self.channel
            for ending in self.formats
        }

    def store_parts(self):
        parts = tuple()
        if self.debug:
            parts += ("debug",)
        if self.merged:
            parts += ("merged",)
        if self.unblinded:
            parts += ("unblinded",)
        if self.signal:
            parts += ("signal",)
        return super(ArrayPlotting, self).store_parts() + (self.analysis_choice,) + parts

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    def get_density(self, hist):
        density = hist / hist.sum()
        if self.divide_by_binwidth:
            areas = np.prod(hist.axes.widths, axis=0)
            density = density / areas
        return density

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        # making clear which index belongs to which variable
        var_names = self.config_inst.variables.names()
        print(var_names)
        for var in tqdm(self.config_inst.variables):
            # defining position of var
            ind = var_names.index(var.name)
            if var.x_discrete:
                ind = var_names.index(var.name.split("_")[0])

            # iterating over lepton keys
            for lep in self.input().keys():
                # accessing the input and unpacking the condor submission structure
                if self.merged:
                    np_dict = self.input()[lep]
                else:
                    np_dict = {}
                    for key in self.input()[lep]["collection"].targets[0].keys():
                        # for key in self.input()[lep].keys():
                        np_dict.update({key: self.input()[lep]["collection"].targets[0][key]})
                        # np_dict.update({key: self.input()[lep][key]})
                for cat in self.config_inst.categories.names():
                    sumOfHists = []
                    if self.unblinded:
                        fig, (ax, rax) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0})
                    else:
                        fig, ax = plt.subplots(figsize=(12, 10))
                    hep.style.use("CMS")
                    # hep.style.use("CMS")
                    # hep.cms.label(
                    # label="Private Work",
                    # loc=0,
                    # ax=ax,
                    # )
                    hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
                    # save histograms for ratio computing
                    hist_counts = {}
                    if self.unblinded:
                        # filling all data in one boost_hist
                        data_boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                    if self.signal:
                        signal_boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                    for dat in self.datasets_to_process:
                        proc = self.config_inst.get_process(dat)
                        if not proc.aux["isData"]:
                            boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                        # for key, value in np_dict.items():
                        # for pro in self.get_proc_list([dat]):
                        key = cat + "_" + dat
                        # this will only be true for merged
                        if key in np_dict.keys():
                            if proc.aux["isData"] and self.unblinded:
                                data_boost_hist.fill(np_dict[key]["array"].load()[:, ind])
                                # np.load(value["array"].path)  # , weight=np.load(value["weights"].path))
                            elif proc.aux["isSignal"] and self.signal:
                                signal_boost_hist.fill(np_dict[key]["array"].load()[:, ind])
                            elif not proc.aux["isData"] and not proc.aux["isSignal"]:
                                boost_hist.fill(np_dict[key]["array"].load()[:, ind], weight=np_dict[key]["weights"].load())
                        if not self.merged:
                            for pro in self.get_proc_list([dat]):
                                boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                                k = cat + "_" + pro
                                for key in np_dict.keys():
                                    if k in key:
                                        boost_hist.fill(np_dict[key]["array"].load()[:, ind], weight=np_dict[key]["weights"].load())
                                hep.histplot(boost_hist, label=k, histtype="step", ax=ax)

                        if self.divide_by_binwidth:
                            # FIXME hist not defined?
                            boost_hist = boost_hist / np.prod(hist.axes.widths, axis=0)
                        if self.density:
                            boost_hist = self.get_density(boost_hist)
                        # don't stack data and signal, defined in config/processes
                        if proc.aux["isData"]:
                            continue
                        if proc.aux["isSignal"]:
                            continue
                        hist_counts.update({dat: {"hist": boost_hist, "label": "{} {}: {}".format(proc.label, lep, np.round(boost_hist.sum(), 2)), "color": proc.color}})  # , histtype=proc.aux["histtype"])})
                        # hep.histplot(boost_hist, label="{} {}: {}".format(proc.label, lep, boost_hist.sum()), color=proc.color, histtype=proc.aux["histtype"], ax=ax)
                        sumOfHists.append(boost_hist.sum())
                    # sorting the labels/handels of the plt hist by descending magnitude of integral
                    order = np.argsort(np.array(sumOfHists))

                    # one histplot together, ordered by integral
                    # can't stack seperate histplot calls, so we have do it like that
                    hist_list, label_list, color_list = [], [], []
                    for key in np.array(list(hist_counts.keys()))[order]:
                        hist_list.append(hist_counts[key]["hist"])
                        label_list.append(hist_counts[key]["label"])
                        color_list.append(hist_counts[key]["color"])
                    if self.merged:
                        hep.histplot(hist_list, histtype="fill", stack=True, label=label_list, color=color_list, ax=ax)
                    # deciated data plotting
                    if self.unblinded:
                        proc = self.config_inst.get_process("data")
                        hep.histplot(data_boost_hist, label="{} {}: {}".format(proc.label, lep, np.round(data_boost_hist.sum(), 2)), color=proc.color, histtype=proc.aux["histtype"], ax=ax)
                        sumOfHists.append(data_boost_hist.sum())
                        hist_counts.update({"data": {"hist": data_boost_hist}})
                    # plot signal last
                    if self.signal:
                        prc = {"0b": self.config_inst.get_process("SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8"), "mb": self.config_inst.get_process("T1tttt")}[self.analysis_choice]
                        hep.histplot(signal_boost_hist, label="{} {}: {}".format(prc.label, lep, np.round(signal_boost_hist.sum(), 2)), color=prc.color, histtype=prc.aux["histtype"], ax=ax)
                        sumOfHists.append(signal_boost_hist.sum())
                        hist_counts.update({prc.name: {"hist": signal_boost_hist}})
                    # missing boost hist divide and density
                    handles, labels = ax.get_legend_handles_labels()
                    if self.merged:
                        # handles = [h for _, h in sorted(zip(sumOfHists, handles))]
                        handles = [h for _, h in total_ordering(zip(sumOfHists, handles))]
                        # labels = [l for _, l in sorted(zip(sumOfHists, labels))]
                        labels = [l for _, l in total_ordering(zip(sumOfHists, labels))]
                    ax.legend(
                        handles,
                        labels,
                        ncol=1,
                        title=cat,
                        loc="upper right",
                        bbox_to_anchor=(1, 1),
                        borderaxespad=0,
                        prop={"size": 12},
                    )
                    ax.set_ylabel(var.get_full_y_title(), fontsize=18)
                    ax.tick_params(axis="both", which="major", labelsize=18)
                    if not var.x_discrete:
                        ax.set_xlim(var.binning[1], var.binning[2])

                    if self.unblinded:
                        MC_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                        data_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                        for dat, hist in hist_counts.items():
                            proc = self.config_inst.get_process(dat)
                            if proc.aux["isData"]:
                                data_hist += hist["hist"]
                            elif not proc.aux["isSignal"]:
                                MC_hist += hist["hist"]
                        ratio = data_hist / MC_hist
                        stat_unc = np.sqrt(ratio * (ratio / MC_hist + ratio / data_hist))
                        rax.axhline(1.0, color="black", linestyle="--")
                        rax.fill_between(ratio.axes[0].centers, 1 - 0.023, 1 + 0.023, alpha=0.3, facecolor="black")
                        hep.histplot(ratio, color="black", histtype="errorbar", stack=False, yerr=stat_unc, ax=rax)
                        rax.set_xlabel(var.get_full_x_title(), fontsize=18)
                        if var.x_discrete:
                            rax.set_xlim(var.binning[0], var.binning[-1])
                        if not var.x_discrete:
                            rax.set_xlim(var.binning[1], var.binning[2])
                        rax.set_ylabel("Ratio Data/MC", fontsize=18)
                        rax.set_ylim(0.5, 1.5)
                        rax.tick_params(axis="both", which="major", labelsize=18)
                    else:
                        ax.set_xlabel(var.get_full_x_title(), fontsize=18)

                    for ending in self.formats:
                        outputKey = var.name + cat + lep + ending
                        if self.merged:
                            outputKey = var.name + cat + ending
                        # create dir
                        self.output()[outputKey]["nominal"].parent.touch()
                        self.output()[outputKey]["log"].parent.touch()

                        ax.set_yscale("linear")
                        plt.savefig(self.output()[outputKey]["nominal"].path, bbox_inches="tight")

                        ax.set_yscale("log")
                        ax.set_ylim(5e-2, 2e6)
                        # ax.set_yticks(np.arange(10))
                        ax.set_yticks([10 ** (i - 1) for i in range(8)])
                        # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                        plt.savefig(self.output()[outputKey]["log"].path, bbox_inches="tight")
                    plt.gcf().clear()
                    plt.close(fig)


class StitchingPlot(CoffeaTask):
    "task to print distribution of binned MC samples"
    # channel = luigi.Parameter(default="Muon")
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    formats = luigi.ListParameter(default=["png", "pdf"])
    variable = luigi.Parameter(default="HT")

    def requires(self):
        return {
            "merged": MergeArrays.req(self, channel=self.channel, datasets_to_process=self.datasets_to_process),
            "base": CoffeaProcessor.req(
                self,
                lepton_selection=self.channel[0],
                datasets_to_process=self.datasets_to_process,
                # workflow="local"
            ),
        }

    def output(self):
        return {dat + ending: self.local_target("{}_weighted_stitching_plot.{}".format(dat, ending)) for dat in self.datasets_to_process for ending in self.formats}

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    def run(self):
        # making clear which index belongs to which variable
        var_names = self.config_inst.variables.names()
        merged = self.input()["merged"]
        base = self.input()["base"]
        var = self.config_inst.get_variable("HT")
        inp_dict = self.input()["base"]["collection"].targets[0]

        for dat in tqdm(self.datasets_to_process, unit="dataset"):
            base_dict = {}
            proc_list = self.get_proc_list([dat])

            # need to combine filesets in case there were multiple for a sub process
            for key in inp_dict.keys():
                for pro in proc_list:
                    if pro in key:
                        k = "_".join(key.split("_")[1:-1])
                        base_dict.update({k: {"array": np.array([]), "weights": np.array([])}})
            for key in inp_dict.keys():
                for pro in proc_list:
                    if pro in key:
                        k = "_".join(key.split("_")[1:-1])
                        base_dict[k]["array"] = np.append(base_dict[k]["array"], inp_dict[key]["array"].load()[:, var_names.index(var.name)])
                        base_dict[k]["weights"] = np.append(base_dict[k]["weights"], inp_dict[key]["weights"].load())

            fig, ax = plt.subplots(figsize=(12, 10))
            hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)

            hist_list, label_list = [], []
            for key, dic in base_dict.items():
                if not "TTTo" in key:
                    boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                    boost_hist.fill(dic["array"], weight=dic["weights"])  # / dic["sum_gen_weights"])
                    hist_list.append(boost_hist)
                    label_list.append(key)

            # hep.histplot(boost_hist, histtype="step", label=key, ax=ax)
            hep.histplot(hist_list, histtype="fill", stack=True, label=label_list, ax=ax)

            # in that order so lines are on top of stacked plot
            for key, dic in base_dict.items():
                if "TTTo" in key:
                    boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                    boost_hist.fill(dic["array"], weight=dic["weights"])  # / dic["sum_gen_weights"])
                    hep.histplot(boost_hist, histtype="step", label=key, ax=ax, linewidth=3)

            for k in list(merged.keys()):
                if dat in k:
                    pro = k
            proc = self.config_inst.get_process(pro.split("_")[1])

            merged_boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
            merged_boost_hist.fill(merged[pro]["array"].load()[:, var_names.index(var.name)], weight=merged[pro]["weights"].load())
            hep.histplot(merged_boost_hist, label=proc.label, histtype="step", ax=ax, linewidth=2)

            ax.set_ylabel(var.get_full_y_title())
            ax.set_xlabel(var.get_full_x_title())
            ax.set_yscale("log")
            ax.legend(
                ncol=1,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
            )

            for ending in self.formats:
                self.output()[dat + ending].parent.touch()
                plt.savefig(self.output()[dat + ending].path, bbox_inches="tight")
            plt.gcf().clear()
            plt.close(fig)


class CutflowPlotting(CoffeaTask):

    """
    Plotting cutflow produced by coffea
    Utility for doing log scale
    """

    log_scale = luigi.BoolParameter()

    def requires(self):
        return {
            "hists": GroupCoffea.req(self),
            "root_plots": CollectInputData.req(self),
        }

    def output(self):
        path = ""
        if self.log_scale:
            path += "_log"
        out = {
            "cutflow": {cat: self.local_target(cat + "_cutflow{}.pdf".format(path)) for cat in self.config_inst.categories.names()},
            "n_minus1": {cat: self.local_target(cat + "_minus{}.pdf".format(path)) for cat in self.config_inst.categories.names()},
        }
        return out

    def store_parts(self):
        return super(CutflowPlotting, self).store_parts() + (self.lepton_selection,)

    def run(self):
        print("Doing Cutflow plots")
        cutflow = self.input()["hists"][self.lepton_selection + "_cutflow"].load()
        root_plots = self.input()["root_plots"]["cutflow"].load()
        root_cuts = ["No cuts", "HLT_Or", "MET Filters", "Good Muon", "Veto Lepton cut", "Njet >=3"]
        root_cuts += ["weights applied"]
        # categories = [c.name for c in cutflow.axis("category").identifiers()]
        # processes = [p.name for p in cutflow.axis("dataset").identifiers() if not "data" in p.name]
        val = cutflow.values()
        for i, cat in enumerate(self.config_inst.categories.names()):
            fig, ax = plt.subplots(figsize=(12, 10))
            hep.cms.text("Private work (CMS simulation)")
            for dat in self.datasets_to_process:
                proc = self.config_inst.get_process(dat)
                proc_list = self.get_proc_list([dat])
                boost_hist = bh.Histogram(bh.axis.Regular(20, 0, 20))
                arr_list = []
                # combine subprocesses
                for pro in proc_list:
                    arr_list.append(val[(pro, cat, cat)])
                weights = root_plots[self.lepton_selection][pro] + list(sum(arr_list))
                boost_hist.fill(np.arange(0, 20, 1), weight=weights[:20])
                hep.histplot(boost_hist, histtype="step", label=proc.label, color=proc.color, ax=ax)

                # coffea.hist.plot1d(
                # cutflow[[dat], category, :].project("dataset", "cutflow"),
                # overlay="dataset",
                # # legend_opts={"labels":category}
                # ax=ax
                # )
                # printing out numbers
                # print("\n Cuts for SingleMuon", category)
                # cuts = self.config_inst.get_category(category).get_aux("cuts")
                # for i, num in enumerate(val[("SingleMuon", category, category)]):
                # if i >= len(cuts):
                # break
                # print(num, cuts[i])
            cuts = root_cuts + self.config_inst.get_category(cat).get_aux("cuts")
            n_cuts = len(cuts)
            ax.set_xticks(np.arange(0.5, n_cuts + 0.5, 1))
            ax.set_xticklabels([" ".join(cut) for cut in cuts], rotation=80, fontsize=12)
            if self.log_scale:
                ax.set_yscale("log")
                ax.set_ylim(1e-1, 1e8)  # potential zero bins
                locmaj = tick.LogLocator(base=10, numticks=10)
                ax.yaxis.set_major_locator(locmaj)
                locmin = tick.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=10)
                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(tick.NullFormatter())
            handles, labels = ax.get_legend_handles_labels()
            # for i in range(len(labels)):
            # labels[i] = labels[i] + " " + categories[i]
            ax.legend(handles, labels, title="Category: ", ncol=1, loc="best")
            self.output()["cutflow"][cat].parent.touch()
            plt.savefig(self.output()["cutflow"][cat].path, bbox_inches="tight")
            ax.figure.clf()

        minus = self.input()["hists"][self.lepton_selection + "_n_minus1"].load()
        val = minus.values()
        for i, cat in enumerate(self.config_inst.categories.names()):
            fig, ax = plt.subplots(figsize=(12, 10))
            hep.cms.text("Private work (CMS simulation)")
            for dat in self.datasets_to_process:
                proc = self.config_inst.get_process(dat)
                proc_list = self.get_proc_list([dat])
                boost_hist = bh.Histogram(bh.axis.Regular(20, 0, 20))
                arr_list = []
                # combine subprocesses
                for pro in proc_list:
                    arr_list.append(val[(pro, cat, cat)])
                boost_hist.fill(np.arange(0, 20, 1), weight=sum(arr_list))
                hep.histplot(boost_hist, histtype="step", label=proc.label, color=proc.color, ax=ax)

                # for dat in self.datasets_to_process:
                # fig, ax = plt.subplots(figsize=(12, 10))
                # hep.cms.text("Private work (CMS simulation)")
                # for i, category in enumerate(self.config_inst.categories.names()):
                # ax = coffea.hist.plot1d(
                # minus[[dat], category, :].project("dataset", "cutflow"),
                # overlay="dataset",
                # # legend_opts={"labels":category}
                # )
            n_cuts = len(self.config_inst.get_category(cat).get_aux("cuts"))
            ax.set_xticks(np.arange(0.5, n_cuts + 1.5, 1))
            ax.set_xticklabels(["total"] + [" ".join(cut) for cut in self.config_inst.get_category(cat).get_aux("cuts")], rotation=80, fontsize=12)
            if self.log_scale:
                ax.set_yscale("log")
                ax.set_ylim(1e-8, 1e8)  # potential zero bins
            handles, labels = ax.get_legend_handles_labels()
            # for i in range(len(labels)):
            # labels[i] = labels[i] + " " + categories[i]
            ax.legend(handles, labels, title="Category: ", ncol=1, loc="best")
            plt.savefig(self.output()["n_minus1"][cat].path, bbox_inches="tight")
            ax.figure.clf()

        """
        N-1 Plots
        allCuts = {"twoElectron", "noMuon", "leadPt20"}
        for cut in allCuts:
            nev = selection.all(*(allCuts - {cut})).sum()
            print(f"Events passing all cuts, ignoring {cut}: {nev}")

        nev = selection.all(*allCuts).sum()
        print(f"Events passing all cuts: {nev}")
        """


"""
Plots to visulize DNN performance
"""


class DNNHistoryPlotting(DNNTask):

    """
    opening history callback and plotting curves for training
    """

    def requires(self):
        return (
            PytorchMulticlass.req(
                self,
                n_layers=self.n_layers,
                n_nodes=self.n_nodes,
                dropout=self.dropout,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,  # , debug=True
            ),
        )

    def output(self):
        return {
            "loss_plot_png": self.local_target("torch_loss_plot.png"),
            "acc_plot_png": self.local_target("torch_acc_plot.png"),
            "loss_plot_pdf": self.local_target("torch_loss_plot.pdf"),
            "acc_plot_pdf": self.local_target("torch_acc_plot.pdf"),
        }

    def store_parts(self):
        # make plots for each use case
        return (
            super(DNNHistoryPlotting, self).store_parts()
            + (self.analysis_choice,)
            # + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
        )

    @law.decorator.timeit(publish_message=True)
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self):
        # retrieve history callback for trainings
        accuracy_stats = self.input()[0]["collection"].targets[0]["accuracy_stats"].load()
        loss_stats = self.input()[0]["collection"].targets[0]["loss_stats"].load()
        # read in values, skip first for val since Trainer does a validation step beforehand
        train_loss = loss_stats["train"]
        val_loss = loss_stats["val"]

        train_acc = accuracy_stats["train"]
        val_acc = accuracy_stats["val"]

        self.output()["loss_plot_png"].parent.touch()
        plt.plot(
            np.arange(0, len(val_loss), 1),
            val_loss,
            label="loss on valid data",
            color="orange",
        )
        plt.plot(
            np.arange(1, len(train_loss) + 1, 1),
            train_loss,
            label="loss on train data",
            color="green",
        )
        plt.legend()
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=16)
        plt.savefig(self.output()["loss_plot_png"].path)
        plt.savefig(self.output()["loss_plot_pdf"].path)
        plt.gcf().clear()

        plt.plot(
            np.arange(0, len(val_acc), 1),
            val_acc,
            label="acc on vali data",
            color="orange",
        )
        plt.plot(
            np.arange(1, len(train_acc) + 1, 1),
            train_acc,
            label="acc on train data",
            color="green",
        )
        plt.legend()
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=16)
        plt.savefig(self.output()["acc_plot_png"].path)
        plt.savefig(self.output()["acc_plot_pdf"].path)
        plt.gcf().clear()


class DNNEvaluationPlotting(DNNTask):
    normalize = luigi.Parameter(default="true", description="if confusion matrix gets normalized")

    def requires(self):
        return dict(
            data=ArrayNormalisation.req(self),
            model=PytorchMulticlass.req(
                self,
                n_layers=self.n_layers,
                n_nodes=self.n_nodes,
                dropout=self.dropout,
                debug=False,
            ),
        )
        # return DNNTrainer.req(
        #    self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout
        # )

    def output(self):
        return {
            "ROC_png": self.local_target("pytorch_ROC.png"),
            "confusion_matrix_png": self.local_target("pytorch_confusion_matrix.png"),
            "ROC_pdf": self.local_target("pytorch_ROC.pdf"),
            "confusion_matrix_pdf": self.local_target("pytorch_confusion_matrix.pdf"),
        }

    def store_parts(self):
        # make plots for each use case
        return (
            super(DNNEvaluationPlotting, self).store_parts()
            + (self.analysis_choice,)
            # + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
        )

    @law.decorator.timeit(publish_message=True)
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self):
        # from IPython import embed;embed()

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")["N" + self.channel].keys())
        all_processes = list(self.config_inst.get_aux("DNN_process_template")["N" + self.channel].keys())

        path = self.input()["model"]["collection"].targets[0]["model"].path

        # load complete model
        reconstructed_model = torch.load(path)

        # load all the prepared data thingies
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()

        test_dataset = util.ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(y_test))

        # val_loss, val_acc = reconstructed_model.evaluate(X_test, y_test)
        # print("Test accuracy:", val_acc)

        y_predictions = []
        with torch.no_grad():
            reconstructed_model.eval()
            for X_test_batch, y_test_batch in test_loader:
                y_test_pred = reconstructed_model(X_test_batch)

                y_predictions.append(y_test_pred.numpy())

            # test_predict = reconstructed_model.predict(X_test)
            y_predictions = np.array(y_predictions[0])
            test_predictions = np.argmax(y_predictions, axis=1)

            # "signal"...
            predict_signal = np.array(y_predictions)[:, 1]

        self.output()["confusion_matrix_png"].parent.touch()

        # Roc curve, compare labels and predicted labels
        fpr, tpr, tresholds = sk.metrics.roc_curve(y_test[:, 1], predict_signal)

        plt.plot(
            fpr,
            tpr,
            label="AUC: {0}".format(np.around(sk.metrics.auc(fpr, tpr), decimals=3)),
        )
        plt.plot([0, 1], [0, 1], ls="--")
        plt.xlabel(" fpr ", fontsize=16)
        plt.ylabel("tpr", fontsize=16)
        # plt.title("ROC", fontsize=16)
        plt.legend(title="ROC")
        hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=12)
        plt.savefig(self.output()["ROC_png"].path, bbox_inches="tight")
        plt.savefig(self.output()["ROC_pdf"].path, bbox_inches="tight")
        plt.gcf().clear()

        # from IPython import embed;embed()
        # Correlation Matrix Plot
        # plot correlation matrix
        pred_matrix = sk.metrics.confusion_matrix(
            np.argmax(y_test, axis=-1),
            test_predictions,  # np.concatenate(test_predictions),
            normalize=self.normalize,
        )

        print(pred_matrix)
        # TODO
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        # cax = ax.matshow(pred_matrix, vmin=-1, vmax=1)
        cax = ax.imshow(pred_matrix, vmin=0, vmax=1, cmap="plasma")
        fig.colorbar(cax)
        for i in range(n_processes):
            for j in range(n_processes):
                text = ax.text(
                    j,
                    i,
                    np.round(pred_matrix[i, j], 3),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=18,
                )
        ticks = np.arange(0, n_processes, 1)
        # Let the horizontal axes labeling appear on bottom
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(all_processes)
        ax.set_yticklabels(all_processes)
        ax.set_xlabel("Predicted Processes", fontsize=18)
        ax.set_ylabel("Real Processes", fontsize=18)
        hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=14, ax=ax)
        # ax.grid(linestyle="--", alpha=0.5)
        plt.savefig(self.output()["confusion_matrix_png"].path, bbox_inches="tight")
        plt.savefig(self.output()["confusion_matrix_pdf"].path)  # , bbox_inches="tight")
        plt.gcf().clear()
