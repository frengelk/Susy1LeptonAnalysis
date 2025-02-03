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
from tasks.makefiles import CollectInputData, CalcBTagSF
from tasks.grouping import GroupCoffea, MergeArrays, MergeShiftArrays
from tasks.arraypreparation import ArrayNormalisation, CrossValidationPrep
from tasks.multiclass import PytorchMulticlass, PredictDNNScores, PytorchCrossVal, CalcNormFactors
from tasks.base import HTCondorWorkflow, DNNTask
from tasks.inference import ConstructInferenceBins, GetShiftedYields
from tasks.bgestimation import EstimateQCDinSR

import utils.pytorch_base as util


# to be used in all plots where 2016 lumi get's merged
def lumi_title(self, ax):
    lumi = self.config_inst.get_aux("lumi")
    if self.year == "2016" and "VFP" in self.version:
        lumi = self.config_inst.get_aux("pre_post_lumi")[self.version.split("VFP")[0]]
    hep.cms.lumitext(text=str(np.round(lumi / 1000, 2)) + r"$fb^{-1}$", ax=ax)


class ArrayPlotting(CoffeaTask):  # , HTCondorWorkflow, law.LocalWorkflow):
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    formats = luigi.ListParameter(default=["png", "pdf"])
    density = luigi.BoolParameter(default=False)
    unblinded = luigi.BoolParameter(default=False)
    signal = luigi.BoolParameter(default=False)
    divide_by_binwidth = luigi.BoolParameter(default=False)
    debug = luigi.BoolParameter(default=False)
    merged = luigi.BoolParameter(default=False)
    do_shifts = luigi.BoolParameter(default=False)

    def requires(self):
        if self.debug:
            return {sel: CoffeaProcessor.req(self, debug=True, workflow="local") for sel in self.channel}

        if self.merged:
            out = {"merged": MergeArrays.req(self)}
            if self.do_shifts:
                out.update({"shifts": MergeShiftArrays.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "QCD", "Rare", "DY"], shifts=["systematic_shifts", "TotalUp", "TotalDown", "JERUp", "JERDown"])})
            return out

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
                for cat in [self.category]  # self.config_inst.categories.names()
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
            for cat in [self.category]  # self.config_inst.categories.names()
            for lep in self.channel
            for ending in self.formats
        }

    def store_parts(self):
        parts = tuple()
        if self.debug:
            parts += ("debug",)
        if self.merged:
            parts += ("merged",)
        if self.do_shifts:
            parts += ("do_shifts",)
        if self.unblinded:
            parts += ("unblinded",)
        if self.signal:
            parts += ("signal",)
        return super(ArrayPlotting, self).store_parts() + ("_".join(self.channel),) + parts

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
        if self.do_shifts:
            syst_per_dat = {}
            shifts = self.input()["shifts"]["collection"].targets[0]
            print(shifts.keys())
            # for k in shifts.keys():
            #    weights = shifts[k]["weights"].load()
            #    print(k,np.sum(weights))
            for dat in self.datasets_to_process:
                relative_shifts = []
                # only bg processes here
                if self.config_inst.get_aux("signal_process").replace("V", "W") in dat or dat in self.config_inst.get_aux("data"):
                    continue
                for shift in self.config_inst.get_aux("systematic_shifts") + ["TotalUp"]:
                    if not "Up" in shift:
                        continue
                    k_up = "_".join((self.category, dat, shift))
                    k_down = "_".join((self.category, dat, shift.replace("Up", "Down")))
                    sum_up = np.sum(shifts[k_up]["weights"].load())
                    sum_down = np.sum(shifts[k_down]["weights"].load())
                    sum_nominal = np.sum(self.input()["merged"]["_".join((self.category, dat))]["weights"].load())

                    # write relative shift per dat to dict
                    relative_shifts.append((abs(sum_nominal - sum_down) + abs(sum_nominal - sum_up)) / 2 / sum_nominal)

                syst_per_dat[dat] = np.sum(np.array(relative_shifts) ** 2)
            # lumi as well
            total_syst = np.sqrt(sum(syst_per_dat.values()) + self.config_inst.get_aux("lumi_unc_per_year")[self.year] ** 2)
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
                    if "shifts" == lep:
                        continue
                    np_dict = self.input()[lep]
                else:
                    np_dict = {}
                    for key in self.input()[lep]["collection"].targets[0].keys():
                        # for key in self.input()[lep].keys():
                        np_dict.update({key: self.input()[lep]["collection"].targets[0][key]})
                        # np_dict.update({key: self.input()[lep][key]})
                # for cat in self.config_inst.categories.names():
                cat = self.category
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
                # lumi is split for 2016
                lumi_title(self, ax)
                # save histograms for ratio computing
                hist_counts = {}
                signal_hists = {}
                if self.unblinded:
                    # filling all data in one boost_hist
                    data_boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                # if self.signal:
                #    signal_boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
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
                            if var.x_discrete:
                                data_boost_hist = data_boost_hist / np.prod(data_boost_hist.axes.widths, axis=0)
                            # np.load(value["array"].path)  # , weight=np.load(value["weights"].path))
                        elif proc.aux["isSignal"] and self.signal:
                            signal_boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                            signal_boost_hist.fill(np_dict[key]["array"].load()[:, ind], weight=np_dict[key]["weights"].load())
                            if var.x_discrete:
                                signal_boost_hist = signal_boost_hist / np.prod(signal_boost_hist.axes.widths, axis=0)
                            signal_hists.update(
                                {
                                    dat: {
                                        "hist": signal_boost_hist,
                                        "label": "{}".format(proc.label),
                                        "color": proc.color,
                                    }
                                }
                            )

                        elif not proc.aux["isData"] and not proc.aux["isSignal"]:
                            boost_hist.fill(np_dict[key]["array"].load()[:, ind], weight=np_dict[key]["weights"].load())
                            if var.x_discrete:
                                boost_hist = boost_hist / np.prod(boost_hist.axes.widths, axis=0)

                    if not self.merged:
                        for pro in self.get_proc_list([dat]):
                            boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                            k = cat + "_" + pro
                            for key in np_dict.keys():
                                if k in key:
                                    boost_hist.fill(np_dict[key]["array"].load()[:, ind], weight=np_dict[key]["weights"].load())
                            hep.histplot(boost_hist, label=k, histtype="step", ax=ax, flow="none")  # flow="sum",

                    if self.divide_by_binwidth:
                        boost_hist = boost_hist / np.prod(boost_hist.axes.widths, axis=0)
                    if self.density:
                        boost_hist = self.get_density(boost_hist)
                    # don't stack data and signal, defined in config/processes
                    if proc.aux["isData"]:
                        continue
                    if proc.aux["isSignal"]:
                        continue
                    hist_counts.update(
                        {
                            dat: {
                                "hist": boost_hist,
                                "label": proc.label,
                                "color": proc.color,
                            }
                        }
                    )  # , histtype=proc.aux["histtype"])})
                    # if you want yields, incorporate them like this:
                    # hist_counts.update({dat: {"hist": boost_hist, "label": "{} {}: {}".format(proc.label, lep, np.round(boost_hist.sum(), 2)), "color": proc.color}})
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
                    sum_bg = sum(hist_list)
                    hep.histplot(hist_list, histtype="fill", stack=True, label=label_list, color=color_list, ax=ax, flow="none")  # flow="sum", err=np.sqrt(sum(hist_list))
                    yerr = np.sqrt(sum_bg)
                    if self.do_shifts:
                        yerr += sum_bg * total_syst
                    # ax.fill_between(sum_bg.axes[0].centers, sum_bg + yerr, sum_bg - yerr, alpha=0.1, facecolor="grey", step="pre", hatch="x", label="Summed BKG error")
                # deciated data plotting
                if self.unblinded:
                    proc = self.config_inst.get_process("data")
                    hep.histplot(data_boost_hist, label="{} {}".format(proc.label, lep), color=proc.color, histtype=proc.aux["histtype"], ax=ax, flow="none")  # , flow="sum"
                    sumOfHists.append(data_boost_hist.sum())
                    hist_counts.update({"data": {"hist": data_boost_hist}})
                # plot signal last
                if self.signal:
                    # prc = {"0b": self.config_inst.get_process("SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8"), "mb": self.config_inst.get_process("T1tttt")}[self.analysis_choice]
                    for key, val in signal_hists.items():
                        prc = self.config_inst.get_process(key)
                        hep.histplot(val["hist"], label="{}".format(val["label"]), color=val["color"], histtype=prc.aux["histtype"], linewidth=3, ax=ax, flow="none")  # ,flow="sum"
                        sumOfHists.append(val["hist"].sum())
                        hist_counts.update({prc.name: {"hist": val["hist"]}})
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
                    ncol=2,
                    # title=cat,
                    loc="upper right",
                    bbox_to_anchor=(1, 1),
                    borderaxespad=0,
                    prop={"size": 20},
                )
                ax.set_ylabel(var.y_title, fontsize=24)  # var.get_full_y_title()
                ax.tick_params(axis="both", which="major", labelsize=24)
                if var.x_discrete:
                    ax.set_xlim(var.binning[0], var.binning[-1])
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
                    if self.do_shifts:
                        rax.fill_between(ratio.axes[0].centers, 1 - total_syst, 1 + total_syst, alpha=0.3, facecolor="black", label="Systematic Errors")
                    # else:
                    #     rax.fill_between(ratio.axes[0].centers, 1 - 0.023, 1 + 0.023, alpha=0.3, facecolor="black")
                    hep.histplot(ratio, color="black", histtype="errorbar", stack=False, yerr=stat_unc, ax=rax, label="Ratio incl. Stat. Unc.", flow="none")
                    rax.set_xlabel(var.get_full_x_title(), fontsize=18)
                    if var.x_discrete:
                        rax.set_xlim(var.binning[0], var.binning[-1])
                    if not var.x_discrete:
                        rax.set_xlim(var.binning[1], var.binning[2])
                    rax.set_ylabel("Data/MC", fontsize=24)
                    rax.set_ylim(0.5, 1.5)
                    rax.tick_params(axis="both", which="major", labelsize=24)
                    print("Ratio integrals", sum(MC_hist) / sum(data_hist))
                else:
                    ax.set_xlabel(var.get_full_x_title(), fontsize=24)

                for ending in self.formats:
                    outputKey = var.name + cat + lep + ending
                    if self.merged:
                        outputKey = var.name + cat + ending
                    # create dir
                    self.output()[outputKey]["nominal"].parent.touch()
                    self.output()[outputKey]["log"].parent.touch()

                    ax.set_yscale("linear")
                    ax.set_ylim(2, 4000)
                    # ax.set_ylim(auto=True)
                    ax.tick_params(axis="both", which="major", labelsize=24)
                    # for Benno ax.set_ylim(2e-2, 2e1)
                    plt.draw()  # dont know if this does anything
                    plt.savefig(self.output()[outputKey]["nominal"].path, bbox_inches="tight", dpi=600)

                    ax.set_yscale("log")
                    ax.set_ylim(2e-2, 2e10)  # FIXME
                    ax.set_yticks([10 ** (i - 1) for i in range(12)])
                    # to make sure ticks are really set
                    ax.tick_params(axis="both", which="major", labelsize=24)
                    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    plt.draw()
                    plt.savefig(self.output()[outputKey]["log"].path, bbox_inches="tight", dpi=600)
                plt.gcf().clear()
                plt.close(fig)


class Plot2Dhist(CoffeaTask):
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    formats = luigi.ListParameter(default=["png", "pdf"])
    unblinded = luigi.BoolParameter(default=False)
    signal = luigi.BoolParameter(default=False)
    divide_by_binwidth = luigi.BoolParameter(default=False)
    debug = luigi.BoolParameter(default=False)
    var1 = luigi.Parameter(default="leptonEta")
    var2 = luigi.Parameter(default="leptonPhi")

    def requires(self):
        out = MergeArrays.req(self)
        return out

    def output(self):
        return self.local_target(self.var1 + "_" + self.var2 + ".png")

    def store_parts(self):
        parts = tuple()
        if self.debug:
            parts += ("debug",)
        if self.unblinded:
            parts += ("unblinded",)
        if self.signal:
            parts += ("signal",)
        return super(Plot2Dhist, self).store_parts() + parts

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        # making clear which index belongs to which variable
        var_names = self.config_inst.variables.names()
        print(var_names)
        var1 = self.config_inst.get_variable(self.var1)
        var2 = self.config_inst.get_variable(self.var2)

        hist2D_MC = bh.Histogram(
            bh.axis.Regular(var1.binning[0], var1.binning[1], var1.binning[2]),
            bh.axis.Regular(var2.binning[0], var2.binning[1], var2.binning[2]),
        )

        hist2D_data = bh.Histogram(
            bh.axis.Regular(var1.binning[0], var1.binning[1], var1.binning[2]),
            bh.axis.Regular(var2.binning[0], var2.binning[1], var2.binning[2]),
        )

        # for var in tqdm([var1, var2]):
        # defining position of var
        ind1 = var_names.index(var1.name)
        ind2 = var_names.index(var2.name)
        # iterating over lepton keys
        # for lep in self.input().keys():
        # accessing the input and unpacking the condor submission structure
        np_dict = self.input()
        cat = self.category
        for dat in self.datasets_to_process:
            proc = self.config_inst.get_process(dat)
            dat_key = cat + "_" + dat
            # this will only be true for merged
            if dat_key in np_dict.keys():
                arr1 = np_dict[dat_key]["array"].load()[:, ind1]
                arr2 = np_dict[dat_key]["array"].load()[:, ind2]
                if proc.aux["isData"] and self.unblinded:
                    hist2D_data.fill(arr1, arr2)

                elif not proc.aux["isData"] and not proc.aux["isSignal"]:
                    hist2D_MC.fill(arr1, arr2, weight=np_dict[dat_key]["weights"].load())

        self.output().parent.touch()
        fig, ax = plt.subplots(figsize=(12, 10))
        hep.style.use("CMS")
        hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
        hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)
        ax.set_ylabel(var2.get_full_x_title(), fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.set_xlabel(var1.get_full_x_title(), fontsize=24)

        hep.hist2dplot(hist2D_MC)

        plt.savefig(self.output().path, bbox_inches="tight")
        plt.gcf().clear()
        plt.close(fig)

        if self.unblinded:
            hist_ratio = hist2D_MC / hist2D_data - 1

            fig, ax = plt.subplots(figsize=(12, 10))
            hep.style.use("CMS")
            hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
            hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)
            ax.set_ylabel(var2.get_full_x_title(), fontsize=24)
            ax.tick_params(axis="both", which="major", labelsize=24)
            ax.set_xlabel(var1.get_full_x_title(), fontsize=24)

            hep.hist2dplot(hist_ratio)

            plt.savefig(self.output().path.replace(".png", "_ratio.png"), bbox_inches="tight")
            plt.gcf().clear()
            plt.close(fig)


class SignalCategoryComparing(CoffeaTask):
    "task to print distribution of binned MC samples"
    # channel = luigi.Parameter(default="Muon")
    channel = luigi.ListParameter(default=["LeptonIncl"])  # , "Electron"
    formats = luigi.ListParameter(default=["png", "pdf"])
    signal_categories = ["SR_Anti", "SR0b"]

    def requires(self):
        return {cat: MergeArrays.req(self, channel=self.channel, datasets_to_process=self.datasets_to_process, category=cat) for cat in self.signal_categories}

    def output(self):
        return {var + "_" + ending: self.local_target("{}_signal_comparison.{}".format(var, ending)) for ending in self.formats for var in self.config_inst.variables.names()}

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    def run(self):
        # making clear which index belongs to which variable
        var_names = self.config_inst.variables.names()
        inp = self.input()
        for var in self.config_inst.variables:
            fig = plt.figure(figsize=(12, 9))
            for cat in self.signal_categories:
                cat_inp = inp[cat]
                cat_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                for key in cat_inp.keys():
                    cat_hist.fill(cat_inp[key]["array"].load()[:, var_names.index(var.name)], weight=cat_inp[key]["weights"].load())
                hep.histplot(cat_hist, label=cat, histtype="step", linewidth=1)
            plt.legend()
            plt.yscale("log")
            plt.xlabel(var.get_full_x_title(), fontsize=16)
            plt.ylabel(var.get_full_y_title(), fontsize=16)
            for ending in self.formats:
                self.output()[var.name + "_" + ending].parent.touch()
                plt.savefig(self.output()[var.name + "_" + ending].path, bbox_inches="tight")
            plt.gcf().clear()
            plt.close(fig)


class StitchingPlot(CoffeaTask):
    "task to print distribution of binned MC samples"
    # channel = luigi.Parameter(default="Muon")
    channel = luigi.ListParameter(default=["Muon"])  # , "Electron"
    formats = luigi.ListParameter(default=["png", "pdf"])
    variable = luigi.Parameter(default="HT")

    def requires(self):
        return {
            "merged": MergeArrays.req(self, channel=self.channel[:1], datasets_to_process=self.datasets_to_process),
            "base": CoffeaProcessor.req(
                self,
                lepton_selection=self.channel[0],
                datasets_to_process=self.datasets_to_process,
                # workflow="local"
            ),
        }

    def output(self):
        return {dat + ending: self.local_target("{}_weighted_stitching_plot.{}".format(dat, ending)) for dat in self.datasets_to_process for ending in self.formats}

    def store_parts(self):
        return super(StitchingPlot, self).store_parts() + (self.channel[0],)

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
                    # print(key, boost_hist.values())
                    label_list.append(key)

            # print("summed_hists", sum(hist_list).values())

            # hep.histplot(boost_hist, histtype="step", label=key, ax=ax)
            hep.histplot(hist_list, histtype="fill", stack=True, label=label_list, ax=ax)

            # in that order so lines are drawn on top of stacked plot
            for key, dic in base_dict.items():
                if "TTTo" in key:
                    boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
                    boost_hist.fill(dic["array"], weight=dic["weights"])  # / dic["sum_gen_weights"])
                    hep.histplot(boost_hist, histtype="step", label=key, ax=ax, linewidth=3)

            for k in list(merged.keys()):
                if dat in k:
                    pro = k
            proc = self.config_inst.get_process(pro.split("_")[-1])

            merged_boost_hist = bh.Histogram(self.construct_axis(var.binning, not var.x_discrete))
            merged_boost_hist.fill(merged[pro]["array"].load()[:, var_names.index(var.name)], weight=merged[pro]["weights"].load())
            hep.histplot(merged_boost_hist, label=proc.label, histtype="step", ax=ax, linewidth=2)
            # print(pro, merged_boost_hist.values())

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
        # root_cuts = ["No cuts", "HLT_Or", "MET Filters", "Good Muon", "Veto Lepton cut", "Njet >=3"]
        root_cuts = ["No cuts", "HLT_Or", "MET Filters", "1 Lepton", "Njet >=3"]
        """
        qcd="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2024_04_22/2017/MC/merged/QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8/QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8_1_merged.root"
        file = ROOT.TFile.Open(qcd)
        hist = file.Get('cutflow_LeptonIncl')
        In [30]: for i in range(1, hist.GetNbinsX() + 1):
            label = hist.GetXaxis().GetBinLabel(i)
            print(f"Bin {i} label: {label}")
        """
        root_cuts += ["weights applied"]
        # categories = [c.name for c in cutflow.axis("category").identifiers()]
        # processes = [p.name for p in cutflow.axis("dataset").identifiers() if not "data" in p.name]
        val = cutflow.values()
        cat = self.category
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


class BTagSFPlotting(CoffeaTask):

    """
    Plotting BTagSF hist
    """

    def requires(self):
        return CalcBTagSF.req(self)

    def output(self):
        return self.local_target("hists.pdf")

    def run(self):
        inp = self.input()["collection"].targets[0]
        arr, arr_up, arr_down = np.array([]), np.array([]), np.array([])
        # slooooooow...
        for key in inp.keys():
            arr = np.append(arr, inp[key]["weights"].load())
            arr_up = np.append(arr_up, inp[key]["up"].load())
            arr_down = np.append(arr_down, inp[key]["down"].load())

        fig, ax = plt.subplots(figsize=(12, 10))
        plt.xlim(-2, 2)
        plt.hist(arr, bins=1000, label="nominal", histtype="step")
        plt.hist(arr_up, bins=1000, label="up", linestyle="--", histtype="step")
        plt.hist(arr_down, bins=1000, label="down", linestyle=":", histtype="step")
        plt.legend()
        self.output().parent.touch()
        plt.savefig(self.output().path, bbox_inches="tight")
        ax.figure.clf()


class JECPlotting(CoffeaTask):
    shifts = luigi.ListParameter(default=["systematic_shifts"])

    """
    Plotting JEC hist
    """

    def requires(self):
        return {"nominal": MergeArrays.req(self), "shifts": MergeShiftArrays.req(self)}

    def output(self):
        return self.local_target("hists.png")

    def run(self):
        self.output().parent.touch()
        var_names = self.config_inst.variables.names()
        for var in ["HT", "jetPt_1", "jetPt_2", "nJets"]:
            var = self.config_inst.get_variable(var)
            ind = var_names.index(var.name)
            print(var)
            # for i, cat in enumerate(self.input()["nominal"].keys()):
            for dat in self.datasets_to_process:
                fig, ax = plt.subplots(figsize=(12, 10))
                hep.cms.text("Private work (CMS simulation)")
                proc = self.config_inst.get_process(dat)
                nominal = self.input()["nominal"][self.category + "_" + dat]
                nom_arr = nominal["array"].load()
                nom_weights = nominal["weights"].load()

                boost_hist = bh.Histogram(bh.axis.Regular(var.binning[0], var.binning[1], var.binning[2]))
                boost_hist.fill(nom_arr[:, ind], weight=nom_weights)
                hep.histplot(boost_hist, histtype="step", label=proc.label + " nominal", color=proc.color, ax=ax)
                print("\nnominal ", dat, len(nom_weights), sum(nom_weights))
                for shift in self.shifts:
                    shifts_inp = self.input()["shifts"]["collection"].targets[0][self.category + "_" + dat + "_" + shift]
                    shifts_arr = shifts_inp["array"].load()
                    shifts_weights = shifts_inp["weights"].load()
                    boost_hist = bh.Histogram(bh.axis.Regular(var.binning[0], var.binning[1], var.binning[2]))
                    boost_hist.fill(shifts_arr[:, ind], weight=shifts_weights)
                    if "Up" in shift:
                        linestyle = "--"
                    if "Down" in shift:
                        linestyle = ":"
                    hep.histplot(boost_hist, histtype="step", label=proc.label + " " + shift, color=proc.color, ax=ax, linestyle=linestyle)
                    print(shift, len(shifts_weights), sum(shifts_weights))
                plt.legend()
                plt.xlabel(var.get_full_x_title(), fontsize=16)
                plt.ylabel(var.get_full_y_title(), fontsize=16)
                plt.savefig(self.output().path.replace(".png", var.name + "_" + dat + ".png"), bbox_inches="tight")
                plt.close()


"""
Plots to visualize DNN performance
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
    @law.decorator.safe_output
    def run(self):
        # from IPython import embed;embed()

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        all_processes = list(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

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
                y_test_pred = reconstructed_model(X_test_batch).softmax(dim=1)

                y_predictions.append(y_test_pred.numpy())

            # test_predict = reconstructed_model.predict(X_test)
            y_predictions = np.array(y_predictions[0])
            test_predictions = np.argmax(y_predictions, axis=1)

            # "signal"...
            predict_signal = np.array(y_predictions)[:, -1]

        self.output()["confusion_matrix_png"].parent.touch()

        # Roc curve, compare labels and predicted labels
        fpr, tpr, tresholds = sk.metrics.roc_curve(y_test[:, -1], predict_signal)

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
        hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=10)
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
        fig = plt.figure()  # figsize=(12, 9)
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
                    fontsize=16,
                )
        ticks = np.arange(0, n_processes, 1)
        # Let the horizontal axes labeling appear on bottom
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(all_processes, fontsize=14)
        ax.set_yticklabels(all_processes, fontsize=14)
        ax.set_xlabel("Predicted Processes", fontsize=16)
        ax.set_ylabel("Real Processes", fontsize=16)
        hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=14, ax=ax)
        # ax.grid(linestyle="--", alpha=0.5)
        plt.savefig(self.output()["confusion_matrix_png"].path, bbox_inches="tight")
        plt.savefig(self.output()["confusion_matrix_pdf"].path)  # , bbox_inches="tight")
        plt.gcf().clear()


class DNNEvaluationCrossValPlotting(DNNTask):
    normalize = luigi.Parameter(default="true", description="if confusion matrix gets normalized")
    kfold = luigi.IntParameter(default=2)

    def requires(self):
        return {
            "data": CrossValidationPrep.req(self, kfold=self.kfold),
            "model": PytorchCrossVal.req(
                self,
                n_layers=self.n_layers,
                n_nodes=self.n_nodes,
                dropout=self.dropout,
                kfold=self.kfold,
                debug=False,
            ),
        }
        # return DNNTrainer.req(
        #    self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout
        # )

    def output(self):
        return {
            "fold_{}".format(i): {
                "ROC_png": self.local_target("fold_{}_pytorch_ROC.png".format(i)),
                "confusion_matrix_png": self.local_target("fold_{}_pytorch_confusion_matrix.png".format(i)),
                "ROC_pdf": self.local_target("fold_{}_pytorch_ROC.pdf".format(i)),
                "confusion_matrix_pdf": self.local_target("fold_{}_pytorch_confusion_matrix.pdf".format(i)),
                "loss_png": self.local_target("fold_{}_pytorch_loss.png".format(i)),
                "accuracy_png": self.local_target("fold_{}_pytorch_accuracy.png".format(i)),
                "loss_pdf": self.local_target("fold_{}_pytorch_loss.pdf".format(i)),
                "accuracy_pdf": self.local_target("fold_{}_pytorch_accuracy.pdf".format(i)),
            }
            for i in range(self.kfold)
        }

    def store_parts(self):
        # make plots for each use case
        return (
            super(DNNEvaluationCrossValPlotting, self).store_parts()
            # + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (self.gamma,)
        )

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        # from IPython import embed;embed()

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        all_processes = list(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        # all_processes[-1] = "Signal"

        for i in range(self.kfold):
            print("fold", i)
            # switch around in 2 point k fold
            j = abs(i - 1)
            path = self.input()["model"]["collection"].targets[0]["fold_" + str(i)]["model"].path

            # load complete model
            reconstructed_model = torch.load(path)

            # load all the prepared data thingies
            inp_data = self.input()["data"]["cross_val_" + str(j)]
            X_test = np.concatenate([inp_data["cross_val_X_train_" + str(j)].load(), inp_data["cross_val_X_val_" + str(j)].load()])
            y_test = np.concatenate([inp_data["cross_val_y_train_" + str(j)].load(), inp_data["cross_val_y_val_" + str(j)].load()])
            weight_test = np.concatenate([inp_data["cross_val_weight_train_" + str(j)].load(), inp_data["cross_val_weight_val_" + str(j)].load()])

            with torch.no_grad():
                reconstructed_model.eval()
                # for X_pred_batch, y_pred_batch, weight_pred_batch in pred_loader:
                y_predictions = reconstructed_model(torch.from_numpy(X_test)).softmax(dim=1)

            test_predictions = np.argmax(y_predictions, axis=1)

            # "signal"...
            predict_signal = np.array(y_predictions)[:, -1]
            self.output()["fold_" + str(i)]["confusion_matrix_png"].parent.touch()

            # Roc curve, compare labels and predicted labels
            fpr, tpr, tresholds = sk.metrics.roc_curve(y_test[:, -1], predict_signal)

            plt.plot(
                fpr,
                tpr,
                label="AUC: {0}".format(np.around(sk.metrics.auc(fpr, tpr), decimals=3)),
            )
            plt.plot([0, 1], [0, 1], ls="--")
            plt.xlabel("fpr", fontsize=16)
            plt.ylabel("tpr", fontsize=16)
            # plt.title("ROC", fontsize=16)
            plt.legend(title="ROC")
            hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=10)
            plt.savefig(self.output()["fold_" + str(i)]["ROC_png"].path, bbox_inches="tight")
            plt.savefig(self.output()["fold_" + str(i)]["ROC_pdf"].path, bbox_inches="tight")
            plt.gcf().clear()

            # Correlation Matrix Plot
            # plot correlation matrix
            pred_matrix = sk.metrics.confusion_matrix(np.argmax(y_test, axis=-1), test_predictions, normalize=self.normalize, sample_weight=weight_test)  # np.concatenate(test_predictions),
            print(pred_matrix)
            # TODO
            fig = plt.figure(figsize=(10, 8))  # figsize=(12, 9)
            ax = fig.add_subplot(111)
            # cax = ax.matshow(pred_matrix, vmin=-1, vmax=1)
            cax = ax.imshow(pred_matrix, vmin=0, vmax=1, cmap="plasma")
            fig.colorbar(cax)
            for ii in range(n_processes):
                for jj in range(n_processes):
                    text = ax.text(
                        jj,
                        ii,
                        np.round(pred_matrix[ii, jj], 3),
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
            ax.set_xticklabels(all_processes, fontsize=14)
            ax.set_yticklabels(all_processes, fontsize=14)
            ax.set_xlabel("Predicted Processes (Incl. Event Weight)", fontsize=18)
            ax.set_ylabel("Original Processes (Normed)", fontsize=18)
            hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=18, ax=ax)
            # ax.grid(linestyle="--", alpha=0.5)
            plt.savefig(self.output()["fold_" + str(i)]["confusion_matrix_png"].path, bbox_inches="tight")
            plt.savefig(self.output()["fold_" + str(i)]["confusion_matrix_pdf"].path, bbox_inches="tight")
            plt.gcf().clear()

            # retrieve history callback for trainings and do history plotting
            performance = self.input()["model"]["collection"].targets[0]["fold_" + str(i)]["performance"].load()
            accuracy_stats = performance["accuracy_stats"]
            loss_stats = performance["loss_stats"]
            # read in values, skip first for val since Trainer does a validation step beforehand
            train_loss = loss_stats["train"]
            val_loss = loss_stats["val"]

            train_acc = accuracy_stats["train"]
            val_acc = accuracy_stats["val"]

            plt.plot(np.arange(0, len(val_loss), 1), val_loss, label="loss on valid data", color="orange", linewidth=3)
            plt.plot(np.arange(1, len(train_loss) + 1, 1), train_loss, label="loss on train data", color="green", linewidth=3)
            plt.tick_params(axis="both", which="major", labelsize=16)
            plt.legend(fontsize=18)
            plt.xlabel("Epochs", fontsize=18)
            plt.ylabel("Loss", fontsize=18)
            hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=18)
            plt.savefig(self.output()["fold_" + str(i)]["loss_png"].path)
            plt.savefig(self.output()["fold_" + str(i)]["loss_pdf"].path)
            plt.gcf().clear()

            plt.plot(np.arange(0, len(val_acc), 1), val_acc, label="acc on vali data", color="orange", linewidth=3)
            plt.plot(np.arange(1, len(train_acc) + 1, 1), train_acc, label="acc on train data", color="green", linewidth=3)
            plt.tick_params(axis="both", which="major", labelsize=16)
            plt.legend(fontsize=18)
            plt.xlabel("Epochs", fontsize=18)
            plt.ylabel("Accuracy", fontsize=18)
            hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=18)
            plt.savefig(self.output()["fold_" + str(i)]["accuracy_png"].path)
            plt.savefig(self.output()["fold_" + str(i)]["accuracy_pdf"].path)
            plt.gcf().clear()


class DNNScorePlotting(DNNTask):
    unblinded = luigi.BoolParameter(default=False)
    density = luigi.BoolParameter(default=False)
    unweighted = luigi.BoolParameter(default=False)

    def requires(self):
        return {"scores": PredictDNNScores.req(self, workflow="local"), "norm": CalcNormFactors.req(self)}
        # return ConstructInferenceBins.req(self)

    def output(self):
        out = {p + "_" + end: self.local_target(p + "." + end) for p in self.config_inst.get_aux("DNN_process_template")[self.category].keys() for end in ["png", "pdf"]}
        new_out = {p + "_argmax_" + end: self.local_target(p + "argmax." + end) for p in self.config_inst.get_aux("DNN_process_template")[self.category].keys() for end in ["png", "pdf"]}
        out.update(new_out)
        return out

    def store_parts(self):
        # make plots for each use case
        parts = tuple()
        if self.unblinded:
            parts += ("unblinded",)
        if self.unweighted:
            parts += ("unweighted",)
        if self.density:
            parts += ("density",)
        return super(DNNScorePlotting, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + parts

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        inp = self.input()["scores"]["collection"].targets[0]
        MC_scores = inp["scores"].load()
        MC_labels = inp["labels"].load()
        weights = inp["weights"].load()
        # alpha * tt + beta * Wjets
        norm = self.input()["norm"].load()
        print("Norm", norm)
        if self.unweighted:
            weights = np.ones_like(weights)
        data_scores = inp["data"].load()
        # collecting scores for respective process
        scores_dict = {}
        for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            scores_dict[key] = MC_scores[MC_labels[:, i] == 1]
            scores_dict[key + "_weight"] = weights[MC_labels[:, i] == 1]
        scores_dict.update({"data": data_scores})
        # FIXME signal as single line, not in stack
        signal_node = False
        # one plot per per output note

        for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                signal_node = True  # FIXME True
            else:
                signal_node = False
            if self.unblinded:
                fig, (ax, rax) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0})
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
            hep.style.use("CMS")
            hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
            hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)
            MC_hists = {}
            signal_dict = {}
            for proc in scores_dict.keys():
                if "weight" in proc:
                    continue
                # without mask, we would be printing complete distribution of DNN scores per node
                mask = np.argmax(scores_dict[proc], axis=1) == i

                if proc != "data" and not self.config_inst.get_aux("signal_process").replace("V", "W") in proc:
                    # constructing hist and filling it with scores
                    if not signal_node:
                        boost_hist = bh.Histogram(self.construct_axis((100, 0, 1)))
                    if signal_node:
                        boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    # assign norm factors below
                    factor = 1
                    # if proc == "ttjets":
                    #     factor = norm["alpha"]
                    # if proc == "Wjets":
                    #     factor = norm["beta"]
                    # if proc == "QCD":
                    #     factor = norm["delta"]
                    print(factor)
                    boost_hist.fill(scores_dict[proc][mask][:, i], weight=factor * scores_dict[proc + "_weight"][mask])
                    MC_hists[proc] = boost_hist

                elif proc == "data" and self.unblinded:
                    # doing data seperate to print on top
                    if not signal_node:
                        data_boost_hist = bh.Histogram(self.construct_axis((100, 0, 1)))
                    if signal_node:
                        data_boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    data_boost_hist.fill(scores_dict[proc][mask][:, i])
                elif self.config_inst.get_aux("signal_process").replace("V", "W") in proc:
                    # signal as line
                    if not signal_node:
                        signal_hist = bh.Histogram(self.construct_axis((100, 0, 1)))
                    if signal_node:
                        signal_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    signal_hist.fill(scores_dict[proc][mask][:, i], weight=scores_dict[proc + "_weight"][mask])
                    # signal_dict[proc] = signal_hist
            hep.histplot(list(MC_hists.values()), histtype="fill", stack=True, label=list(MC_hists.keys()), color=["blue", "orange"], flow="none", density=self.density, ax=ax)
            if self.unblinded:
                yerr = np.where(data_boost_hist.values() > 0, np.sqrt(data_boost_hist), 0)
                hep.histplot(data_boost_hist, histtype="errorbar", yerr=yerr, label="Data", color="black", flow="none", density=self.density, ax=ax)
            try:
                hep.histplot(signal_hist, histtype="step", label=self.config_inst.get_aux("signal_process").replace("V", "W"), color="red", flow="none", density=self.density, ax=ax)
            except:
                signal_hist = bh.Histogram(self.construct_axis((100, 0, 1)))
                signal_hist.fill([0], weight=[0])
            ax.set_ylabel("Counts", fontsize=24)
            ax.set_xlim(0.25, 1)
            ax.set_yscale("log")
            ax.set_ylim(2e-1, 2e4)
            ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0, prop={"size": 18})
            ax.tick_params(axis="both", which="major", labelsize=24)

            if self.density:
                ax.set_ylim(5e-2, 2e1)
            if self.unblinded:
                MC = sum(list(MC_hists.values()))
                data = data_boost_hist
                ratio = data / MC
                stat_unc = np.sqrt(ratio * (ratio / MC + ratio / data))
                rax.axhline(1.0, color="black", linestyle="--")
                hep.histplot(ratio, color="black", histtype="errorbar", stack=False, yerr=stat_unc, ax=rax)
                rax.set_xlabel("DNN Scores in node " + key, fontsize=24)
                rax.set_xlim(0.25, 1)
                rax.set_ylabel("Data/MC", fontsize=24)
                rax.set_ylim(0.5, 1.5)
                rax.tick_params(axis="both", which="major", labelsize=24)
            else:
                ax.set_xlabel("DNN Scores in node " + key, fontsize=24)

            # # doing significance
            # bg = sum(MC_hists.values())
            # sig = signal_hist / np.sqrt(sum(MC_hists.values()))
            # sig_err = np.sqrt(np.power(1 / np.sqrt(bg) * np.sqrt(signal_hist), 2) + np.power(signal_hist / (2 * np.power(bg, 1.5)) * np.sqrt(bg), 2))
            # hep.histplot(sig, color="red", histtype="step", stack=False, yerr=sig_err, ax=sig_ax, label="Significance")

            # sig_ax.set_xlabel("DNN Scores in node " + key, fontsize=24)
            # sig_ax.set_ylabel("$signal / \sqrt{bg}$", fontsize=24)
            # sig_ax.set_xlim(0, 1)
            # sig_ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0, prop={"size": 18})

            self.output()[key + "_png"].parent.touch()
            plt.savefig(self.output()[key + "_argmax_png"].path, bbox_inches="tight")
            plt.savefig(self.output()[key + "_argmax_pdf"].path, bbox_inches="tight")
            plt.gcf().clear()

            # hist all scores in column
            scores_i = MC_scores[:, i]
            fig = plt.figure(figsize=(12, 9))
            hep.style.use("CMS")
            hep.cms.text("Private work (CMS simulation)", loc=0)
            hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$")
            plt.hist(scores_i, bins=np.arange(0, 1, 0.01))
            plt.xlabel("All DNN Scores in node " + key, fontsize=24)
            plt.ylabel("Counts", fontsize=24)
            plt.xlim(0.25, 1)
            plt.yscale("log")
            plt.savefig(self.output()[key + "_png"].path.replace(".png", "all_scores.png"), bbox_inches="tight")
            plt.gcf().clear()

        ################# plots with norm###############without argmax
        print("argmax also for norm plots")
        for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                signal_node = True  # FIXME True
            else:
                signal_node = False
            if self.unblinded:
                fig, (ax, rax) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0})
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
            hep.style.use("CMS")
            hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
            hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)
            MC_hists = {}
            signal_dict = {}
            for proc in scores_dict.keys():
                if "weight" in proc:
                    continue
                mask = np.argmax(scores_dict[proc], axis=1) == i
                if proc != "data" and not self.config_inst.get_aux("signal_process").replace("V", "W") in proc:
                    # constructing hist and filling it with scores
                    if not signal_node:
                        boost_hist = bh.Histogram(self.construct_axis((100, 0, 1)))
                    if signal_node:
                        boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    # assign norm factors
                    factor = 1
                    if proc == "ttjets":
                        factor = norm["alpha"]
                    if proc == "Wjets":
                        factor = norm["beta"]
                    if proc == "QCD":
                        factor = norm["delta"]
                    print(factor)
                    boost_hist.fill(scores_dict[proc][mask][:, i], weight=factor * scores_dict[proc + "_weight"][mask])
                    # assign legend labels and hists at same time using dict
                    MC_hists[proc + " *{}".format(np.round(factor, 3))] = boost_hist

                elif proc == "data" and self.unblinded:
                    # doing data seperate to print on top
                    if not signal_node:
                        data_boost_hist = bh.Histogram(self.construct_axis((100, 0, 1)))
                    if signal_node:
                        data_boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    data_boost_hist.fill(scores_dict[proc][mask][:, i])
                elif self.config_inst.get_aux("signal_process").replace("V", "W") in proc:
                    #  signal as line
                    if not signal_node:
                        signal_hist = bh.Histogram(self.construct_axis((100, 0, 1)))
                    if signal_node:
                        signal_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    signal_hist.fill(scores_dict[proc][mask][:, i], weight=scores_dict[proc + "_weight"][mask])

            hep.histplot(list(MC_hists.values()), histtype="fill", stack=True, label=list(MC_hists.keys()), color=["blue", "orange"], flow="none", density=self.density, ax=ax)
            if self.unblinded:
                yerr = np.where(data_boost_hist.values() > 0, np.sqrt(data_boost_hist), 0)
                hep.histplot(data_boost_hist, histtype="errorbar", yerr=yerr, label="Data", color="black", flow="none", density=self.density, ax=ax)
            hep.histplot(signal_hist, histtype="step", label=self.config_inst.get_aux("signal_process").replace("V", "W"), color="red", flow="none", density=self.density, ax=ax)

            ax.set_ylabel("Counts", fontsize=24)
            ax.set_xlim(0.25, 1)
            ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0, prop={"size": 18})
            ax.tick_params(axis="both", which="major", labelsize=24)

            if self.unblinded:
                MC = sum(list(MC_hists.values()))
                data = data_boost_hist
                ratio = data / MC
                stat_unc = np.sqrt(ratio * (ratio / MC + ratio / data))
                rax.axhline(1.0, color="black", linestyle="--")
                hep.histplot(ratio, color="black", histtype="errorbar", stack=False, yerr=stat_unc, ax=rax)
                rax.set_xlabel("DNN Scores in node " + key, fontsize=24)
                rax.set_xlim(0.25, 1)
                rax.set_ylabel("Data/MC", fontsize=24)
                rax.set_ylim(0.5, 1.5)
                rax.tick_params(axis="both", which="major", labelsize=24)
            else:
                ax.set_xlabel("DNN Scores in node " + key, fontsize=24)

            self.output()[key + "_png"].parent.touch()
            plt.savefig(self.output()[key + "_png"].path, bbox_inches="tight")
            plt.savefig(self.output()[key + "_pdf"].path, bbox_inches="tight")

            ax.set_yscale("log")
            if self.density:
                ax.set_ylim(5e-2, 2e1)
            else:
                ax.set_ylim(2e-1, 2e4)
            plt.savefig(self.output()[key + "_png"].path.replace(".png", "_log.png"), bbox_inches="tight")
            plt.savefig(self.output()[key + "_pdf"].path.replace(".pdf", "_log.pdf"), bbox_inches="tight")

            plt.gcf().clear()


class DNNScorePerProcess(CoffeaTask, DNNTask):
    unblinded = luigi.BoolParameter(default=False)
    signal = luigi.BoolParameter(default=False)
    density = luigi.BoolParameter(default=False)
    norm = luigi.BoolParameter(default=False)
    fine_binning = luigi.BoolParameter(default=False)
    QCD_estimate = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "merged": MergeArrays.req(self),
            "norm": CalcNormFactors.req(self, QCD_estimate=self.QCD_estimate),
            "QCD_pred": EstimateQCDinSR.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "QCD", "Rare", "DY", "MET", "SingleMuon", "SingleElectron"]),
            "model": PytorchCrossVal.req(
                self,
                n_layers=self.n_layers,
                n_nodes=self.n_nodes,
                dropout=self.dropout,
                kfold=self.kfold,
                debug=False,
                category="All_Lep",  # FIXME, this is trained on inclusive cuts
            ),
        }

    def output(self):
        return self.local_target("DNNScores.png")

    def store_parts(self):
        # make plots for each use case
        parts = tuple()
        if self.unblinded:
            parts += ("unblinded",)
        if self.signal:
            parts += ("signal",)
        if self.norm:
            parts += ("norm",)
        if self.density:
            parts += ("density",)
        if self.fine_binning:
            parts += ("fine_binning",)
        if self.QCD_estimate:
            parts += ("QCD_estimate",)
        return super(DNNScorePerProcess, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + parts

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        if self.QCD_estimate and self.fine_binning:
            raise Exception("\nQCD estimate is not fine binned...\n")
        inp = self.input()["merged"]
        norm_factors = self.input()["norm"].load()
        proc_factor_dict = {"ttjets": "alpha", "Wjets": "beta", "QCD": "delta"}
        # loading all models
        models = self.input()["model"]["collection"].targets[0]
        print("loading models")
        models_loaded = {fold: torch.load(models["fold_" + str(fold)]["model"].path) for fold in range(self.kfold)}

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        if self.unblinded:
            fig, (ax, rax) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0})
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
        hep.style.use("CMS")
        hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
        hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)
        # save histograms for ratio computing
        hists = {"MC": {}, "signal": {}, "data": {}}
        for dat in tqdm(self.datasets_to_process):
            if self.QCD_estimate and dat == "QCD":
                continue
            # print(dat)
            # accessing the input and unpacking the condor submission structure
            cat = self.category
            sumOfHists = []
            arr = inp[self.category + "_" + dat]["array"].load()
            weights = inp[self.category + "_" + dat]["weights"].load()
            DNNId = inp[self.category + "_" + dat]["DNNId"].load()
            SR_scores, SR_weights = [], []
            for fold, Id in enumerate(self.config_inst.get_aux("DNNId")):
                # to get respective switched id per fold
                j = -1 * Id
                # load complete model
                reconstructed_model = models_loaded[fold]

                # load all the prepared data thingies
                X_test = torch.tensor(arr[DNNId == j])
                weight_test = weights[DNNId == j]

                with torch.no_grad():
                    reconstructed_model.eval()
                    y_predictions = reconstructed_model(X_test).softmax(dim=1).numpy()
                mask = np.argmax(y_predictions, axis=-1) == (n_processes - 1)
                SR_scores.append(y_predictions[mask][:, -1])
                SR_weights.append(weight_test[mask])

            scores = np.concatenate(SR_scores)
            weights = np.concatenate(SR_weights)
            proc = self.config_inst.get_process(dat)
            boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
            if self.fine_binning:
                boost_hist = bh.Histogram(self.construct_axis([100, 0, 1]))
            # # We blind here!!!
            print("\nNot blinded!!!\n")
            # if proc.aux["isData"] and self.unblinded:
            #     weights[scores > 0.8] = 0  # FIXME
            boost_hist.fill(scores, weight=weights)

            if proc.aux["isData"] and self.unblinded:
                hists["data"][dat] = boost_hist
            elif proc.aux["isSignal"] and self.signal:
                hists["signal"][dat] = {
                    "hist": boost_hist,
                    "label": proc.label,
                    "color": proc.color,
                }
            elif not proc.aux["isData"] and not proc.aux["isSignal"]:
                # MC categorisation by exclusion, apply norm factors here, if we want
                factor = 1.0
                if self.norm:
                    for factor_key in self.config_inst.get_aux("DNN_process_template")[self.category].keys():
                        if dat in self.config_inst.get_aux("DNN_process_template")[self.category][factor_key]:
                            factor = norm_factors[proc_factor_dict[factor_key]]
                hists["MC"][dat] = {
                    "hist": boost_hist * factor,
                    "label": proc.label,
                    "color": proc.color,
                }

        # hand filling custom made QCD prediction
        if self.QCD_estimate:
            QCD_yields = self.input()["QCD_pred"].load()["SR_prediction"]
            boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
            boost_hist.fill(self.config_inst.get_aux("signal_binning")[:-1], weight=QCD_yields)
            proc = self.config_inst.get_process("QCD")
            hists["MC"]["QCD"] = {
                "hist": boost_hist,
                "label": proc.label + " data driven",
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

        if self.unblinded:
            data_hist = sum(list(hists["data"].values()))
            MC_hist = sum(hist_list)
            yerr = np.where(data_hist.values() > 0, np.sqrt(data_hist), 0)
            hep.histplot(sum(list(hists["data"].values())), histtype="errorbar", yerr=yerr, label="data (last bins blinded)", color="black", ax=ax)
            ratio = data_hist / MC_hist
            stat_unc = np.sqrt(ratio * (ratio / MC_hist + ratio / data_hist))
            rax.axhline(1.0, color="black", linestyle="--")
            # rax.fill_between(ratio.axes[0].centers, 1 - 0.023, 1 + 0.023, alpha=0.3, facecolor="black")
            hep.histplot(ratio, color="black", histtype="errorbar", stack=False, yerr=stat_unc, ax=rax)
            rax.set_xlabel("DNN Scores in Signal node {}".format(self.category), fontsize=24)
            rax.set_xlim(0, 1)
            rax.set_ylabel("Data/MC", fontsize=24)
            rax.set_ylim(0.5, 1.5)
            rax.tick_params(axis="both", which="major", labelsize=24)
        else:
            ax.set_xlabel("DNN Scores in Signal node {}".format(self.category), fontsize=24)

        if self.signal:
            for key in hists["signal"].keys():
                hep.histplot(hists["signal"][key]["hist"], histtype="step", label=hists["signal"][key]["label"], color=hists["signal"][key]["color"], ax=ax)

        ax.set_ylabel("Counts", fontsize=24)
        ax.set_xlim(0.25, 1)
        if self.norm:
            ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0, prop={"size": 16}, title="MC bg norm factors")
        else:
            ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0, prop={"size": 16})

        self.output().parent.touch()
        ax.set_yscale("linear")
        ax.tick_params(axis="both", which="major", labelsize=24)
        plt.savefig(self.output().path, bbox_inches="tight")
        plt.savefig(self.output().path.replace(".png", ".pdf"), bbox_inches="tight")
        ax.set_yscale("log")
        locmaj = tick.LogLocator(base=10, numticks=10)
        ax.yaxis.set_major_locator(locmaj)
        locmin = tick.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=10)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(tick.NullFormatter())
        ax.tick_params(right=True, top=False, labelright=True, labeltop=False, labelrotation=0, labelsize=24)

        ax.set_ylim(5e-2, 2e5)
        plt.savefig(self.output().path.replace(".png", "_log.png"), bbox_inches="tight")
        plt.savefig(self.output().path.replace(".png", "_log.pdf"), bbox_inches="tight")


class QCDComparison(CoffeaTask, DNNTask):
    unblinded = luigi.BoolParameter(default=False)
    signal = luigi.BoolParameter(default=False)
    density = luigi.BoolParameter(default=False)
    norm = luigi.BoolParameter(default=False)
    fine_binning = luigi.BoolParameter(default=False)
    QCD_estimate = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "merged": MergeArrays.req(self, datasets_to_process=["QCD"]),
            "DNN_scores": PredictDNNScores.req(self, workflow="local"),
            "QCD_pred": EstimateQCDinSR.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "QCD", "Rare", "DY", "MET", "SingleMuon", "SingleElectron"]),
        }

    def output(self):
        return self.local_target("QCD_comp.png")

    def store_parts(self):
        # make plots for each use case
        return (
            super(QCDComparison, self).store_parts()
            # + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
        )

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        MC_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False), storage=bh.storage.Weight())
        QCD_scores = self.input()["DNN_scores"]["collection"].targets[0]["QCD_scores"].load()
        QCD_weights = self.input()["DNN_scores"]["collection"].targets[0]["QCD_weights"].load()
        QCD_in_signal = np.argmax(QCD_scores, axis=-1) == 3
        MC_hist.fill(QCD_scores[QCD_in_signal][:, 3], weight=QCD_weights[QCD_in_signal])

        QCD_pred = self.input()["QCD_pred"].load()["SR_prediction"]
        QCD_err = self.input()["QCD_pred"].load()["SR_err"]
        pred_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
        pred_hist.fill(self.config_inst.get_aux("signal_binning")[:-1], weight=QCD_pred)

        for i, p in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            print("SR prediction in node ", p, ":", self.input()["QCD_pred"].load()["diff_data_EWK"][i] * self.input()["QCD_pred"].load()["F_Sel_Anti"])
            QCD_in_node = np.argmax(QCD_scores, axis=-1) == i
            print("MC sum in same node:", np.sum(QCD_weights[QCD_in_node]))

        fig = plt.figure()  # figsize=(12, 9)
        ax = fig.add_subplot(111)
        hep.cms.text("Private work (CMS simulation)", loc=0, fontsize=14, ax=ax)

        hep.histplot(pred_hist, histtype="errorbar", stack=False, yerr=QCD_err, ax=ax, label="Data driven QCD prediction")
        hep.histplot(MC_hist, histtype="errorbar", stack=False, yerr=np.sqrt(MC_hist.variances()), ax=ax, label="QCD MC")
        ax.set_ylabel("Counts", fontsize=18)
        ax.set_xlabel("DNN Score in Signal Node", fontsize=18)

        ax.legend()
        self.output().parent.touch()
        plt.savefig(self.output().path, bbox_inches="tight")
        plt.savefig(self.output().path.replace(".png", ".pdf"), bbox_inches="tight")
        ax.set_yscale("log")
        plt.savefig(self.output().path.replace(".png", "_log.png"), bbox_inches="tight")
        plt.savefig(self.output().path.replace(".png", "_log.pdf"), bbox_inches="tight")
        plt.gcf().clear()


class AllFittedBinsPlotting(CoffeaTask, DNNTask):
    unblinded = luigi.BoolParameter(default=False)
    signal = luigi.BoolParameter(default=False)
    norm = luigi.BoolParameter(default=False)
    QCD_estimate = luigi.BoolParameter(default=False)
    density = luigi.BoolParameter(default=False)
    unweighted = luigi.BoolParameter(default=False)
    split_signal = luigi.BoolParameter(default=False)
    do_shifts = luigi.BoolParameter(default=False)

    def requires(self):
        req = {
            "scores": PredictDNNScores.req(self, workflow="local", do_shifts=self.do_shifts),
            "norm": CalcNormFactors.req(self, QCD_estimate=self.QCD_estimate),
            "QCD_pred": EstimateQCDinSR.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "QCD", "Rare", "DY", "MET", "SingleMuon", "SingleElectron"]),
        }
        if self.do_shifts:
            req.update({"shifted_yields": GetShiftedYields.req(self)})
        return req

    def output(self):
        # out = {p + "_" + end: self.local_target(p + "." + end) for p in self.config_inst.get_aux("DNN_process_template")[self.category].keys() for end in ["png", "pdf"]}
        # new_out = {p + "_argmax_" + end: self.local_target(p + "argmax." + end) for p in self.config_inst.get_aux("DNN_process_template")[self.category].keys() for end in ["png", "pdf"]}
        # out.update(new_out)
        out = self.local_target("all_bins_{}.png".format("_".join(self.datasets_to_process)))
        return out

    def store_parts(self):
        # make plots for each use case
        parts = tuple()
        if self.do_shifts:
            parts += ("do_shifts",)
        if self.split_signal:
            parts += ("split_signal",)
        if self.unblinded:
            parts += ("unblinded",)
        if self.norm:
            parts += ("norm",)
        if self.QCD_estimate:
            parts += ("QCD_estimate",)
        return super(AllFittedBinsPlotting, self).store_parts() + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,) + parts

    def construct_axis(self, binning, isRegular=True):
        if isRegular:
            return bh.axis.Regular(binning[0], binning[1], binning[2])
        else:
            return bh.axis.Variable(binning)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        inp = self.input()["scores"]["collection"].targets[0]
        MC_scores = inp["scores"].load()
        MC_labels = inp["labels"].load()
        weights = inp["weights"].load()
        # data driven QCD
        QCD_estimate = self.input()["QCD_pred"].load()["SR_prediction"]
        QCD_bg_est = self.input()["QCD_pred"].load()["diff_data_EWK"]
        F_Sel_Anti = self.input()["QCD_pred"].load()["F_Sel_Anti"]
        # alpha * tt + beta * Wjets
        norm = self.input()["norm"].load()
        print("Norm", norm)
        if self.unweighted:
            weights = np.ones_like(weights)
        data_scores = inp["data"].load()
        # ### doing shifts for up and down
        # if self.do_shifts:
        #     syst_per_dat = {}
        #     shifts = inp["shifts"]
        #     print(shifts.keys())
        print("\nNot blinded!!!\n")
        # if self.unblinded:
        #     blind_mask = data_scores[:, -1] > 0.8
        #     data_scores[blind_mask] = 0  # FIXME blinding
        # collecting scores for respective process
        scores_dict = {}
        for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            scores_dict[key] = MC_scores[MC_labels[:, i] == 1]
            scores_dict[key + "_weight"] = weights[MC_labels[:, i] == 1]
        scores_dict.update({"data": data_scores})
        # FIXME signal as single line, not in stack
        signal_node = False
        all_hists = {"node_" + key: {} for key in self.config_inst.get_aux("DNN_process_template")[self.category].keys()}
        bin_names = []
        for i, key in enumerate(self.config_inst.get_aux("DNN_process_template")[self.category].keys()):
            if key == self.config_inst.get_aux("signal_process").replace("V", "W"):
                signal_node = True
            else:
                signal_node = False
            for proc in scores_dict.keys():
                if "weight" in proc:
                    continue
                # without mask, we would be printing complete distribution of DNN scores per node
                mask = np.argmax(scores_dict[proc], axis=1) == i
                if proc != "data" and not self.config_inst.get_aux("signal_process").replace("V", "W") in proc:
                    # constructing hist and filling it with scores
                    if not signal_node:
                        boost_hist = bh.Histogram(self.construct_axis((1, 0, 1)))
                    if signal_node:
                        boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    # assign norm factors
                    factor = 1
                    if self.norm:
                        if proc == "ttjets":
                            factor = norm["alpha"]
                        if proc == "Wjets":
                            factor = norm["beta"]
                        if proc == "QCD":
                            factor = norm["delta"]
                    boost_hist.fill(scores_dict[proc][:, i][mask], weight=factor * scores_dict[proc + "_weight"][mask])
                    # assign legend labels and hists at same time using dict
                    # MC_hists[proc + " *{}".format(np.round(factor, 3))] = boost_hist
                    if self.QCD_estimate and proc == "QCD":
                        if not signal_node:
                            boost_hist = bh.Histogram(self.construct_axis((1, 0, 1)))
                            boost_hist.fill([0], weight=QCD_bg_est[i] * F_Sel_Anti * factor)
                        if signal_node:
                            boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                            boost_hist.fill(self.config_inst.get_aux("signal_binning")[:-1], weight=QCD_estimate)

                    all_hists["node_" + key][proc] = boost_hist

                elif proc == "data" and self.unblinded:
                    # doing data seperate to print on top
                    if not signal_node:
                        data_boost_hist = bh.Histogram(self.construct_axis((1, 0, 1)))
                    if signal_node:
                        data_boost_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                    data_boost_hist.fill(scores_dict[proc][:, i][mask])
                    all_hists["node_" + key][proc] = data_boost_hist

                elif self.config_inst.get_aux("signal_process").replace("V", "W") in proc:
                    #  signal as line
                    if not signal_node:
                        signal_hist = bh.Histogram(self.construct_axis((1, 0, 1)))
                        bin_names.append("DNN Node " + key)
                    if signal_node:
                        signal_hist = bh.Histogram(self.construct_axis(self.config_inst.get_aux("signal_binning"), isRegular=False))
                        for b in self.config_inst.get_aux("signal_binning")[:-1]:
                            bin_names.append(key + "_" + str(b))
                    if self.split_signal:
                        for sig_dat in self.datasets_to_process:
                            if proc in sig_dat:
                                signal_hist.reset()
                                arr = self.input()["scores"]["collection"].targets[0][sig_dat].load()
                                mask_arr = np.argmax(arr, axis=1) == i
                                arr_weights = self.input()["scores"]["collection"].targets[0][sig_dat + "_weights"].load()
                                signal_hist.fill(arr[:, i][mask_arr], weight=arr_weights[mask_arr])
                                # to avoid reference getting copied
                                all_hists["node_" + key][sig_dat] = signal_hist.copy()
                    else:
                        signal_hist.fill(scores_dict[proc][:, i][mask], weight=scores_dict[proc + "_weight"][mask])
                        all_hists["node_" + key][proc] = signal_hist

        # we want to have on histogram per process group with entries for all relevant bins
        bin_count = 0
        nodes = list(all_hists.keys())
        # find process per node, so we have one bin per binning
        for node in nodes:
            for k in all_hists[node].keys():
                if node.split("_")[-1] in k:
                    bin_count += len(all_hists[node][k].counts())
                    # once per node
                    break
        bins = np.linspace(0, bin_count, bin_count + 1)
        clear_hists = {k: bh.Histogram(self.construct_axis((bin_count, 0, bin_count))) for k in all_hists[list(all_hists.keys())[0]].keys()}
        current_bin = 0
        for key in all_hists.keys():
            for proc in all_hists[key].keys():
                hist = all_hists[key][proc].counts()
                hist_len = len(hist)
                # now fill total counts of bin into respective combined hist bin
                weights = np.zeros(bin_count + 1)
                for i, count in enumerate(hist):
                    weights[current_bin + i] = count
                clear_hists[proc].fill(bins, weight=weights)
            current_bin += hist_len

        if self.do_shifts:
            syst_per_bin = []
            all_systs_json = self.input()["shifted_yields"].load()
            for bin in bin_names:
                syst_sum = 0
                for syst in all_systs_json:
                    for unc in all_systs_json[syst]:
                        # if "0.9335" in bin or "0.967" in bin:
                        #     print(bin.replace(" ", "_"), unc.replace("Score_", ""))
                        #         # print(bin, unc, all_systs_json[syst][unc])
                        if bin.replace(" ", "_") in unc.replace("Score_", "") and (unc.endswith("ttjets") or unc.endswith("Wjets")):
                            syst_sum += (all_systs_json[syst][unc] - 1) ** 2
                syst_per_bin.append(np.sqrt(syst_sum))

        # do one plot for all
        if self.unblinded:
            fig, (ax, rax) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0})
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
        # no idea why, but we have to allow rotation beforehand
        plt.xticks(rotation=70)
        hep.style.use("CMS")
        hep.cms.text("Private work (CMS simulation)", loc=0, ax=ax)
        hep.cms.lumitext(text=str(np.round(self.config_inst.get_aux("lumi") / 1000, 2)) + r"$fb^{-1}$", ax=ax)
        hist_list, label_list, color_list = [], [], []
        for dat in clear_hists.keys():
            proc = self.config_inst.get_process(dat.replace("ttjets", "TTbar").replace("Wjets", "WJets"))
            if dat == "data" and self.unblinded:
                hep.histplot(clear_hists[dat], label=proc.label, color=proc.color, histtype=proc.aux["histtype"], ax=ax, zorder=2)
            elif self.config_inst.get_aux("signal_process").replace("V", "W") in dat:
                hep.histplot(clear_hists[dat], label=proc.label, color=proc.color, histtype=proc.aux["histtype"], ax=ax, zorder=2)
            else:
                hist_list.append(clear_hists[dat])
                label = "DNN group " + dat
                if "QCD" in label and self.QCD_estimate:
                    label = "QCD data driven"
                if self.norm:
                    label += " * norm factor"
                label_list.append(label)
                color_list.append(proc.color)
        # so plotting on log scale is easier to read
        sum_of_hists = [h.sum() for h in hist_list]
        label_list = [h for _, h in sorted(zip(sum_of_hists, label_list))]
        hist_list = [h for _, h in sorted(zip(sum_of_hists, hist_list))]
        color_list = [h for _, h in sorted(zip(sum_of_hists, color_list))]

        # setting zorder so it is in background
        hep.histplot(hist_list, histtype="fill", stack=True, label=label_list, color=color_list, ax=ax, zorder=1)

        ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0, prop={"size": 18})
        ax.set_ylabel("Counts", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=24)
        # Set custom bin labels
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        if self.unblinded:
            data_hist = clear_hists["data"]
            MC_hist = sum(hist_list)
            ratio = data_hist / MC_hist
            stat_unc = np.sqrt(ratio * (ratio / MC_hist + ratio / data_hist))
            rax.axhline(1.0, color="black", linestyle="--")
            if self.do_shifts:
                print(syst_per_bin)
                # ensure last bin is plotted as well
                syst_per_bin.append(syst_per_bin[-1])
                centers = list(ratio.axes[0].centers)
                centers.append(ratio.axes[0].centers[-1] + 1)
                rax.fill_between(np.array(centers) - 0.5, 1 - np.array(syst_per_bin), 1 + np.array(syst_per_bin), alpha=0.3, facecolor="black", step="post")
            hep.histplot(ratio, color="black", histtype="errorbar", stack=False, yerr=stat_unc, ax=rax)
            rax.set_xlim(0, 1)
            rax.set_ylabel("Data/MC", fontsize=24)
            rax.set_ylim(0.5, 1.5)
            rax.set_xticks(bin_centers, bin_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=16)
            rax.tick_params(axis="both", which="major", labelsize=24)
            rax.set_xlim(0, 12)
        else:
            ax.set_xticks(bin_centers, bin_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=16)

        ax.set_xlim(0, 12)
        ax.set_yscale("log")
        ax.set_ylim(2e-2, 2e5)
        self.output().parent.touch()
        plt.savefig(self.output().path, bbox_inches="tight")  #
        plt.savefig(self.output().path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.gcf().clear()
        plt.close(fig)
