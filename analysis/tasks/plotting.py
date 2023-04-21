# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import boost_histogram as bh
import mplhep as hep
import coffea
from tqdm.auto import tqdm

# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask
from tasks.grouping import GroupCoffea, MergeArrays


class ArrayPlotting(CoffeaTask):
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    formats = luigi.ListParameter(default=["png", "pdf"])
    density = luigi.BoolParameter(default=False)
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
        # create dir
        print(var_names)
        for var in tqdm(self.config_inst.variables):
            # iterating over lepton keys
            for lep in self.input().keys():
                np_dict = self.input()[lep]["collection"].targets[0]
                if self.merged:
                    np_dict = self.input()[lep]
                for cat in self.config_inst.categories.names():
                    sumOfHists = []
                    fig, ax = plt.subplots(figsize=(12, 10))
                    # hep.style.use("CMS")
                    # hep.cms.label(
                    # label="Private Work",
                    # loc=0,
                    # ax=ax,
                    # )
                    hep.cms.text("Private work (CMS simulation)")
                    for dat in self.datasets_to_process:
                        # accessing the input and unpacking the condor submission structure
                        boost_hist = bh.Histogram(self.construct_axis(var.binning, True))
                        for key, value in np_dict.items():
                            if cat in key and dat in key:
                                boost_hist.fill(np.load(value.path)[:, var_names.index(var.name)])
                        if self.divide_by_binwidth:
                            boost_hist = boost_hist / np.prod(hist.axes.widths, axis=0)
                        if self.density:
                            boost_hist = self.get_density(boost_hist)

                        hep.histplot(boost_hist, label="{} {}: {}".format(lep, dat, boost_hist.sum()))
                        sumOfHists.append(-1 * boost_hist.sum())
                    # sorting the labels/handels of the plt hist by descending magnitude of integral
                    order = np.argsort((-1) * np.array(sumOfHists))
                    handles, labels = plt.gca().get_legend_handles_labels()
                    if not self.merged:
                        handles = [h for _, h in sorted(zip(sumOfHists, handles))]
                        labels = [l for _, l in sorted(zip(sumOfHists, labels))]
                    ax.legend(
                        handles,
                        labels,
                        ncol=1,
                        loc="upper left",
                        bbox_to_anchor=(1, 1),
                        borderaxespad=0,
                    )
                    ax.set_xlabel(var.get_full_x_title())
                    ax.set_ylabel(var.get_full_y_title())
                    for ending in self.formats:
                        outputKey = var.name + cat + lep + ending
                        if self.merged:
                            outputKey = var.name + cat + ending
                        self.output()[outputKey]["nominal"].parent.touch()
                        self.output()[outputKey]["log"].parent.touch()

                        ax.set_yscale("linear")
                        plt.savefig(self.output()[outputKey]["nominal"].path, bbox_inches="tight")

                        ax.set_yscale("log")
                        plt.savefig(self.output()[outputKey]["log"].path, bbox_inches="tight")
                    plt.gcf().clear()
                    plt.close(fig)


class CutflowPlotting(CoffeaTask):

    """
    Plotting cutflow produced by coffea
    Utility for doing log scale, only debug plotting
    """

    log_scale = luigi.BoolParameter()

    def requires(self):
        return GroupCoffea.req(self)

    def output(self):
        path = ""
        if self.log_scale:
            path += "_log"
        out = {dat: self.local_target(dat + "_cutflow{}.pdf".format(path)) for dat in self.datasets_to_process}
        return out

    def run(self):
        cutflow = self.input().load()
        categories = [c.name for c in cutflow.axis("category").identifiers()]
        # processes = [p.name for p in cutflow.axis("dataset").identifiers() if not "data" in p.name]
        for dat in self.datasets_to_process:
            fig, ax = plt.subplots(figsize=(12, 10))
            hep.cms.text("Private work (CMS simulation)")
            for i, category in enumerate(categories):
                ax = coffea.hist.plot1d(
                    cutflow[[dat], category, :].project("dataset", "cutflow"),
                    overlay="dataset",
                    # legend_opts={"labels":category}
                )
            n_cuts = len(self.config_inst.get_category(category).get_aux("cuts"))
            ax.set_xticks(np.arange(0.5, n_cuts + 1.5, 1))
            ax.set_xticklabels(["total"] + self.config_inst.get_category(category).get_aux("cuts"), rotation=80, fontsize=12)
            if self.log_scale:
                ax.set_yscale("log")
                ax.set_ylim(1e-8, 1e8)  # potential zero bins
            handles, labels = ax.get_legend_handles_labels()
            # from IPython import embed; embed()
            for i in range(len(labels)):
                labels[i] = labels[i] + " " + categories[i]
            ax.legend(handles, labels, title="Category: ", ncol=1, loc="best")
            self.output()[dat].parent.touch()
            plt.savefig(self.output()[dat].path, bbox_inches="tight")
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
