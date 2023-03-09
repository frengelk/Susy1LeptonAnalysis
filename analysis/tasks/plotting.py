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
from tqdm.auto import tqdm

# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask


class ArrayPlotting(CoffeaTask):
    channel = luigi.ListParameter(default=["Muon", "Electron"])
    formats = luigi.ListParameter(default=["png", "pdf"])
    density = luigi.BoolParameter(default=False)
    divide_by_binwidth = luigi.BoolParameter(default=False)

    def requires(self):
        # return CoffeaProcessor.req(self)
        return {
            sel: CoffeaProcessor.req(
                self,
                lepton_selection=sel,
                # workflow="local",
            )
            for sel in ["Electron", "Muon"]  # , "Electron"]
        }

    def output(self):
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
        return super(ArrayPlotting, self).store_parts() + (self.analysis_choice,)

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
                for cat in self.config_inst.categories.names():
                    sumOfHists = []
                    fig, ax = plt.subplots(figsize=(12, 10))
                    hep.style.use("CMS")
                    hep.cms.label(
                        label="Work in progress",
                        loc=0,
                        ax=ax,
                    )
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
                        self.output()[outputKey]["nominal"].parent.touch()
                        self.output()[outputKey]["log"].parent.touch()

                        ax.set_yscale("linear")
                        plt.savefig(self.output()[outputKey]["nominal"].path, bbox_inches="tight")

                        ax.set_yscale("log")
                        plt.savefig(self.output()[outputKey]["log"].path, bbox_inches="tight")
                    plt.gcf().clear()
                    plt.close
