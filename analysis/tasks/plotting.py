# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
from tqdm.auto import tqdm

# other modules
from tasks.coffea import CoffeaProcessor, CoffeaTask


class ArrayPlotting(CoffeaTask):
    density = luigi.BoolParameter(default=False)

    def requires(self):
        #return CoffeaProcessor.req(self)
        return {
            sel: CoffeaProcessor.req(
                self,
                lepton_selection=sel,
                #workflow="local",
            )
            for sel in ["Electron", "Muon"]  # , "Electron"]
        }

    def output(self):
        ending = ".png"
        if self.density:
            ending = "_density" + ending
        return {
            var: {  # cat + "_" +
                "nominal": self.local_target(var + ending),
                "log": self.local_target(var + "_log" + ending),
            }
            for var in self.config_inst.variables.names()
            # for cat in self.config_inst.categories.names()
        }

    def store_parts(self):
        return super(ArrayPlotting, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        #in_dict = self.input()  # ["collection"].targets
        # in_dict = self.input()["Muon"]["collection"].targets[0]
        # making clear which index belongs to which variable
        var_names = self.config_inst.variables.names()
        # create dir
        print(var_names)
        # for cat in in_dict.keys():
        # signal_events = 0
        for var in tqdm(self.config_inst.variables):
            # new canvas for each variable
            fig, ax = plt.subplots(figsize=(18, 10))
            # hep.set_style("CMS")
            hep.style.use("CMS")
            hep.cms.label(
                llabel="Work in progress",
                loc=0,
                ax=ax,
            )
            # iterate over the indices for each file
            feedback = []  # {}
            # ieterating over lepton keys
            for lep in self.input().keys():
                np_dict = self.input()[lep]["collection"].targets[0]
                # sort inputs by datasets
                for dat in self.datasets_to_process:
                    # different key for each file, we ignore it for now, only interested in values
                    # define empty hists
                    np_hist = np.array([])
                    # accessing the input and unpacking the condor submission structure
                    for key, value in np_dict.items():
                        cat = "N0b" # or loop over self.config_inst.categories.names()
                        if cat in key and dat in key:
                            np_0b = np.load(value.path)
                            np_hist = np.append(np_hist, np_0b[:, var_names.index(var.name)])
                    # integrate hist
                    bins = np.arange(
                        var.binning[1],
                        var.binning[2],
                        (var.binning[2] - var.binning[1]) / var.binning[0],
                    )
                    back = np.sum(np.histogram(np_hist, bins=bins)[0])
                    plt.hist(
                        np_hist,
                        bins=bins,
                        histtype="step",
                        label="{} {}: {}".format(lep, dat, back),
                        density=self.density,
                    )
                    # feedback.update({key:np.sum(back[0])})
                    feedback.append(back)
            # sorting the labels/handels of the plt hist by descending magnitude of integral
            order = np.argsort((-1) * np.array(feedback))
            # print(var.name, feedback)
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend(np.array(handles)[order], np.array(labels)[order])
            ax.set_xlabel(var.get_full_x_title())
            ax.set_ylabel(var.get_full_y_title())
            self.output()[var.name]["nominal"].parent.touch()
            plt.savefig(self.output()[var.name]["nominal"].path)
            ax.set_yscale("log")
            plt.savefig(self.output()[var.name]["log"].path)
            plt.gcf().clear()
