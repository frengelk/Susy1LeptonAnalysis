import os
import law
import uproot as up
import ROOT
import numpy as np
from luigi import Parameter, BoolParameter
from tasks.base import AnalysisTask, HTCondorWorkflow
import json
import awkward as ak
from tqdm import tqdm

"""
Tasks to write config for datasets from target directory
Then write a fileset directory as an input for coffea
"""


class BaseMakeFilesTask(AnalysisTask):
    # Basis class only for inheritance
    directory_path = Parameter(default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2023_01_18/2017/Data/root")
    skip_extra = BoolParameter(default=False)


class CheckFiles(BaseMakeFilesTask):
    def output(self):
        return self.local_target("datasets_{}.json".format(self.year))

    def run(self):
        lep = ["Muon", "Electron"]
        # lep = ["Events"]
        var = ["HT", "LT", "LeptonPt_1", "MuonPt", "ElectronPt"]
        for root, dirs, files in os.walk(self.directory_path):
            for directory in dirs:
                # print(directory)
                for r, d, f in os.walk(self.directory_path + "/" + directory):
                    for filename in f:
                        print("\n max values in file:", filename)
                        with up.open(self.directory_path + "/" + directory + "/" + filename) as file:
                            for l in lep:
                                for v in var:
                                    max_v = np.min(file[l][v].array())
                                    # print(filename, l, v, max_v)
                                    # mask = file[l]["nJet"].array() > 6
                                    # diff = file[l]["HT"].array() - file[l]["JetPt_1"].array() - file[l]["JetPt_2"].array()
                                    # argmin = np.argmin(diff)
                                    # print(filename, l, "HT-jetpt1-jetpt2", diff[argmin], "with njet", file[l]["nJet"].array()[argmin])
                                    if max_v > 1e4:
                                        print(v, "max in", l, max_v)


class WriteDatasets(BaseMakeFilesTask):
    def output(self):
        return self.local_target("datasets_{}.json".format(self.year))

    def store_parts(self):
        return super(WriteDatasets, self).store_parts()

    def run(self):
        self.output().parent.touch()
        print("\n skipping extra files for now \n")
        file_dict = {}
        for root, dirs, files in os.walk(self.directory_path):
            for directory in sorted(dirs):
                # print(directory)
                file_list = []
                for r, d, f in os.walk(self.directory_path + "/" + directory):
                    for file in sorted(f):
                        if self.skip_extra:
                            if "extra" in file:
                                print("skipping extra")
                                continue
                        file_list.append(directory + "/" + file)
                file_dict.update({directory: file_list})  # self.directory_path + "/" +

        # replacing signal mass points
        for dat in list(file_dict.keys()):
            if self.config_inst.get_aux("signal_process") in dat:
                for masspoint in self.config_inst.get_aux("signal_masspoints"):
                    new_dat = dat + "_{}_{}".format(masspoint[0], masspoint[1])
                    file_dict[new_dat] = file_dict[dat]
                # remove general signal
                # file_dict.pop(dat)

        file_dict.update({"directory_path": self.directory_path})  # self.directory_path + "/" +
        with open(self.output().path, "w") as out:
            json.dump(file_dict, out)

        # copy the json file directly to the aux part
        print("Writing job dict to:", self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + "_" + self.category + ".json"))
        os.system("cp {} {}".format(self.output().path, self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + "_" + self.category + ".json")))


class WriteDatasetPathDict(BaseMakeFilesTask):

    """
    if processing self skimmed files, the provided path should end in the root/ or merged/ dir
    """

    def requires(self):
        return WriteDatasets.req(self, directory_path=self.directory_path)

    def output(self):
        return {
            "dataset_dict": self.local_target("datasets_{}.json".format(self.year)),
            "dataset_path": self.local_target("path.json"),  # save this so you'll find the files
            "job_number_dict": self.local_target("job_number_dict.json"),
        }

    def store_parts(self):
        return super(WriteDatasetPathDict, self).store_parts()

    def run(self):
        self.output()["dataset_dict"].parent.touch()
        file_dict = {}
        for root, dirs, files in os.walk(self.directory_path):
            for directory in sorted(dirs):
                # print(directory)
                file_list = []
                for r, d, f in os.walk(self.directory_path + "/" + directory):
                    for file in sorted(f):
                        if self.skip_extra:
                            if "extra" in file:
                                print("skipping extra")
                                continue
                        file_list.append(directory + "/" + file)
                file_dict.update({directory: file_list})  # self.directory_path + "/" +
        assert len(file_dict.keys()) > 0, "No files found in {}".format(self.directory_path)
        job_number_dict = {}
        for k in sorted(file_dict.keys()):
            print(k, len(file_dict[k]))
            job_number_dict.update({k: len(file_dict[k])})
        """
        # replacing signal mass points
        for dat in list(file_dict.keys()):
            if self.config_inst.get_aux("signal_process") in dat:
                for masspoint in self.config_inst.get_aux("signal_masspoints"):
                    new_dat = dat + "_{}_{}".format(masspoint[0], masspoint[1])
                    file_dict[new_dat] = file_dict[dat]
                    job_number_dict[new_dat] = job_number_dict[dat]
                # remove general masspoint
                job_number_dict.pop(dat)
                file_dict.pop(dat)
        """
        self.output()["dataset_dict"].dump(file_dict)
        self.output()["dataset_path"].dump(self.directory_path)
        self.output()["job_number_dict"].dump(job_number_dict)


class WriteConfigData(BaseMakeFilesTask):
    def requires(self):
        return WriteDatasets.req(self)

    def output(self):
        return self.local_target("files_per_dataset_{}.py".format(self.year))

    def write_add_dataset(self, name, number, keys):
        return [
            "cfg.add_dataset(",
            "'{}',".format(name),
            "{},".format("1" + str(number)),
            "campaign=campaign,",
            "keys={})".format(keys),
        ]

    def run(self):
        with open(self.input().path, "r") as read_file:
            file_dict = json.load(read_file)

        self.output().parent.touch()

        with open(self.output().path, "w") as out:
            # out.write("import os")
            # out.write("\n" + "#####datasets#####" + "\n")
            # out.write("def setup_datasets(cfg, campaign):" + "\n")
            # for proc in self.config_inst.processes:
            # for child in proc.walk_processes():
            # child = child[0]
            # # print(child.name)
            # # from IPython import embed;embed()
            # # for sets in file_dict.keys():
            # if child.name in proc_dict.keys():
            # sets = []
            # for directory in proc_dict[child.name]:
            # sets.extend(file_dict[directory])
            # for line in self.write_add_dataset(child.name, child.id, sets):
            # out.write(line + "\n")

            for key in self.input().load().keys():
                out.write(key + ":" + str(self.input().load()[key]))


class WriteFileset(BaseMakeFilesTask):
    def requires(self):
        return WriteConfigData.req(self)

    def output(self):
        return self.local_target("fileset.json")

    def run(self):
        # make the output directory
        out = self.output().parent
        out.touch()

        # unchanged syntax
        # test_file = "/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"

        fileset = {}

        for dat in self.config_inst.datasets:
            fileset.update(
                {
                    dat.name: [self.directory_path + "/" + key for key in dat.keys],
                }
            )

        with open(self.output().path, "w") as file:
            json.dump(fileset, file)


class CollectMasspoints(BaseMakeFilesTask):
    def requires(self):
        return WriteDatasetPathDict.req(self)

    def output(self):
        return self.local_target("masspoints.json")

    # written by ChatGPT
    def find_filled_bins(self, hist):
        # Get the number of bins along X and Y axes
        n_bins_x = hist.GetNbinsX()
        n_bins_y = hist.GetNbinsY()

        masspoints = []

        # Iterate over the bins
        for i in range(1, n_bins_x + 1):
            for j in range(1, n_bins_y + 1):
                # Check the bin content
                bin_content = hist.GetBinContent(i, j)
                if bin_content != 0.0:
                    bin_center_x = hist.GetXaxis().GetBinCenter(i)
                    bin_center_y = hist.GetYaxis().GetBinCenter(j)
                    # print(f"Bin ({i}, {j}) with label ({bin_center_x:.2f}, {bin_center_y:.2f}) is filled with content: {bin_content}")
                    masspoints.append((bin_center_x, bin_center_y))
        return masspoints

    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        path = inp["dataset_path"].load()
        # if this process is not defined, you are using the wrong signal
        proc = self.config_inst.get_process("T5qqqqWW")
        sub_name = [b[0].name for b in proc.walk_processes()][0]
        signal_names = inp["dataset_dict"].load()[sub_name]
        mp = []
        for name in signal_names:
            file = ROOT.TFile(path + "/" + name)
            hist = file.Get("numberOfT5qqqqWWGenEvents;1")
            masspoints = self.find_filled_bins(hist)
            print("Found ", len(masspoints), " masspoints")
            if len(masspoints) > len(mp):
                mp = masspoints
        print("final length: ", len(mp))
        self.output().dump({"masspoints": masspoints})
        print("You should maybe do 2017 first if you see less than 861 masspoints")
        os.system("cp {} {}".format(self.output().path, self.config_inst.get_aux("masspoints")))


class CollectInputData(BaseMakeFilesTask):
    def requires(self):
        # return {
        # sel: CoffeaProcessor.req(
        # self,
        # lepton_selection=sel,
        # # workflow="local",
        # datasets_to_process=self.datasets_to_process,
        # )
        # for sel in self.channel[:1]
        # }
        return WriteDatasetPathDict.req(self)

    def output(self):
        # if not "SR" in self.category or "SR_Anti" in self.category:
        return {
            "sum_gen_weights": self.local_target("sum_gen_weights.json"),
            "cutflow": self.local_target("cutflow_dict.json"),
        }

    # else:
    #     return {
    #         "sum_gen_weights": self.local_target("sum_gen_weights.json"),
    #         "cutflow": self.local_target("cutflow_dict.json"),
    #     }

    def store_parts(self):
        return super(CollectInputData, self).store_parts()

    def load_masspoints(self):
        with open(self.config_inst.get_aux("masspoints")) as f:
            masspoints = json.load(f)
        ints = [[int(x) for x in lis] for lis in masspoints["masspoints"]]
        strs = [[str(x) for x in lis] for lis in ints]
        return ints, strs

    # written by ChatGPT
    def find_filled_bins(hist):
        # Get the number of bins along X and Y axes
        n_bins_x = hist.GetNbinsX()
        n_bins_y = hist.GetNbinsY()

        # Iterate over the bins
        for i in range(1, n_bins_x + 1):
            for j in range(1, n_bins_y + 1):
                # Check the bin content
                bin_content = hist.GetBinContent(i, j)
                if bin_content != 0.0:
                    bin_center_x = hist.GetXaxis().GetBinCenter(i)
                    bin_center_y = hist.GetYaxis().GetBinCenter(j)
                    print(f"Bin ({i}, {j}) with label ({bin_center_x:.2f}, {bin_center_y:.2f}) is filled with content: {bin_content}")

    @law.decorator.safe_output
    def run(self):
        sum_gen_weights_dict = {}
        # if we want to include other channels as well
        cutflow_dict = {
            "LeptonIncl": {},
            "Muon": {},
            "Electron": {},
        }
        # normalization seperate for each mass point on its own
        masspoints, str_masspoints = self.load_masspoints()
        masspoint_dict = {"_".join([hist, str(masspoint[0]), str(masspoint[1])]): 0 for hist in ["nGen", "ISR"] for masspoint in masspoints}
        # for masspoint in self.config_inst.get_aux("signal_masspoints")}
        # sum weights same for both trees, do it once
        inp = self.input()  # [self.channel[0]].collection.targets[0]
        data_path = inp["dataset_path"].load()
        dataset_dict = inp["dataset_dict"].load()
        for key in dataset_dict.keys():
            for path in dataset_dict[key]:
                with up.open(data_path + "/" + path) as file:
                    sum_gen_weight = float(np.sum(file["MetaData;1"]["SumGenWeight"].array()))
                    if self.config_inst.get_aux("signal_process") in path:
                        # for masspoint in self.config_inst.get_aux("signal_masspoints"):
                        for masspoint in masspoints:
                            mGlu, mLSP = masspoint[0], masspoint[1]
                            f_path = data_path + "/" + path
                            inFile = ROOT.TFile.Open(f_path, " READ ")
                            hist_nGen_2D = inFile.Get("numberOfT5qqqqWWGenEvents")
                            count_nGen = hist_nGen_2D.GetBinContent(int(hist_nGen_2D.GetXaxis().FindBin(mGlu)), int(hist_nGen_2D.GetYaxis().FindBin(mLSP)), 0)
                            hist_ISR_2D = inFile.Get("numberOfT5qqqqWWGenEventsIsrWeighted")
                            count_ISR = hist_ISR_2D.GetBinContent(int(hist_ISR_2D.GetXaxis().FindBin(mGlu)), int(hist_ISR_2D.GetYaxis().FindBin(mLSP)), 0)
                            # print(path, count)
                            # sum_nGen += count
                            masspoint_dict["_".join(["nGen", str(masspoint[0]), str(masspoint[1])])] += count_nGen
                            masspoint_dict["_".join(["ISR", str(masspoint[0]), str(masspoint[1])])] += count_ISR
                            new_key = key + "_{}_{}".format(mGlu, mLSP)
                            if new_key not in sum_gen_weights_dict.keys():
                                cutflow_dict["LeptonIncl"][key] = file["cutflow_LeptonIncl;1"].values()
                                # cutflow_dict["Electron"][key] = file['cutflow_Electron;1'].values()
                                sum_gen_weights_dict[new_key] = sum_gen_weight
                                # if "SR" in self.category and not "SR_Anti" in self.category:
                                lep_arr = file["cutflow_LeptonIncl;1"].values()
                                #     electron_arr = file["cutflow_Electron;1"].values()
                            elif new_key in sum_gen_weights_dict.keys():
                                sum_gen_weights_dict[new_key] += sum_gen_weight
                                # if "SR" in self.category and not "SR_Anti" in self.category:
                                #     muon_arr += file["cutflow_Muon;1"].values()
                                #     electron_arr += file["cutflow_Electron;1"].values()

                    # else:
                    # keys should do the same for both dicts
                    if key not in sum_gen_weights_dict.keys():
                        cutflow_dict["LeptonIncl"][key] = file["cutflow_LeptonIncl;1"].values()
                        # cutflow_dict["Electron"][key] = file['cutflow_Electron;1'].values()
                        # if "SR" in self.category and not "SR_Anti" in self.category:
                        lep_arr = file["cutflow_LeptonIncl;1"].values()
                        #     electron_arr = file["cutflow_Electron;1"].values()
                        sum_gen_weights_dict[key] = sum_gen_weight
                        print("bg", key)
                    elif key in sum_gen_weights_dict.keys():
                        sum_gen_weights_dict[key] += sum_gen_weight
                        # if "SR" in self.category and not "SR_Anti" in self.category:
                        lep_arr += file["cutflow_LeptonIncl;1"].values()
                        #     electron_arr += file["cutflow_Electron;1"].values()

            cutflow_dict["LeptonIncl"][key] = lep_arr[lep_arr > 0].tolist()
            #     cutflow_dict["Electron"][key] = electron_arr[electron_arr > 0].tolist()

            # that is our data
            if key in ["MET", "SingleMuon", "SingleElectron"]:
                sum_gen_weights_dict[key] = 1

            # apply /nGen to respective signal masspoint genWeight
            # sumgenWeight getting computed like normal, and then we need to normalize with nGen as well
            # for each signal masspoint, two fractions -> therefore two separate sums
            if self.config_inst.get_aux("signal_process") in key:
                # for masspoint in self.config_inst.get_aux("signal_masspoints"):
                for masspoint in masspoints:
                    new_key = key + "_{}_{}".format(masspoint[0], masspoint[1])
                    print(new_key)
                    nGen = masspoint_dict["_".join(["nGen", str(masspoint[0]), str(masspoint[1])])]
                    ISR = masspoint_dict["_".join(["ISR", str(masspoint[0]), str(masspoint[1])])]
                    sumGenWeights = sum_gen_weights_dict[new_key]
                    # we do it inverse, since it will be in the denominator later
                    weighting = ISR / nGen * nGen  # FIXME  sumGenWeights / nGen *
                    sum_gen_weights_dict[new_key] = weighting

        self.output()["sum_gen_weights"].dump(sum_gen_weights_dict)
        # if "SR" in self.category and not "SR_Anti" in self.category:
        self.output()["cutflow"].dump(cutflow_dict)


class CalcBTagSF(BaseMakeFilesTask, HTCondorWorkflow, law.LocalWorkflow):
    # require more ressources
    RAM = 2500
    hours = 14
    debug = BoolParameter(default=False)

    def requires(self):
        return WriteDatasets.req(self, category=self.category)

    def create_branch_map(self):
        # return list(range(len(self.T5_ids)))
        # define job number according to number of files of the dataset that you want to process
        data_list_keys, _ = self.get_datalist_items()
        if self.debug:
            data_list_keys = data_list_keys[:1]
        return list(range(len(data_list_keys)))

    def output(self):
        values = self.get_datalist_items()[1]
        out = {lep + "_" + val.split("/")[1]: {"weights": self.local_target(lep + "_" + val.split("/")[1].replace(".root", ".npy")), "up": self.local_target(lep + "_" + val.split("/")[1].replace(".root", "_up.npy")), "down": self.local_target(lep + "_" + val.split("/")[1].replace(".root", "_down.npy"))} for val in values for lep in self.config_inst.get_aux("channels")[self.category]}
        return out

    def store_parts(self):
        parts = tuple()
        if self.debug:
            parts += ("debug",)
        return super(CalcBTagSF, self).store_parts() + parts

    def get_datalist_items(self):
        with open(self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + "_" + self.category + ".json")) as f:
            data_list = json.load(f)
        # so we can pop in place
        keys = list(data_list.keys())
        for key in keys:
            # FIXME
            # if not "DYJetsToLL_M-50_HT-100to200" in key:
            # need to delete last entry, which is the path
            if key in self.config_inst.get_aux("data") or "directory_path" in key:
                data_list.pop(key)

        values = []
        data_list_values = map(values.extend, list(data_list.values()))  # [:-1])
        # executes the map somehow?
        list(data_list_values)
        return (list(data_list.keys()), list(values))  # [:-1]

    def get_bin_contents(self, hist, pt, eta):
        """
        Get the bin contents for given points (pt, eta) in the histogram.

        Parameters:
        hist (tuple): The histogram data consisting of (bin_contents, pt_bins, eta_bins).
        pt (list): List of arrays with pt values.
        eta (list): List of arrays with eta values.

        Returns:
        list: List of arrays containing bin contents for the specified (pt, eta) points.
        """
        bin_contents, pt_bins, eta_bins = hist

        # Calculate bin indices for each point
        i = np.digitize(pt, pt_bins) - 1
        j = np.digitize(eta, eta_bins) - 1

        # Ensure indices are within valid bin range
        i = np.clip(i, 0, bin_contents.shape[0] - 1)
        j = np.clip(j, 0, bin_contents.shape[1] - 1)

        # Retrieve bin contents for the points
        bin_contents_for_points = bin_contents[i, j]

        return bin_contents_for_points

    @law.decorator.safe_output
    def run(self):
        import matplotlib.pyplot as plt

        # if this process is not defined, you are using the wrong signal
        proc = self.config_inst.get_process("T5qqqqWW")
        sub_name = [b[0].name for b in proc.walk_processes()][0]

        data_list_keys, data_list_values = self.get_datalist_items()
        sum_gen_weights_dict = {}
        # cutflow_dict = {
        #     "Muon": {},
        #     "Electron": {},
        # }
        dataset_dict = self.input().load()  # [self.channel[0]].collection.targets[0]
        # remove path from dict and save it for later
        data_path = dataset_dict.pop("directory_path")
        key = data_list_keys[self.branch]
        print(key)
        true_counts_b, true_counts_c, true_counts_l = np.array([]), np.array([]), np.array([])
        tagged_counts_b, tagged_counts_c, tagged_counts_l = np.array([]), np.array([]), np.array([])
        for path in dataset_dict[key]:
            with up.open(data_path + "/" + path) as file:
                # following method from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
                TrueB = file["nTrueB;1"].to_numpy()
                nBTag = file["nMediumBbTagDeepJet;1"].to_numpy()
                TrueC = file["nTrueC;1"].to_numpy()
                nCTag = file["nMediumCbTagDeepJet;1"].to_numpy()
                TrueL = file["nTrueLight;1"].to_numpy()
                nLTag = file["nMediumLightbTagDeepJet;1"].to_numpy()
                # not filled yet, fill all at once
                if true_counts_b.shape == (0,):
                    true_counts_b = TrueB[0]
                    tagged_counts_b = nBTag[0]
                    true_counts_c = TrueC[0]
                    tagged_counts_c = nCTag[0]
                    true_counts_l = TrueL[0]
                    tagged_counts_l = nLTag[0]
                    pt_bins, eta_bins = TrueB[1], TrueB[2]
                else:
                    true_counts_b += TrueB[0]
                    tagged_counts_b += nBTag[0]
                    true_counts_c += TrueC[0]
                    tagged_counts_c += nCTag[0]
                    true_counts_l += TrueL[0]
                    tagged_counts_l += nLTag[0]

        # total efficiency per file
        efficiency_hist_b = tagged_counts_b / true_counts_b
        efficiency_hist_c = tagged_counts_c / true_counts_c
        efficiency_hist_l = tagged_counts_l / true_counts_l
        # special hardcoded case to do complete Btag SF for mass scan
        if key == sub_name:
            total_arr, total_arr_up, total_arr_down = [], [], []

        # FIXME
        for path in tqdm(sorted(dataset_dict[key])):  # , reverse=True
            print(path)
            with up.open(data_path + "/" + path) as file:
                # having to do each trees seperatly
                for lep in self.config_inst.get_aux("channels")[self.category]:
                    # tto semileptonic was branch 30
                    # if os.path.exists(self.output()[lep + "_" + path.split("/")[1]]["weights"].path):
                    #     print(self.output()[lep + "_" + path.split("/")[1]]["weights"].path, "exists")
                    #     continue
                    # print("doing", self.output()[lep + "_" + path.split("/")[1]]["weights"].path)
                    # following method from https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/
                    # if "QCD_HT200to300" in path or "QCD_HT200to300" in data_path:
                    #     from IPython import embed; embed()

                    tree = file[lep]
                    tagged = tree["JetDeepJetMediumId"].array()
                    SF = tree["JetDeepJetMediumSf"].array()
                    SF_up = tree["JetDeepJetMediumSf_uncorrelatedUp"].array()
                    SF_down = tree["JetDeepJetMediumSf_uncorrelatedDown"].array()
                    jetPt = tree["JetPt"].array()
                    jetEta = tree["JetEta"].array()
                    # additional fastsim SF have to be applied
                    if key == sub_name:
                        SF_fast = tree["JetDeepJetLooseFastSf"].array()
                        SF_fast_up = tree["JetDeepJetMediumFastSfUp"].array()
                        SF_fast_down = tree["JetDeepJetTightFastSfDown"].array()
                        # *= does not work
                        SF = SF * SF_fast
                        SF_up = SF_up * SF_fast_up
                        SF_down = SF_down * SF_fast_down
                    # effs=(efficiency, pt_bins, eta_bins)
                    # computing efficiencies, do it flat to avoid for loop in python
                    efficiencies_b = ak.Array(self.get_bin_contents((efficiency_hist_b, pt_bins, eta_bins), ak.flatten(jetPt), ak.flatten(jetEta)))
                    efficiencies_c = ak.Array(self.get_bin_contents((efficiency_hist_c, pt_bins, eta_bins), ak.flatten(jetPt), ak.flatten(jetEta)))
                    efficiencies_l = ak.Array(self.get_bin_contents((efficiency_hist_l, pt_bins, eta_bins), ak.flatten(jetPt), ak.flatten(jetEta)))
                    # split them back up, we know the amount of jets
                    nJet = tree["nJet"].array()
                    efficiencies_b = ak.Array(np.split(efficiencies_b, np.cumsum(nJet)))
                    efficiencies_c = ak.Array(np.split(efficiencies_c, np.cumsum(nJet)))
                    efficiencies_l = ak.Array(np.split(efficiencies_l, np.cumsum(nJet)))
                    if len(efficiencies_b) > len(tagged):
                        # cumsum may result in empty array at the end
                        efficiencies_b = efficiencies_b[:-1]
                        efficiencies_c = efficiencies_c[:-1]
                        efficiencies_l = efficiencies_l[:-1]

                    # now build together arrays to multiply per efficiency(flav), divide in flavor 5 b, 4 c, else light
                    # there may be a faster way to do it columnar, I did not find it
                    jetFlav = tree["JetPartFlav"].array()
                    b_flav = abs(jetFlav) == 5
                    c_flav = abs(jetFlav) == 4
                    l_flav = (abs(jetFlav) != 5) & (abs(jetFlav) != 4)
                    efficiencies = ak.concatenate((efficiencies_b[b_flav], efficiencies_c[c_flav], efficiencies_l[l_flav]), axis=-1)

                    P_MC = np.prod(efficiencies[tagged], axis=-1) * np.prod((1 - efficiencies)[~tagged], axis=-1)
                    P_data = np.prod((SF * efficiencies)[tagged], axis=-1) * np.clip(np.prod((1 - SF * efficiencies)[~tagged], axis=-1), 0, 10)
                    weights = P_data / P_MC
                    # sum_gen_weights_dict[path.split("/")[1]] = np.array(weights)

                    # doing syst for one year here
                    P_MC_up = np.prod(efficiencies[tagged], axis=-1) * np.prod((1 - efficiencies)[~tagged], axis=-1)
                    P_data_up = np.prod((SF_up * efficiencies)[tagged], axis=-1) * np.clip(np.prod((1 - SF_up * efficiencies)[~tagged], axis=-1), 0, 10)
                    weights_up = P_data_up / P_MC_up
                    P_MC_down = np.prod(efficiencies[tagged], axis=-1) * np.prod((1 - efficiencies)[~tagged], axis=-1)
                    P_data_down = np.prod((SF_down * efficiencies)[tagged], axis=-1) * np.clip(np.prod((1 - SF_down * efficiencies)[~tagged], axis=-1), 0, 10)
                    weights_down = P_data_down / P_MC_down
                    self.output()[lep + "_" + path.split("/")[1]]["weights"].parent.touch()
                    self.output()[lep + "_" + path.split("/")[1]]["weights"].dump(np.array(weights))
                    self.output()[lep + "_" + path.split("/")[1]]["up"].dump(np.array(weights_up))
                    self.output()[lep + "_" + path.split("/")[1]]["down"].dump(np.array(weights_down))
                    if key == sub_name:
                        total_arr.append(weights)
                        total_arr_up.append(weights_up)
                        total_arr_down.append(weights_down)

                    print("\ndone:", self.output()[lep + "_" + path.split("/")[1]]["weights"].path)

                    # # contigency plots
                    # weights_small = weights < 0.3
                    # jets5 = nJet >=5
                    # fig = plt.figure()
                    # plt.hist(ak.flatten(SF), label="SF from skimming", bins=20, density=True)
                    # plt.hist(ak.flatten(SF[weights_small]), label="SF in events with small weights", bins=20, density=True, histtype="step")
                    # plt.xlabel("SF")
                    # plt.ylabel("a.u.")
                    # plt.legend()
                    # plt.savefig(self.output()[lep + "_" + path.split("/")[1]]["weights"].parent.path + "/SF.png")

                    # fig = plt.figure()
                    # plt.hist(ak.flatten(efficiencies), label="efficiencies from skimming", bins=20, density=True)
                    # plt.hist(ak.flatten(efficiencies[weights_small]), label="efficiencies in events with small weights", bins=20, density=True, histtype="step")
                    # plt.xlabel("Efficiencies")
                    # plt.ylabel("a.u.")
                    # plt.legend()
                    # plt.savefig(self.output()[lep + "_" + path.split("/")[1]]["weights"].parent.path + "/efficiencies.png")

                    # fig = plt.figure()
                    # plt.hist(weights, label="weights", bins=50, density=True)
                    # plt.hist(weights[jets5], label="weights for >=5 jet events", bins=50, density=True, histtype="step")
                    # plt.xlabel("Computed event weights")
                    # plt.ylabel("a.u.")
                    # plt.legend()
                    # plt.savefig(self.output()[lep + "_" + path.split("/")[1]]["weights"].parent.path + "/weights.png")
                    # from IPython import embed; embed()
        if key == sub_name:
            print("In Signal")
            # store signal merged once so YieldPerMasspoint doesn't have to do redundant stuff
            total_signal_weights = np.array(np.concatenate(total_arr))
            total_signal_weights_up = np.array(np.concatenate(total_arr_up))
            total_signal_weights_down = np.array(np.concatenate(total_arr_down))
            np.save(self.output()[lep + "_" + path.split("/")[1]]["weights"].parent.path + self.config_inst.get_aux("all_btag_SF"), total_signal_weights)
            np.save(self.output()[lep + "_" + path.split("/")[1]]["weights"].parent.path + self.config_inst.get_aux("all_btag_SF").replace("_T5qqqqWW", "_up_T5qqqqWW"), total_signal_weights_up)
            np.save(self.output()[lep + "_" + path.split("/")[1]]["weights"].parent.path + self.config_inst.get_aux("all_btag_SF").replace("_T5qqqqWW", "_down_T5qqqqWW"), total_signal_weights_down)
