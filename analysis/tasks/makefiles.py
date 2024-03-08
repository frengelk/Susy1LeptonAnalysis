import os
import law
import uproot as up
import ROOT
import numpy as np
from luigi import Parameter, BoolParameter
from tasks.base import AnalysisTask, HTCondorWorkflow
import json
import awkward as ak

"""
Tasks to write config for datasets from target directory
Then write a fileset directory as an input for coffea
"""


class BaseMakeFilesTask(AnalysisTask):
    # Basis class only for inheritance
    directory_path = Parameter(default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2023_01_18/2017/Data/root")


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

        file_dict = {}
        for root, dirs, files in os.walk(self.directory_path):
            for directory in dirs:
                # print(directory)
                file_list = []
                for r, d, f in os.walk(self.directory_path + "/" + directory):
                    for file in f:
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
            for directory in dirs:
                # print(directory)
                file_list = []
                for r, d, f in os.walk(self.directory_path + "/" + directory):
                    for file in f:
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
                    print(f"Bin ({i}, {j}) with label ({bin_center_x:.2f}, {bin_center_y:.2f}) is filled with content: {bin_content}")
                    masspoints.append((bin_center_x, bin_center_y))
        return masspoints

    @law.decorator.safe_output
    def run(self):
        inp = self.input()
        path = inp["dataset_path"].load()
        signal_name = inp["dataset_dict"].load()["SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8"][0]
        file = ROOT.TFile(path + "/" + signal_name)
        hist = file.Get("numberOfT5qqqqWWGenEvents;1")
        masspoints = self.find_filled_bins(hist)
        self.output().dump({"masspoints": masspoints})
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
        cutflow_dict = {
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
                                # cutflow_dict["Muon"][key] = file['cutflow_Muon;1'].values()
                                # cutflow_dict["Electron"][key] = file['cutflow_Electron;1'].values()
                                sum_gen_weights_dict[new_key] = sum_gen_weight
                                # if "SR" in self.category and not "SR_Anti" in self.category:
                                #     muon_arr = file["cutflow_Muon;1"].values()
                                #     electron_arr = file["cutflow_Electron;1"].values()
                            elif new_key in sum_gen_weights_dict.keys():
                                sum_gen_weights_dict[new_key] += sum_gen_weight
                                # if "SR" in self.category and not "SR_Anti" in self.category:
                                #     muon_arr += file["cutflow_Muon;1"].values()
                                #     electron_arr += file["cutflow_Electron;1"].values()

                    # else:
                    # keys should do the same for both dicts
                    if key not in sum_gen_weights_dict.keys():
                        # cutflow_dict["Muon"][key] = file['cutflow_Muon;1'].values()
                        # cutflow_dict["Electron"][key] = file['cutflow_Electron;1'].values()
                        # if "SR" in self.category and not "SR_Anti" in self.category:
                        #     muon_arr = file["cutflow_Muon;1"].values()
                        #     electron_arr = file["cutflow_Electron;1"].values()
                        sum_gen_weights_dict[key] = sum_gen_weight
                        print("bg", key)
                    elif key in sum_gen_weights_dict.keys():
                        sum_gen_weights_dict[key] += sum_gen_weight
                        # if "SR" in self.category and not "SR_Anti" in self.category:
                        #     muon_arr += file["cutflow_Muon;1"].values()
                        #     electron_arr += file["cutflow_Electron;1"].values()

            # if "SR" in self.category and not "SR_Anti" in self.category:
            #     cutflow_dict["Muon"][key] = muon_arr[muon_arr > 0].tolist()
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
        #     self.output()["cutflow"].dump(cutflow_dict)


class CalcBTagSF(BaseMakeFilesTask, HTCondorWorkflow, law.LocalWorkflow):
    # require more ressources
    RAM = 2000
    hours = 5

    def requires(self):
        return WriteDatasets.req(self, category=self.category)

    def create_branch_map(self):
        # define job number according to number of files of the dataset that you want to process
        data_list_keys, _ = self.get_datalist_items()
        if self.debug:
            data_list_keys = data_list_keys[0]
        return list(range(len(data_list_keys)))

    def output(self):
        values = self.get_datalist_items()[1]
        out = {lep + "_" + val.split("/")[1]: self.local_target(lep + "_" + val.split("/")[1].replace(".root", ".npy")) for val in values for lep in self.config_inst.get_aux("channels")[self.category]}
        return out

    def store_parts(self):
        return super(CalcBTagSF, self).store_parts()

    def get_datalist_items(self):
        with open(self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + "_" + self.category + ".json")) as f:
            data_list = json.load(f)
        # so we can pop in place
        keys = list(data_list.keys())
        for key in keys:
            if key in self.config_inst.get_aux("data"):
                data_list.pop(key)
        # need to delete last entry, which is the path
        values = []
        data_list_values = map(values.extend, list(data_list.values())[:-1])
        # executes the map somehow?
        list(data_list_values)
        return (list(data_list.keys())[:-1], list(values))

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
        data_list_keys, data_list_values = self.get_datalist_items()
        sum_gen_weights_dict = {}
        # cutflow_dict = {
        #     "Muon": {},
        #     "Electron": {},
        # }
        dataset_dict = self.input().load()  # [self.channel[0]].collection.targets[0]
        # remove path from dict and save it for later
        data_path = dataset_dict.pop("directory_path")
        # for key in dataset_dict.keys():
        key = data_list_keys[self.branch]
        print(key)
        true_counts = np.array([])
        tagged_counts = np.array([])
        for path in dataset_dict[key]:
            with up.open(data_path + "/" + path) as file:
                # following method from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
                TrueB = file["nTrueB;1"].to_numpy()
                nBTag = file["nMediumBbTagDeepJet;1"].to_numpy()
                # not filled yet
                if true_counts.shape == (0,):
                    true_counts = TrueB[0]
                    tagged_counts = nBTag[0]
                    pt_bins, eta_bins = TrueB[1], TrueB[2]
                else:
                    true_counts += TrueB[0]
                    tagged_counts += nBTag[0]

        # total efficiency per file
        efficiency_hist = tagged_counts / true_counts
        # special hardcoded case to do complete Btag SF for mass scan
        # if key == 'SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8':
        #     total_arr = []

        for path in dataset_dict[key]:
            print(data_path + "/" + path)
            with up.open(data_path + "/" + path) as file:
                # having to do both trees seperatly
                for lep in self.config_inst.get_aux("channels")[self.category]:
                    # following method from https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
                    TrueB = file["nTrueB;1"].to_numpy()
                    nBTag = file["nMediumBbTagDeepJet;1"].to_numpy()

                    tree = file[lep]

                    tagged = tree["JetDeepJetMediumId"].array()
                    SF = tree["JetDeepJetMediumSf"].array()
                    jetPt = tree["JetPt"].array()

                    jetEta = tree["JetEta"].array()

                    # effs=(efficiency, pt_bins, eta_bins)
                    # computing efficiencies, do it flat to avoid for loop in python
                    efficiencies = ak.Array(self.get_bin_contents((efficiency_hist, pt_bins, eta_bins), ak.flatten(jetPt), ak.flatten(jetEta)))
                    # split them back up, we know the amount of jets
                    nJet = tree["nJet"].array()
                    efficiencies = ak.Array(np.split(efficiencies, np.cumsum(nJet)))
                    if len(efficiencies) > len(tagged):
                        # cumsum may result in empty array at the end
                        efficiencies = efficiencies[:-1]
                    P_MC = np.prod(efficiencies[tagged], axis=-1) * np.prod((1 - efficiencies)[~tagged], axis=-1)
                    P_data = np.prod((SF * efficiencies)[tagged], axis=-1) * np.prod((1 - SF * efficiencies)[~tagged], axis=-1)

                    weights = P_MC / P_data
                    # sum_gen_weights_dict[path.split("/")[1]] = np.array(weights)
                    self.output()[lep + "_" + path.split("/")[1]].parent.touch()
                    print("\n", self.output()[lep + "_" + path.split("/")[1]].path)
                    self.output()[lep + "_" + path.split("/")[1]].dump(np.array(weights))
                    # if key == 'SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8':
                    #     total_arr.append(weights)

        # if key == 'SMS-T5qqqqVV_TuneCP2_13TeV-madgraphMLM-pythia8':
        #     total_signal_weights=np.concatenate(total_arr)
        #     np.save(self.output()[lep + "_" + path.split("/")[1]].path.replace(".npy", "_total_signal.npy"), np.array(total_signal_weights))
