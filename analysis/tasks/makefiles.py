import os
import law
import uproot as up
from luigi import Parameter, BoolParameter
from tasks.basetasks import *
import json

"""
Tasks to write config for datasets from target directory
Then write a fileset directory as an input for coffea
"""


# class DownloadFiles(ConfigTask):
# class DownloadFilesWrapper(CampaignTask, law.WrapperTask):


class MakeFilesTask(AnalysisTask):
    """
    placeholder task to define common keywords
    """

    directory_path = Parameter(
        default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2023_01_18/2017/Data/root"
    )


class WriteDatasets(MakeFilesTask):
    def output(self):
        return {
            "dataset_dict": self.local_target("datasets_{}.json".format(self.year)),
            "dataset_path": self.local_target(
                "path.json"
            ),  # save this so you'll find the files
            "job_number_dict": self.local_target("job_number_dict.json"),
        }

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

        # print amount of files per dataset
        job_number_dict = {}
        for k in file_dict.keys():
            print(k, len(file_dict[k]))
            job_number_dict.update({k: len(file_dict[k])})

        # with open(self.output().path, "w") as out:
        #    json.dump(file_dict, out)
        self.output()["dataset_dict"].dump(file_dict)
        self.output()["dataset_path"].dump(self.directory_path)
        self.output()["job_number_dict"].dump(job_number_dict)


class WriteConfigData(MakeFilesTask):
    def requires(self):
        return WriteDatasets.req(self, directory_path=self.directory_path)

    def output(self):
        return self.local_target("datasets_{}.py".format(self.year))

    def write_add_dataset(self, name, number, keys):
        return [
            "cfg.add_dataset(",
            "'{}',".format(name),
            "{},".format("1" + str(number)),
            "campaign=campaign,",
            "keys={})".format(keys),
        ]

    # cfg.add_dataset(
    # "ttJets",
    # 1100,
    # campaign=campaign,
    # # keys=["/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"],
    # keys=[
    # "/nfs/dust/cms/user/frengelk/Testing/TTJets_TuneCP5_RunIISummer19UL16NanoAODv2_1.root"
    # ],
    # )

    def run(self):
        self.output().parent.touch()

        with open(self.input().path, "r") as read_file:
            file_dict = json.load(read_file)

        with open(self.output().path, "w") as out:
            out.write("import os")
            out.write("\n" + "#####datasets#####" + "\n")
            out.write("def setup_datasets(cfg, campaign):" + "\n")
            # from IPython import embed;embed()

            # for correct mapping of processes and all datasets
            proc_dict = self.config_inst.get_aux("proc_data_template")

            for proc in self.config_inst.processes:
                for child in proc.walk_processes():
                    child = child[0]
                    # print(child.name)
                    # from IPython import embed;embed()
                    # for sets in file_dict.keys():
                    if child.name in proc_dict.keys():
                        sets = []
                        for directory in proc_dict[child.name]:
                            sets.extend(file_dict[directory])
                        for line in self.write_add_dataset(child.name, child.id, sets):
                            out.write(line + "\n")


class WriteFileset(MakeFilesTask):

    directory_path = Parameter(
        default="/nfs/dust/cms/user/wiens/CMSSW/CMSSW_10_6_8/Skimming/2020_12_04/merged"
    )

    def requires(self):
        return WriteConfigData.req(self, directory_path=self.directory_path)

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
