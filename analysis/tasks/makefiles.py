import os
import law
import uproot as up
from luigi import Parameter, BoolParameter
from tasks.base import *
import json

"""
Tasks to write config for datasets from target directory
Then write a fileset directory as an input for coffea
"""


class BaseMakeFilesTask(AnalysisTask):
    # Basis class only for inheritance
    directory_path = Parameter(default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2023_01_18/2017/Data/root")


class WriteDatasets(BaseMakeFilesTask):
    def output(self):
        return self.local_target("datasets_{}.json".format(self.year))

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

        file_dict.update({"directory_path": self.directory_path})  # self.directory_path + "/" +
        with open(self.output().path, "w") as out:
            json.dump(file_dict, out)

        # copy the json file directly to the aux part
        print("Writing job dict to:", self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + ".json"))
        os.system("cp {} {}".format(self.output().path, self.config_inst.get_aux("job_dict").replace(".json", "_" + self.version + ".json")))


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
        for k in file_dict.keys():
            print(k, len(file_dict[k]))
            job_number_dict.update({k: len(file_dict[k])})
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
