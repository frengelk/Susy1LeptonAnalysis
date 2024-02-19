__all__ = ["AnalysisTask", "ConfigTask", "ShiftTask", "DatasetTask"]

import os

import luigi
import law
import order as od
import importlib
import math

from utils.sandbox import CMSSWSandboxTask

law.contrib.load("numpy", "tasks", "root", "htcondor", "hdf5", "coffea", "matplotlib")  # "wlcg",


class BaseTask(law.Task):
    version = luigi.Parameter(default="dev1", description="version of current workflow")

    def local_path(self, *path):
        parts = [str(p) for p in self.store_parts() + path]
        return os.path.join("/nfs/dust/cms/group/susy-desy/Susy1Lepton", *parts)

    def local_target(self, *args):
        cls = law.LocalFileTarget if args else law.LocalDirectoryTarget
        return cls(self.local_path(*args))

    def local_directory_target(self, *args):
        return law.LocalDirectoryTarget(self.local_path(*args))


class CampaignTask(BaseTask):
    year = luigi.Parameter(description="Year", default="2016")
    config = luigi.Parameter(default="SUSY_1lep_ML", description="current analysis")
    # analysis_choice = "common"
    category = luigi.Parameter(default="SR0b", description="Category defines set cuts")

    def __init__(self, *args, **kwargs):
        super(CampaignTask, self).__init__(*args, **kwargs)
        self.campaign_name = "Run2_pp_13TeV_{}".format(self.year)
        self.campaign_inst = importlib.import_module("config.{}".format(self.campaign_name)).campaign

    def store_parts(self):
        parts = (self.analysis_choice, self.campaign_name, self.__class__.__name__)
        if self.version is not None:
            parts += (self.version,)
        if self.category is not None:
            parts += (self.category,)
        return parts


class AnalysisTask(CampaignTask):
    analysis_id = "0b"
    task_namespace = "{}".format(analysis_id)
    analysis_choice = analysis_id

    def __init__(self, *args, **kwargs):
        super(AnalysisTask, self).__init__(*args, **kwargs)
        self.analysis_inst = self._import("config.analysis").analysis
        self.config_inst = self.analysis_inst.get_config(self.campaign_name)

    def _import(self, *parts):
        return importlib.import_module(".".join((self.analysis_choice,) + parts))

    @classmethod
    def modify_param_values(cls, params):
        return params


class ConfigTask(AnalysisTask):
    def __init__(self, *args, **kwargs):
        super(ConfigTask, self).__init__(*args, **kwargs)
        self.config_inst = self.analysis_inst.get_config(self.campaign_name)

    def store_parts(self):
        parts = (self.analysis_choice, self.campaign_name, self.__class__.__name__)
        if self.version is not None:
            parts += (self.version,)
        return parts


class DNNTask(AnalysisTask):
    """
    define parameters here for all relevant tasks
    """

    category = luigi.Parameter(default="N0b", description="channel to train on")
    epochs = luigi.IntParameter(default=100)
    batch_size = luigi.IntParameter(default=10000)
    learning_rate = luigi.FloatParameter(default=0.01)
    debug = luigi.BoolParameter(default=False)
    n_layers = luigi.IntParameter(default=3)
    n_nodes = luigi.IntParameter(default=256)
    steps_per_epoch = luigi.IntParameter(default=100)
    dropout = luigi.FloatParameter(default=0.2)

    # should be 2 for now, maybe larger later
    kfold = luigi.IntParameter(default=2)

    def __init__(self, *args, **kwargs):
        super(DNNTask, self).__init__(*args, **kwargs)


class ShiftTask(ConfigTask):
    shift = luigi.Parameter(
        default="nominal",
        significant=False,
        description="systematic shift to " "apply, default: nominal",
    )
    effective_shift = luigi.Parameter(default="nominal")
    shifts = set()
    exclude_params_index = {"effective_shift"}
    exclude_params_req = {"effective_shift"}
    exclude_params_sandbox = {"effective_shift"}

    @classmethod
    def modify_param_values(cls, params):
        if params["shift"] == "nominal":
            return params
        # shift known to config?
        config_inst = od.Config(cls.config)
        if params["shift"] not in config_inst.shifts:
            raise Exception("shift {} unknown to config {}".format(params["shift"], config_inst))
        # check if the shift is known to the task
        if params["shift"] in cls.shifts:
            params["effective_shift"] = params["shift"]
        return params

    def __init__(self, *args, **kwargs):
        super(ShiftTask, self).__init__(*args, **kwargs)
        # store the shift instance
        self.shift_inst = self.config_inst.get_shift(self.effective_shift)

    @property
    def store_parts(self):
        return super(ShiftTask, self).store_parts + (self.effective_shift,)


class DatasetTask(ConfigTask):  # ShiftTask
    dataset = luigi.Parameter(default="TTJets_sl_fromt", description="the dataset name, default: ")

    def __init__(self, *args, **kwargs):
        super(DatasetTask, self).__init__(*args, **kwargs)
        self.dataset_inst = self.config_inst.get_dataset(self.dataset)
        if len(self.dataset_inst.processes) == 1:
            self.process_inst = list(self.dataset_inst.processes.values())[0]
        else:
            self.process_inst = None


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    debug = luigi.BoolParameter()
    # configurable in task Parameters
    RAM = luigi.IntParameter(default=750)
    hours = luigi.IntParameter(default=1)

    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """

    def create_branch_map(self):
        n = 1
        # n = n * 1
        # if self.debug:
        # n = 1
        return list(range(n))

    def htcondor_post_submit_delay(self):
        return self.poll_interval * 60

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path("$ANALYSIS_BASE", "htcondor_bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        # render_variables are rendered into all files sent with a job
        config.render_variables["analysis_base"] = os.getenv("ANALYSIS_BASE")
        # copy the entire environment
        config.custom_content.append(("getenv", "True"))
        # config.custom_content.append(("SCHEDD_NAME", "bird-htc-sched14.desy.de"))
        config.custom_content.append(("universe", "vanilla"))
        # require more RAM on CPU
        # config.custom_content.append(("request_cpus", "1"))
        config.custom_content.append(("request_memory", str(self.RAM)))  # 20000
        # config.custom_content.append(("+RequestRuntime = 86400"))
        config.custom_content.append(("+RequestRuntime = {}*60*60".format(self.hours)))
        # config.custom_content.append(("Request_GPUs", "0"))
        # config.custom_content.append(("Request_GpuMemory", "0"))

        # condor logs
        config.stdout = "out.txt"
        config.stderr = "err.txt"
        config.log = "log.txt"
        return config

    def htcondor_use_local_scheduler(self):
        return True

    # def htcondor_create_job_manager(self, **kwargs):
    # kwargs = law.util.merge_dicts(
    # self.htcondor_job_manager_defaults, kwargs)
    # return HTCondorJobManagerRWTH(**kwargs)
