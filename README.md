# SusyLeptonAnalysis

*Getting moved to the respective wiki pages, keeping it right now*

Creation of the needed conda environment from .yml file (points to my directory, a lot of packages)
(Currently not neccessary when using setup.sh, but can be customized)
```shell
conda environment: conda env create -f law_env.yml
conda activate law_env
```

Beautifying all .py-files from command line (before commiting!): 
```shell
python beautify.py
```

Setup law indexing, autocompletion and task recognition
law setup: 
```shell
source analysis/setup.sh
law index 
```
law index collect all tasks, needed for new tasks

By default, this example uses a local scheduler, which - by definition - offers no visualization tools in the browser. If you want to see how the task tree is built and subsequently run, run luigid in a second terminal. This will start a central scheduler at localhost:8082 (the default address). To inform tasks (or rather workers) about the scheduler, either add --local-scheduler False to the law run command, or set the local-scheduler value in the [luigi_core] config section in the law.cfg file to False.


# Current analysis tasks

If you want to add datasets, add the corresponding processes, then map them.

Writes the config/datasets.py, taken from previously skimmed files
```shell
law run mj.WriteConfigData --local-scheduler --version dev1
```

Creating fileset (loops over all datasets)
```shell
law run mj.WriteFileset --local-scheduler --version dev1
```

Histogram the files and save as coffea.hists

This includes a Hstogram for every category-variable-dataset combination.
```shell
law run mj.CoffeaProcessor --version dev1 --local-scheduler (--processor Histogramer|ArrayExporter)
```
Debug the selection and processor
```shell
law run mj.CoffeaProcessor --version test1 --local-scheduler --debug --workflow local (--processor Histogramer|ArrayExporter)
```

Plotting of hists into a pdf, log scale allows for y-axis change

Groups subprocesses into parents, seperate plots for category-variables

Unblinded keyword allows for addition of data points and adds a ratio plot of backgrounds to data

```shell
law run mj.PlotCoffeaHists --version dev1 --local-scheduler (--log-scale) (--unblinded) 
```

Save variables into arrays for later computation, can be investigated by ArrayPlotting
```shell
law run mj.CoffeaProcessor --version dev1 --local-scheduler --processor ArrayExporter
```

transfer coffea hists to hists in a root file
```shell
law run 0b.GroupCoffeaProcesses --version test1 --local-scheduler
```

Modify produced numpy arrays, e.g. for DNN
```shell
law run 0b.ArrayNormalisation --version testDNN --local-scheduler
```

Use the numpy arrays to train a deep neural network, currently fully connected feed forward
Keywords include options for layers, nodes, dropout, epochs and more (look in basetasks.DNNTask)
```shell
law run 0b.DNNTrainer --version test1 --local-scheduler --workflow local --dropout 0.2 --n-nodes 128 --n-layers 3 --epochs 100 
```



helpful to kill local jobs still running somewhere
```shell
--cleanup-jobs
```

# Documentation links:

Jagged arrays: https://github.com/scikit-hep/awkward-0.x/blob/master/docs/classes.adoc#methods

coffea plotting: https://coffeateam.github.io/coffea/api/coffea.hist.plot1d.html

coffea hists: https://coffeateam.github.io/coffea/api/coffea.hist.Hist.html

law: https://github.com/riga/law

order: https://github.com/riga/order

Previous work (and some recycled code snippets): https://git.rwth-aachen.de/3pia/cms_analyses/common/-/tree/mva_prep
