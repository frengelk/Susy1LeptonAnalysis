import law
import numpy as np
import pickle
from luigi import BoolParameter, IntParameter, FloatParameter, ChoiceParameter
import sklearn as sk
import sklearn.model_selection as skm
from rich.console import Console
from tqdm.auto import tqdm
from time import time
import ipdb

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data  # import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

# captum
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

from tasks.basetasks import DNNTask, HTCondorWorkflow
from tasks.arraypreparation import ArrayNormalisation, CrossValidationPrep

import utils.PytorchHelp as util


class PytorchMulticlass(DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    def create_branch_map(self):
        # overwrite branch map
        n = 1
        return list(range(n))

        # return {i: i for i in range(n)}

    def requires(self):
        # return PrepareDNN.req(self)
        return ArrayNormalisation.req(self, channel="N0b_CR")

    def output(self):
        return {
            "model": self.local_target("model.pt"),
            "loss_stats": self.local_target("loss_stats.json"),
            "accuracy_stats": self.local_target("accuracy_stats.json"),
            # test acc for optimizationdata for plotting
            # "test_acc": self.local_target("test_acc.json"),
        }

    def store_parts(self):
        # debug_str = ''
        if self.debug:
            debug_str = "debug"
        else:
            debug_str = ""

        # put hyperparameters in path to make an easy optimization search
        return (
            super(PytorchMulticlass, self).store_parts()
            + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (debug_str,)
        )

    def calc_class_weights(self, y_train, norm=1, sqrt=False):
        # calc class weights to battle imbalance
        # norm to tune down huge factors, sqrt to smooth the distribution
        from sklearn.utils import class_weight

        weight_array = norm * class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(np.argmax(y_train, axis=-1)),
            y=np.argmax(y_train, axis=-1),
        )

        if sqrt:
            # smooth by exponential function
            # return dict(enumerate(np.sqrt(weight_array)))
            return dict(enumerate((weight_array) ** 0.88))
        if not sqrt:
            # set at minimum to 1.0
            # return dict(enumerate([a if a>1.0 else 1.0 for a in weight_array]))
            return dict(enumerate(weight_array))

    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)
        _, y_test_tags = torch.max(y_test, dim=-1)

        correct_pred = (y_pred_tags == y_test_tags).float()
        acc = correct_pred.sum() / len(correct_pred)
        # from IPython import embed;embed()
        # acc = torch.round(acc * 100)
        return acc

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):

        tic = time()

        # define dimensions, working with aux template for processes
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template").keys())

        # load the prepared data and labels
        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()
        X_val = self.input()["X_val"].load()
        y_val = self.input()["y_val"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()

        # definition for the normalization layer
        means, stds = (
            self.input()["means_stds"].load()[0],
            self.input()["means_stds"].load()[1],
        )

        class_weights = self.calc_class_weights(y_train)
        # class_weights = {1: 1, 2: 1, 3: 1}

        # declare device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # datasets are loaded
        train_dataset = util.ClassifierDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        )
        val_dataset = util.ClassifierDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
        )
        test_dataset = util.ClassifierDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        )

        self.steps_per_epoch = (
            n_processes * np.sum(y_test[:, 0] == 1) // self.batch_size
        )

        # all in dat
        """
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            # batch_size=self.batch_size,
            batch_sampler=util.EventBatchSampler(
                y_train,
                self.batch_size,
                n_processes,
                self.steps_per_epoch,
            ),
            num_workers=8,
        )

        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=10 * self.batch_size,
            #shuffle=True,  # len(val_dataset
            num_workers=8,
        )  # =1
        """

        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=10 * self.batch_size,
            num_workers=8,  # , shuffle=True  # self.batch_size
        )

        # declare lighnting callbacks
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="max",
            strict=False,
        )

        swa_callback = pl.callbacks.StochasticWeightAveraging(
            swa_epoch_start=0.5,
        )

        # have to apply softmax somewhere on validation/inference FIXME

        # declare model
        model = util.MulticlassClassification(
            num_feature=n_variables,
            num_class=n_processes,
            means=means,
            stds=stds,
            dropout=self.dropout,
            class_weights=torch.tensor(
                list(class_weights.values())
            ),  # no effect right now
            n_nodes=self.n_nodes,
        )

        # define data
        data_collection = util.DataModuleClass(
            X_train,
            y_train,
            X_val,
            y_val,
            # X_test,
            # y_test,
            self.batch_size,
            n_processes,
            self.steps_per_epoch,
        )

        # needed for test evaluation
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        print(model)
        # accuracy_stats = {"train": [], "val": []}
        # loss_stats = {"train": [], "val": []}

        # collect callbacks
        callbacks = [early_stop_callback, swa_callback]

        # Trainer, for gpu gpus=1
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            num_nodes=1,
            callbacks=callbacks,
            enable_progress_bar=False,
            check_val_every_n_epoch=1,
        )

        if self.debug:
            from IPython import embed

            embed()
        # fit the trainer, includes the whole training loop
        # pdb.run(trainer.fit(model, dat))
        # ipdb.set_trace()

        trainer.fit(model, data_collection)

        # replace this loop with model(torch.tensor(X_test)) ?
        # evaluate test set
        with torch.no_grad():

            test_epoch_loss = 0
            test_epoch_acc = 0

            model.eval()
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.squeeze(
                    0
                ), y_test_batch.squeeze(0)

                y_test_pred = model(X_test_batch)

                test_loss = criterion(y_test_pred, y_test_batch)
                test_acc = self.multi_acc(y_test_pred, y_test_batch)

                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()

        # print result
        console = Console()
        console.print(
            "\n[u][bold magenta]Test accuracy on channel {}:[/bold magenta][/u]".format(
                self.channel
            )
        )
        console.print(test_acc, "\n")
        if self.debug:
            from IPython import embed

            embed()
        # save away all stats
        self.output()["model"].touch()
        torch.save(model, self.output()["model"].path)
        self.output()["loss_stats"].dump(model.loss_stats)
        self.output()["accuracy_stats"].dump(model.accuracy_stats)


class PytorchCrossVal(DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    # define it here again so training can be started from here
    kfold = IntParameter(default=5)

    def create_branch_map(self):
        # overwrite branch map
        n = 1
        return list(range(n))

        # return {i: i for i in range(n)}

    def requires(self):
        # return PrepareDNN.req(self)
        return {
            "data": CrossValidationPrep.req(self, kfold=self.kfold),
            "mean_std": ArrayNormalisation.req(self),
        }

    def output(self):
        return self.local_target("performance.json")

    def store_parts(self):
        # debug_str = ''
        if self.debug:
            debug_str = "debug"
        else:
            debug_str = ""

        # put hyperparameters in path to make an easy optimization search
        return (
            super(PytorchCrossVal, self).store_parts()
            + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (debug_str,)
        )

    def calc_class_weights(self, y_train, norm=1, sqrt=False):
        # calc class weights to battle imbalance
        # norm to tune down huge factors, sqrt to smooth the distribution
        from sklearn.utils import class_weight

        weight_array = norm * class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(np.argmax(y_train, axis=-1)),
            y=np.argmax(y_train, axis=-1),
        )

        if sqrt:
            # smooth by exponential function
            # return dict(enumerate(np.sqrt(weight_array)))
            return dict(enumerate((weight_array) ** 0.88))
        if not sqrt:
            # set at minimum to 1.0
            # return dict(enumerate([a if a>1.0 else 1.0 for a in weight_array]))
            return dict(enumerate(weight_array))

    def reset_weights(self, m):
        """
        Try resetting model weights to avoid
        weight leakage.
        """
        for layer in m.children():
            if hasattr(layer, "reset_parameters"):
                print(f"Reset trainable parameters of layer = {layer}")
                layer.reset_parameters()

    def run(self):

        # define dimensions, working with aux template for processes
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template").keys())

        # definition for the normalization layer
        means, stds = (
            self.input()["mean_std"]["means_stds"].load()[0],
            self.input()["mean_std"]["means_stds"].load()[1],
        )

        performance = {}

        for i in range(self.kfold):

            # only load needed data set config in each step
            X_train = self.input()["data"]["cross_val_{}".format(i)][
                "cross_val_X_train_{}".format(i)
            ].load()
            y_train = self.input()["data"]["cross_val_{}".format(i)][
                "cross_val_y_train_{}".format(i)
            ].load()
            X_val = self.input()["data"]["cross_val_{}".format(i)][
                "cross_val_X_val_{}".format(i)
            ].load()
            y_val = self.input()["data"]["cross_val_{}".format(i)][
                "cross_val_y_val_{}".format(i)
            ].load()

            class_weights = self.calc_class_weights(y_train)

            # declare model
            model = util.MulticlassClassification(
                num_feature=n_variables,
                num_class=n_processes,
                means=means,
                stds=stds,
                dropout=self.dropout,
                class_weights=torch.tensor(list(class_weights.values())),
                n_nodes=self.n_nodes,
            )

            # weight resetting
            model.apply(self.reset_weights)

            # datasets are loaded
            train_dataset = util.ClassifierDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
            )
            val_dataset = util.ClassifierDataset(
                torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
            )

            # define data
            data_collection = util.DataModuleClass(
                X_train,
                y_train,
                X_val,
                y_val,
                # X_test,
                # y_test,
                self.batch_size,
                n_processes,
                self.steps_per_epoch,
            )

            # Trainer, for gpu gpus=1
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                num_nodes=1,
                enable_progress_bar=False,
                check_val_every_n_epoch=1,
            )

            trainer.fit(model, data_collection)

            # Print fold results
            print("K-FOLD CROSS VALIDATION RESULTS FOR {} FOLDS".format(i))
            print("--------------------------------")

            # for key, value in results.items():
            print(
                "Latest accuracy train: {} val: {}".format(
                    model.accuracy_stats["train"][-1], model.accuracy_stats["val"][-1]
                )
            )
            print(
                "Latest loss train: {} val: {} \n".format(
                    model.loss_stats["train"][-1], model.loss_stats["val"][-1]
                )
            )
            # sum += value
            # print(f'Average: {sum/len(results.items())} %')

            performance.update(
                {
                    i: [
                        model.accuracy_stats["train"][-1],
                        model.accuracy_stats["val"][-1],
                        model.loss_stats["train"][-1],
                        model.loss_stats["val"][-1],
                    ]
                }
            )

        self.output().dump(performance)
