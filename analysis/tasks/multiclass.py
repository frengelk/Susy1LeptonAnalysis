# coding: utf-8

import law
import numpy as np
from luigi import IntParameter
from rich.console import Console
from time import time

# torch imports
import torch
import torch.nn as nn
import torch.utils.data as data  # import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from tasks.base import DNNTask, HTCondorWorkflow, AnalysisTask
from tasks.grouping import MergeArrays
from tasks.arraypreparation import ArrayNormalisation, CrossValidationPrep
from tasks.coffea import CoffeaTask

import utils.pytorch_base as util


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

    def calc_class_weights(self, y_train, norm=1, sqrt=False):  # , weight_train
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
        n_processes = len(self.config_inst.get_aux("DNN_process_template")["N" + self.channel].keys())

        # load the prepared data and labels
        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()
        X_val = self.input()["X_val"].load()
        y_val = self.input()["y_val"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()

        weight_train = self.input()["weight_train"].load()

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
        train_dataset = util.ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = util.ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        test_dataset = util.ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

        self.steps_per_epoch = n_processes * np.sum(y_test[:, 0] == 1) // self.batch_size

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
            num_workers=0,  # , shuffle=True  # self.batch_size
        )

        # declare lighnting callbacks
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="max",
            strict=False,
        )

        # FIXME
        # swa_callback = pl.callbacks.StochasticWeightAveraging(
        # swa_epoch_start=0.5,
        # )

        # have to apply softmax somewhere on validation/inference FIXME

        # declare model
        model = util.MulticlassClassification(
            num_feature=n_variables,
            num_class=n_processes,
            means=means,
            stds=stds,
            dropout=self.dropout,
            class_weights=torch.tensor(list(class_weights.values())),  # no effect right now
            n_nodes=self.n_nodes,
        )

        # define data
        data_collection = util.DataModuleClass(
            X_train,
            y_train,
            weight_train,
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
        callbacks = [early_stop_callback]  # , swa_callback

        # Trainer, for gpu gpus=1
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            # num_nodes=1,
            callbacks=callbacks,
            enable_progress_bar=True,  # False
            check_val_every_n_epoch=1,
        )

        if self.debug:
            from IPython import embed

            embed()
        # fit the trainer, includes the whole training loop
        # pdb.run(trainer.fit(model, dat))
        # ipdb.set_trace()

        data_collection.setup("train")
        trainer.fit(model, data_collection)

        # replace this loop with model(torch.tensor(X_test)) ?
        # evaluate test set
        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_acc = 0

            model.eval()
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.squeeze(0), y_test_batch.squeeze(0)

                y_test_pred = model(X_test_batch)

                test_loss = criterion(y_test_pred, y_test_batch)
                test_acc = self.multi_acc(y_test_pred, y_test_batch)

                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()

        # print result
        console = Console()
        console.print("\n[u][bold magenta]Test accuracy on channel {}:[/bold magenta][/u]".format(self.channel))
        console.print(test_acc.item(), "\n")
        if self.debug:
            from IPython import embed

            embed()
        # save away all stats
        self.output()["model"].touch()
        torch.save(model, self.output()["model"].path)
        self.output()["loss_stats"].dump(model.loss_stats)
        self.output()["accuracy_stats"].dump(model.accuracy_stats)


class PytorchCrossVal(DNNTask, HTCondorWorkflow, law.LocalWorkflow):  # , CoffeaTask
    # define it here again so training can be started from here
    kfold = IntParameter(default=2)
    # setting needed RAM high
    RAM = 40000

    def create_branch_map(self):
        # overwrite branch map
        return list(range(self.kfold))

    def requires(self):
        # return PrepareDNN.req(self)
        return {
            "data": CrossValidationPrep.req(self, kfold=self.kfold),
            "mean_std": ArrayNormalisation.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "Rare", "DY", "T5qqqqVV", "MET", "SingleMuon", "SingleElectron"], channel=["Muon", "Electron"]),
        }

    def output(self):
        out = {"fold_{}".format(i): {"model": self.local_target("model_{}.pt".format(i)), "performance": self.local_target("performance_{}.json".format(i))} for i in range(self.kfold)}

        #        out.update({"performance": self.local_target("performance.json")})
        return out

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

    def calc_class_weights(self, y_train, y_weight, norm=1, sqrt=False):
        # calc class weights to battle imbalance
        # norm to tune down huge factors, sqrt to smooth the distribution
        """
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
        """
        n_processes = len(self.config_inst.get_aux("DNN_process_template")["N" + self.channel].keys())
        total_weight = np.sum(y_weight)
        class_weights = {}
        for i in range(n_processes):
            mask = np.argmax(y_train, axis=-1) == i
            class_weights[i] = total_weight / np.sum(y_weight[mask])
        return class_weights

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
        n_processes = len(self.config_inst.get_aux("DNN_process_template")["N" + self.channel].keys())

        # definition for the normalization layer
        means, stds = (
            self.input()["mean_std"]["means_stds"].load()[0],
            self.input()["mean_std"]["means_stds"].load()[1],
        )

        performance = {}

        i = self.branch
        # only load needed data set config in each step
        X_train = self.input()["data"]["cross_val_{}".format(i)]["cross_val_X_train_{}".format(i)].load()
        y_train = self.input()["data"]["cross_val_{}".format(i)]["cross_val_y_train_{}".format(i)].load()
        weight_train = self.input()["data"]["cross_val_{}".format(i)]["cross_val_weight_train_{}".format(i)].load()
        X_val = self.input()["data"]["cross_val_{}".format(i)]["cross_val_X_val_{}".format(i)].load()
        y_val = self.input()["data"]["cross_val_{}".format(i)]["cross_val_y_val_{}".format(i)].load()

        class_weights = self.calc_class_weights(y_train, weight_train)
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
        train_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(weight_train).float())
        val_dataset = util.ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

        # define data
        data_collection = util.DataModuleClass(
            X_train,
            y_train,
            weight_train,
            X_val,
            y_val,
            # X_test,
            # y_test,
            self.batch_size,
            n_processes,
            self.steps_per_epoch,
        )

        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode="max",
            strict=False,
        )
        # collect callbacks
        callbacks = [early_stop_callback]  # , swa_callback

        # Trainer, for gpu gpus=1
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            # num_nodes=1,
            callbacks=callbacks,
            enable_progress_bar=False,
            check_val_every_n_epoch=1,
        )

        data_collection.setup("train")
        trainer.fit(model, data_collection)

        # Print fold results
        print("K-FOLD CROSS VALIDATION RESULTS FOR {} FOLDS".format(i))
        print("--------------------------------")

        # for key, value in results.items():
        print("Latest accuracy train: {} val: {}".format(model.accuracy_stats["train"][-1], model.accuracy_stats["val"][-1]))
        print("Latest loss train: {} val: {} \n".format(model.loss_stats["train"][-1], model.loss_stats["val"][-1]))
        # sum += value
        # print(f'Average: {sum/len(results.items())} %')

        performance.update(
            {
                "accuracy_stats": model.accuracy_stats,
                "loss_stats": model.loss_stats,
            }
        )

        self.output()["fold_" + str(i)]["model"].parent.touch()
        torch.save(model, self.output()["fold_" + str(i)]["model"].path)
        self.output()["fold_" + str(i)]["performance"].dump(performance)


class PredictDNNScores(DNNTask):  # Requiring MergeArrays from a DNNTask leads to JSON errors
    # should be
    kfold = IntParameter(default=2)

    def requires(self):
        out = {
            "samples": CrossValidationPrep.req(self, kfold=self.kfold),
            "models": PytorchCrossVal.req(self, kfold=self.kfold, workflow="local"),
            # "data": ArrayNormalisation.req(self),
        }

        return out

    def output(self):
        return {"scores": self.local_target("scores.npy"), "labels": self.local_target("labels.npy"), "data": self.local_target("data_scores.npy"), "weights": self.local_target("weights.npy")}

    def store_parts(self):
        # put hyperparameters in path to make an easy optimization search
        return (
            super(PredictDNNScores, self).store_parts()
            + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
        )

    def run(self):
        models = self.input()["models"]["collection"].targets[0]
        samples = self.input()["samples"]
        scores, labels, weights = [], [], []
        data_scores = {}
        data = self.input()["samples"]["data"].load()
        for i in range(self.kfold):
            # to switch training/validation
            j = abs(i - 1)
            # each model should now predict labels for the validation data
            model = torch.load(models["fold_" + str(i)]["model"].path)
            inp_data = self.input()["samples"]["cross_val_" + str(j)]
            X_test = np.concatenate([inp_data["cross_val_X_train_" + str(j)].load(), inp_data["cross_val_X_val_" + str(j)].load()])
            y_test = np.concatenate([inp_data["cross_val_y_train_" + str(j)].load(), inp_data["cross_val_y_val_" + str(j)].load()])
            weight_test = np.concatenate([inp_data["cross_val_weight_train_" + str(j)].load(), inp_data["cross_val_weight_val_" + str(j)].load()])

            pred_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(weight_test).float())
            pred_loader = torch.utils.data.DataLoader(dataset=pred_dataset, batch_size=len(X_test))

            with torch.no_grad():
                model.eval()
                for X_pred_batch, y_pred_batch, weight_pred_batch in pred_loader:
                    X_scores = model(X_pred_batch)

                    scores.append(X_scores.numpy())
                    labels.append(y_pred_batch)
                    weights.append(weight_pred_batch)

                    # test_predict = reconstructed_model.predict(X_test)
                    # y_predictions_0 = np.array(y_predictions[0])

            data_pred_list = []
            data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=1000)
            with torch.no_grad():
                model.eval()
                for X_data in data_loader:
                    data_pred = model(X_data)
                    data_pred_list.append(data_pred.numpy())

            data_scores.update({i: np.concatenate(data_pred_list)})

        averaged_data_scores = sum(list(data_scores.values())) / len(data_scores.values())
        self.output()["scores"].dump(np.concatenate(scores))
        self.output()["labels"].dump(np.concatenate(labels))
        self.output()["weights"].dump(np.concatenate(weights))
        self.output()["data"].dump(averaged_data_scores)


"""
class SavingDNNModel(CoffeaTask):

    def requires(self):
        return ArrayNormalisation.req(self, data=True)

    def output(self):
        return self.local_target("test.txt")

    def run(self):
        from IPython import embed; embed()
        print("Saving model from {} to {}".format(self.input()["collection"].targets[0]["model"].path, self.config_inst.get_aux("DNN_model")))
        os.system("cp {} {}".format(self.input()["collection"].targets[0]["model"].path, self.config_inst.get_aux("DNN_model")))

        #placeholder so task is complete
        self.output().touch()
"""
