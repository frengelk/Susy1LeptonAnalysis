# coding: utf-8

import law
import numpy as np
from luigi import IntParameter, ListParameter, BoolParameter
from rich.console import Console
from time import time

# torch imports
import torch
import torch.nn as nn
import torch.utils.data as data  # import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import matplotlib.pyplot as plt


from tasks.base import DNNTask, HTCondorWorkflow, AnalysisTask
from tasks.grouping import MergeArrays, MergeShiftArrays
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
        return ArrayNormalisation.req(self)

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
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (debug_str,)
        )

    def calc_class_weights(self, y_train, y_weight, norm=1, sqrt=False):  # , weight_train
        # calc class weights to battle imbalance
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        total_weight = np.sum(y_weight)
        class_weights = {}
        for i in range(n_processes):
            mask = np.argmax(y_train, axis=-1) == i
            class_weights[i] = total_weight / np.sum(y_weight[mask])
        return class_weights

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
        # multiprocessing enabling
        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy("file_system")

        tic = time()

        # define dimensions, working with aux template for processes
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

        # load the prepared data and labels
        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()  # [:,1]
        weight_train = self.input()["weight_train"].load()
        X_val = self.input()["X_val"].load()
        y_val = self.input()["y_val"].load()  # [:,1]
        weight_val = self.input()["weight_val"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()  # [:,1]

        # definition for the normalization layer
        means, stds = (
            self.input()["means_stds"].load()[0],
            self.input()["means_stds"].load()[1],
        )

        class_weights = self.calc_class_weights(y_train, weight_train)
        # class_weights = {1: 1, 2: 1, 3: 1}

        # declare device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # datasets are loaded
        train_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(weight_train).float())
        val_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float(), torch.from_numpy(weight_val).float())
        test_dataset = util.ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

        self.steps_per_epoch = n_processes * np.sum(y_test[:, 0] == 1) // self.batch_size

        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=10 * self.batch_size,
            num_workers=0,  # , shuffle=True  # self.batch_size
        )

        # declare lighnting callbacks
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=10,
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
            class_weights=torch.tensor(list(class_weights.values())),
            n_nodes=self.n_nodes,
            learning_rate=self.learning_rate,
        )

        # define data
        data_collection = util.DataModuleClass(
            X_train,
            y_train,
            weight_train,
            X_val,
            y_val,
            weight_val,
            # X_test,
            # y_test,
            self.batch_size,
            n_processes,
            # self.steps_per_epoch,
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

        # data_collection.setup("train")
        trainer.fit(model, data_collection)

        # replace this loop with model(torch.tensor(X_test)) ?
        # evaluate test set
        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_acc = 0

            model.eval()
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.squeeze(0), y_test_batch.squeeze(0)

                y_test_pred = model(X_test_batch).softmax(dim=1)

                test_loss = criterion(y_test_pred, y_test_batch)
                test_acc = self.multi_acc(y_test_pred, y_test_batch)

                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()

        # print result
        console = Console()
        console.print("\n[u][bold magenta]Test accuracy on channel {}:[/bold magenta][/u]".format(self.category))
        console.print(test_acc.item(), "\n")
        if self.debug:
            from IPython import embed

            embed()
        # save away all stats
        self.output()["model"].touch()
        torch.save(model, self.output()["model"].path)
        self.output()["loss_stats"].dump(model.loss_stats)
        self.output()["accuracy_stats"].dump(model.accuracy_stats)

        # plot loss comp
        fig = plt.figure()
        plt.plot(np.arange(0, len(model.loss_per_node["val_MC"]), 1), torch.tensor(model.loss_per_node["val_MC"]).numpy(), label="bkg")
        plt.plot(np.arange(0, len(model.loss_per_node["val_T5"]), 1), torch.tensor(model.loss_per_node["val_T5"]).numpy(), label="T5")
        plt.xlabel("Iterations")
        plt.ylabel("val_loss")
        plt.legend()
        plt.savefig(self.output()["model"].parent.path + "/val_losses.png")

        # plot loss comp
        fig = plt.figure()
        plt.plot(np.arange(0, len(model.loss_per_node["val_MC_weight"]), 1), torch.tensor(model.loss_per_node["val_MC_weight"]).numpy(), label="bkg")
        plt.plot(np.arange(0, len(model.loss_per_node["val_T5_weight"]), 1), torch.tensor(model.loss_per_node["val_T5_weight"]).numpy(), label="T5")
        plt.xlabel("Iterations")
        plt.ylabel("val_loss_weight")
        plt.legend()
        plt.savefig(self.output()["model"].parent.path + "/val_losses_weight.png")

        # signal_batch_fraction
        fig = plt.figure()
        plt.hist(torch.tensor(model.signal_batch_fraction).numpy(), label="Signal per batch", bins=np.arange(0.1, 0.75, 0.01))
        plt.xlabel("Fraction")
        plt.ylabel("a.u.")
        plt.legend()
        plt.savefig(self.output()["model"].parent.path + "/signal_fraction.png")


class PytorchCrossVal(DNNTask, CoffeaTask, HTCondorWorkflow, law.LocalWorkflow):
    # define it here again so training can be started from here
    kfold = IntParameter(default=2)
    # setting needed RAM high and time long
    RAM = 12000
    hours = 20

    # This parameters worked
    #  --batch-size 512 --dropout 0.3 --learning-rate 0.0003 --n-nodes 256

    def create_branch_map(self):
        # overwrite branch map
        return list(range(self.kfold))

    def requires(self):
        # return PrepareDNN.req(self)
        return {
            "data": CrossValidationPrep.req(self, kfold=self.kfold),
            "mean_std": ArrayNormalisation.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "Rare", "DY", "T5qqqqWW", "MET", "SingleMuon", "SingleElectron"]),
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
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (self.gamma if self.gamma != 1.0 else "",)  # so old output can still be found
            + (debug_str,)
        )

    def calc_class_weights(self, y_train, y_weight, norm=1, sqrt=False):
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

        # n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        # total_weight = np.sum(y_weight)
        # class_weights = {}
        # for i in range(n_processes):
        #     mask = np.argmax(y_train, axis=-1) == i
        #     class_weights[i] = total_weight / np.sum(y_weight[mask])
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
        # problems with opening files on NAF
        torch.multiprocessing.set_sharing_strategy("file_system")
        # define dimensions, working with aux template for processes
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

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
        weight_val = self.input()["data"]["cross_val_{}".format(i)]["cross_val_weight_val_{}".format(i)].load()

        class_weights = self.calc_class_weights(y_train, weight_train, sqrt=False)
        # declare model
        model = util.MulticlassClassification(
            num_feature=n_variables,
            num_class=n_processes,
            means=means,
            stds=stds,
            dropout=self.dropout,
            class_weights=torch.tensor(list(class_weights.values())),
            n_nodes=self.n_nodes,
            learning_rate=self.learning_rate,
            gamma=self.gamma,  # from Hyperparam Opt 1.5,
        )

        # weight resetting
        model.apply(self.reset_weights)

        # datasets are loaded
        # train_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(weight_train).float())
        # # val_dataset = util.ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        # val_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float(), torch.from_numpy(weight_val).float())

        # define data
        data_collection = util.DataModuleClass(
            X_train,
            y_train,
            weight_train,
            X_val,
            y_val,
            weight_val,
            # X_test,
            # y_test,
            self.batch_size,
            n_processes,
            # self.steps_per_epoch,
        )

        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",  # "val_acc",
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode="min",  # "max"
            strict=False,
        )

        # should help better generalization, reduce learning rate later
        swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=self.learning_rate / 2, swa_epoch_start=0.2)
        # collect callbacks
        callbacks = [pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"), early_stopping_callback, swa_callback]

        # Trainer, for gpu gpus=1
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            # num_nodes=1,
            callbacks=callbacks,
            enable_progress_bar=False,
            check_val_every_n_epoch=1,
        )

        print(model)
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

        # plot loss comp
        fig = plt.figure()
        plt.plot(np.arange(0, len(model.loss_per_node["val_MC"]), 1), torch.tensor(model.loss_per_node["val_MC"]).numpy(), label="bkg")
        plt.plot(np.arange(0, len(model.loss_per_node["val_T5"]), 1), torch.tensor(model.loss_per_node["val_T5"]).numpy(), label="T5")
        plt.xlabel("Iterations")
        plt.ylabel("val_loss")
        plt.legend()
        plt.savefig(self.output()["fold_" + str(i)]["model"].parent.path + "/val_losses_fold{}.png".format(str(i)))

        # plot loss comp
        fig = plt.figure()
        plt.plot(np.arange(0, len(model.loss_per_node["val_MC_weight"]), 1), torch.tensor(model.loss_per_node["val_MC_weight"]).numpy(), label="bkg")
        plt.plot(np.arange(0, len(model.loss_per_node["val_T5_weight"]), 1), torch.tensor(model.loss_per_node["val_T5_weight"]).numpy(), label="T5")
        plt.xlabel("Iterations")
        plt.ylabel("val_loss_weight")
        plt.legend()
        plt.savefig(self.output()["fold_" + str(i)]["model"].parent.path + "/val_losses_fold{}_weight.png".format(str(i)))

        # signal_batch_fraction
        fig = plt.figure()
        plt.hist(torch.tensor(model.signal_batch_fraction).numpy(), label="Signal per batch", bins=np.arange(0.25, 0.75, 0.01))
        plt.xlabel("Fraction")
        plt.ylabel("a.u.")
        plt.legend()
        plt.savefig(self.output()["fold_" + str(i)]["model"].parent.path + "/signal_fraction{}.png".format(str(i)))


class PredictDNNScores(CoffeaTask, DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    # needs RAM or it fails
    RAM = 5000
    hours = 1
    datasets_to_process = ListParameter(default=["QCD"])
    do_shifts = BoolParameter(default=False)

    def requires(self):
        out = {
            "samples": CrossValidationPrep.req(self, kfold=self.kfold),
            "models": PytorchCrossVal.req(self, kfold=self.kfold, category="All_Lep"),
            "merged": MergeArrays.req(self, datasets_to_process=["QCD"] + list(self.datasets_to_process), channel=["LeptonIncl"]),
            # "data": ArrayNormalisation.req(self),
        }
        if self.do_shifts:
            out.update({"shifts": MergeShiftArrays.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "QCD", "Rare", "DY"], shifts=["systematic_shifts", "TotalUp", "TotalDown", "JERUp", "JERDown"])})
        return out

    def output(self):
        out = {"scores": self.local_target("scores.npy"), "labels": self.local_target("labels.npy"), "data": self.local_target("data_scores.npy"), "weights": self.local_target("weights.npy"), "QCD_scores": self.local_target("QCD_scores.npy"), "QCD_weights": self.local_target("QCD_weights.npy")}
        out.update({dat: self.local_target(dat + "_scores.npy") for dat in self.datasets_to_process})
        out.update({dat + "_weights": self.local_target(dat + "_weights.npy") for dat in self.datasets_to_process})
        if self.do_shifts:
            targets = {}
            for key in self.input()["shifts"].keys():
                targets["scores_" + key] = self.local_target(key + "_scores.npy")
                targets["weights_" + key] = self.local_target(key + "_weights.npy")
            out.update({"shifts": targets})
        return out

    def store_parts(self):
        # put hyperparameters in path to make an easy optimization search
        parts = tuple()
        if self.do_shifts:
            parts += ("do_shifts",)
        return (
            super(PredictDNNScores, self).store_parts()
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (self.gamma,)
            + parts
        )

    def run(self):
        models = self.input()["models"]  # ["collection"].targets[0]
        samples = self.input()["samples"]
        scores, labels, weights = [], [], []
        data_scores = {}
        QCD_scores = {}
        data = self.input()["samples"]["data"].load()
        QCD = self.input()["merged"][self.category + "_QCD"]

        # first extractz QCD to substract predicted from data
        QCD_arr = QCD["array"].load()
        QCD_weights = QCD["weights"].load()

        dats_to_predict = {}
        if self.do_shifts:
            shifts_to_predict = {}
            shifts_in = self.input()["shifts"]
        for i in range(self.kfold):
            # to switch training/validation
            j = abs(i - 1)
            # each model should now predict labels for the validation data
            model = torch.load(models["fold_" + str(i)]["model"].path)
            inputs = self.input()["samples"]["cross_val_" + str(j)]
            X_test = np.concatenate([inputs["cross_val_X_train_" + str(j)].load(), inputs["cross_val_X_val_" + str(j)].load()])
            y_test = np.concatenate([inputs["cross_val_y_train_" + str(j)].load(), inputs["cross_val_y_val_" + str(j)].load()])
            weight_test = np.concatenate([inputs["cross_val_weight_train_" + str(j)].load(), inputs["cross_val_weight_val_" + str(j)].load()])

            # pred_dataset = util.ClassifierDatasetWeight(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(weight_test).float())
            # pred_loader = torch.utils.data.DataLoader(dataset=pred_dataset, batch_size=len(X_test))

            with torch.no_grad():
                model.eval()
                X_scores = model(torch.tensor(X_test)).softmax(dim=1)
                data_pred = model(torch.tensor(data)).softmax(dim=1)
                QCD_pred = model(torch.tensor(QCD_arr)).softmax(dim=1)
                for dat in self.datasets_to_process:
                    arr = self.input()["merged"][self.category + "_" + dat]["array"].load()
                    arr_weights = self.input()["merged"][self.category + "_" + dat]["weights"].load()
                    arr_DNNId = self.input()["merged"][self.category + "_" + dat]["DNNId"].load()
                    dats_to_predict[dat + "_scores" + str(i)] = model(torch.tensor(arr[arr_DNNId == 1 - 2 * i])).softmax(dim=1)
                    dats_to_predict[dat + "_weights" + str(i)] = arr_weights[arr_DNNId == 1 - 2 * i]
                if self.do_shifts:
                    for key in shifts_in.keys():
                        shift = shifts_in[key]["array"].load()
                        shift_weights = shifts_in[key]["weights"].load()
                        shift_DNNId = shifts_in[key]["DNNId"].load()
                        shifts_to_predict[key + "_scores" + str(i)] = model(torch.tensor(shift[shift_DNNId == 1 - 2 * i])).softmax(dim=1)
                        shifts_to_predict[key + "_weights" + str(i)] = shift_weights[shift_DNNId == 1 - 2 * i]

            scores.append(X_scores.numpy())
            labels.append(y_test)
            weights.append(weight_test)
            data_scores.update({i: data_pred.numpy()})
            QCD_scores.update({i: QCD_pred.numpy()})
            # for X_pred_batch, y_pred_batch, weight_pred_batch in pred_loader:
            # X_scores = model(X_pred_batch)

            # scores.append(X_scores.numpy())
            # labels.append(y_pred_batch)
            # weights.append(weight_pred_batch)

            # test_predict = reconstructed_model.predict(X_test)
            # y_predictions_0 = np.array(y_predictions[0])

            # data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=1000)
            # for X_data in data_loader:

        for dat in self.datasets_to_process:
            self.output()[dat].dump(np.concatenate((dats_to_predict[dat + "_scores0"], dats_to_predict[dat + "_scores1"])))
            self.output()[dat + "_weights"].dump(np.concatenate((dats_to_predict[dat + "_weights0"], dats_to_predict[dat + "_weights1"])))
        keep_note = self.output()["shifts"].keys()
        if self.do_shifts:
            for key in shifts_in.keys():
                self.output()["shifts"]["scores_" + key].dump(np.concatenate((shifts_to_predict[key + "_scores0"], shifts_to_predict[key + "_scores1"])))
                self.output()["shifts"]["weights_" + key].dump(np.concatenate((shifts_to_predict[key + "_weights0"], shifts_to_predict[key + "_weights1"])))
        #         keep_note.remove("scores_"+key)
        #         keep_note.remove("weights_"+key)
        # print(keep_note)

        averaged_data_scores = sum(list(data_scores.values())) / len(data_scores.values())
        averaged_QCD_scores = sum(list(QCD_scores.values())) / len(QCD_scores.values())
        self.output()["scores"].dump(np.concatenate(scores))
        self.output()["labels"].dump(np.concatenate(labels))
        self.output()["weights"].dump(np.concatenate(weights))
        self.output()["data"].dump(averaged_data_scores)
        self.output()["QCD_scores"].dump(averaged_QCD_scores)
        self.output()["QCD_weights"].dump(QCD_weights)


class CalcNormFactors(DNNTask):  # Requiring MergeArrays from a DNNTask leads to JSON errors
    # should be
    kfold = IntParameter(default=2)
    QCD_estimate = BoolParameter(default=False)

    def requires(self):
        out = {
            "samples": CrossValidationPrep.req(self, kfold=self.kfold),
            "models": PytorchCrossVal.req(self, kfold=self.kfold, category="All_Lep"),  # ,category="All_Lep" FIXME, this is trained on inclusive cuts,
            "scores": PredictDNNScores.req(self, workflow="local"),
        }
        if self.QCD_estimate:
            # preventing circular imports
            from tasks.bgestimation import EstimateQCDinSR

            out.update({"QCD_pred": EstimateQCDinSR.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "QCD", "Rare", "DY", "MET", "SingleMuon", "SingleElectron"])})
        return out

    def output(self):
        return self.local_target("Normalisation_factors.json")

    def store_parts(self):
        parts = tuple()
        if self.QCD_estimate:
            parts += ("QCD_est",)
        return (
            super(CalcNormFactors, self).store_parts()
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (self.gamma,)
            + parts
        )

    def run(self):
        models = self.input()["models"]["collection"].targets[0]
        samples = self.input()["samples"]
        inp_scores = self.input()["scores"]["collection"].targets[0]

        # loading predicted scores for EWK and data
        scores, labels, weights = [], [], []
        data_scores = {}
        data = self.input()["samples"]["data"].load()
        MC_scores = inp_scores["scores"].load()

        data_scores = inp_scores["data"].load()

        MC_labels = inp_scores["labels"].load()
        MC_labels = np.argmax(MC_labels, axis=-1)
        MC_pred = np.argmax(MC_scores, axis=-1)
        weights = inp_scores["weights"].load()

        # QCD
        # QCD_scores = inp_scores["QCD_scores"].load()
        # QCD_weights = inp_scores["QCD_weights"].load()
        # QCD_node_0 = QCD_weights[np.argmax(QCD_scores, axis=-1) == 0]
        # QCD_node_1 = QCD_weights[np.argmax(QCD_scores, axis=-1) == 1]
        # QCD_node_2 = QCD_weights[np.argmax(QCD_scores, axis=-1) == 2]
        QCD_node_0 = weights[(MC_labels == 2) & (MC_pred == 0)]
        QCD_node_1 = weights[(MC_labels == 2) & (MC_pred == 1)]
        QCD_node_2 = weights[(MC_labels == 2) & (MC_pred == 2)]
        if self.QCD_estimate:
            print("Using data driven QCD", self.input()["QCD_pred"].load())
            # if we calculate QCD beforehand, readjust yields
            QCD_bg_est = self.input()["QCD_pred"].load()["diff_data_EWK"]
            F_Sel_Anti = self.input()["QCD_pred"].load()["F_Sel_Anti"]
            QCD_node_0 = QCD_bg_est[0] * F_Sel_Anti
            QCD_node_1 = QCD_bg_est[1] * F_Sel_Anti
            QCD_node_2 = QCD_bg_est[2] * F_Sel_Anti

        # assigning nodes
        tt_node_0 = weights[(MC_labels == 0) & (MC_pred == 0)]
        tt_node_1 = weights[(MC_labels == 0) & (MC_pred == 1)]
        tt_node_2 = weights[(MC_labels == 0) & (MC_pred == 2)]

        WJets_node_0 = weights[(MC_labels == 1) & (MC_pred == 0)]
        WJets_node_1 = weights[(MC_labels == 1) & (MC_pred == 1)]
        WJets_node_2 = weights[(MC_labels == 1) & (MC_pred == 2)]

        data_node_0 = np.argmax(data_scores, axis=-1) == 0
        data_node_1 = np.argmax(data_scores, axis=-1) == 1
        data_node_2 = np.argmax(data_scores, axis=-1) == 2

        # constructing and solving linear equation system, QCD for now weighted with delta = 1
        # left_side = np.array([[np.sum(tt_node_0), np.sum(WJets_node_0)], [np.sum(tt_node_1), np.sum(WJets_node_1)]])
        # right_side = np.array([np.sum(data_node_0) - np.sum(QCD_node_0), np.sum(data_node_1) - np.sum(QCD_node_1)])
        left_side = np.array([[np.sum(tt_node_0), np.sum(WJets_node_0), np.sum(QCD_node_0)], [np.sum(tt_node_1), np.sum(WJets_node_1), np.sum(QCD_node_1)], [np.sum(tt_node_2), np.sum(WJets_node_2), np.sum(QCD_node_2)]])
        right_side = np.array([np.sum(data_node_0), np.sum(data_node_1), np.sum(data_node_2)])
        factors = np.linalg.solve(left_side, right_side)

        print("Left: ", left_side, "\nRight: ", right_side)

        # alpha * tt + beta * Wjets
        # factor_dict = {"alpha": factors[0], "beta": factors[1]}
        factor_dict = {"alpha": factors[0], "beta": factors[1], "delta": factors[2]}
        print(factor_dict)
        self.output().dump(factor_dict)


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


class OptimizeHyperParam(DNNTask, HTCondorWorkflow, law.LocalWorkflow):  # , CoffeaTask
    # define it here again so training can be started from here
    nParameters = IntParameter(default=1)
    # setting needed RAM high and time long
    RAM = 10000
    hours = 8

    def create_parameter_dict(self):
        parameter_dict = {}
        n_nodes = [16, 32, 64, 128, 256]
        dropouts = [0.01, 0.1, 0.2, 0.3, 0.5]
        batch_sizes = [16, 32, 64, 128, 256]
        learning_rates = [0.001, 0.0003, 0.0001, 0.00005, 0.00001]
        gammas = [0.75, 0.875, 1, 1.125, 1.25]
        branch_counter = 0
        for inode, node in enumerate(n_nodes[: self.nParameters]):
            for idrop, drop in enumerate(dropouts[: self.nParameters]):
                for ibatch, batch in enumerate(batch_sizes[: self.nParameters]):
                    for ilr, lr in enumerate(learning_rates[: self.nParameters]):
                        for igamma, gam in enumerate(gammas[: self.nParameters]):
                            parameter_dict[branch_counter] = "_".join([str(node), str(drop), str(batch), str(lr), str(gam)])
                            branch_counter += 1
        return parameter_dict

    def create_branch_map(self):
        # overwrite branch map
        parameter_dict = self.create_parameter_dict()
        return list(parameter_dict.keys())

    def requires(self):
        # return PrepareDNN.req(self)
        return {
            "data": CrossValidationPrep.req(self, kfold=self.kfold),
            "mean_std": ArrayNormalisation.req(self, datasets_to_process=["WJets", "SingleTop", "TTbar", "Rare", "DY", "T5qqqqVV", "MET", "SingleMuon", "SingleElectron"]),
        }

    def output(self):
        # one output with metrics for each configuration
        parameter_dict = self.create_parameter_dict()
        out = {branch: self.local_target(parameters + ".json") for branch, parameters in parameter_dict.items()}
        return out

    # def store_parts(self):
    #     # debug_str = ''
    #     if self.debug:
    #         debug_str = "debug"
    #     else:
    #         debug_str = ""

    #     # put hyperparameters in path to make an easy optimization search
    #     return (
    #         super(PytorchCrossVal, self).store_parts()
    #         # + (self.n_layers,)
    #         + (self.n_nodes,)
    #         + (self.dropout,)
    #         + (self.batch_size,)
    #         + (self.learning_rate,)
    #         + (debug_str,)
    #     )

    def calc_class_weights(self, y_train, y_weight, norm=1, sqrt=False):
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

        # n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())
        # total_weight = np.sum(y_weight)
        # class_weights = {}
        # for i in range(n_processes):
        #     mask = np.argmax(y_train, axis=-1) == i
        #     class_weights[i] = total_weight / np.sum(y_weight[mask])
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
        # problems with opening files on NAF
        torch.multiprocessing.set_sharing_strategy("file_system")
        # define dimensions, working with aux template for processes
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template")[self.category].keys())

        # definition for the normalization layer
        means, stds = (
            self.input()["mean_std"]["means_stds"].load()[0],
            self.input()["mean_std"]["means_stds"].load()[1],
        )

        performance = {}

        # train setup only on one fold to be faster
        i = self.branch % 2
        # only load needed data set config in each step
        X_train = self.input()["data"]["cross_val_{}".format(i)]["cross_val_X_train_{}".format(i)].load()
        y_train = self.input()["data"]["cross_val_{}".format(i)]["cross_val_y_train_{}".format(i)].load()
        weight_train = self.input()["data"]["cross_val_{}".format(i)]["cross_val_weight_train_{}".format(i)].load()
        X_val = self.input()["data"]["cross_val_{}".format(i)]["cross_val_X_val_{}".format(i)].load()
        y_val = self.input()["data"]["cross_val_{}".format(i)]["cross_val_y_val_{}".format(i)].load()
        weight_val = self.input()["data"]["cross_val_{}".format(i)]["cross_val_weight_val_{}".format(i)].load()

        class_weights = self.calc_class_weights(y_train, weight_train, sqrt=False)
        # get hyperparameters for branch
        parameter_dict = self.create_parameter_dict()
        print(parameter_dict[self.branch])
        n_nodes, dropout, batch_size, lr, gamma = parameter_dict[self.branch].split("_")
        # define data
        data_collection = util.DataModuleClass(
            X_train,
            y_train,
            weight_train,
            X_val,
            y_val,
            weight_val,
            int(batch_size),
            n_processes,
        )

        # declare model
        model = util.MulticlassClassification(
            num_feature=n_variables,
            num_class=n_processes,
            means=means,
            stds=stds,
            dropout=float(dropout),
            class_weights=torch.tensor(list(class_weights.values())),
            n_nodes=int(n_nodes),
            learning_rate=float(lr),
            gamma=float(gamma),
        )

        # Trainer, for gpu gpus=1
        # train for 10 epochs to see result
        trainer = pl.Trainer(
            max_epochs=self.epochs,
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

        performance.update(
            {
                "accuracy_stats": model.accuracy_stats,
                "loss_stats": model.loss_stats,
            }
        )

        self.output()[self.branch].parent.touch()
        self.output()[self.branch].dump(performance)


class FindBestTraining(DNNTask):
    nParameters = IntParameter(default=1)

    def requires(self):
        return OptimizeHyperParam.req(self, nParameters=self.nParameters)

    def output(self):
        return self.local_target("best_training.json")

    def find_highest_value(self, dic):
        # Initialize variables to track the max value, key, and index
        max_value = float("-inf")
        max_key = None
        max_index = None

        # Iterate through each key and its corresponding list
        for key, values in dic.items():
            # Find the maximum value and its index in the current list
            current_max = max(values)
            current_index = values.index(current_max)

            # Update if a new maximum is found
            if current_max > max_value:
                max_value = current_max
                max_key = key
                max_index = current_index

        # Print results
        print(f"The maximum value is {max_value} in list '{max_key}' at index {max_index}.")

    def run(self):
        from collections import defaultdict

        inp = self.input()["collection"].targets[0]
        # declaring stuff to fill in loop
        n_nodes = defaultdict(list)
        dropout = defaultdict(list)
        batch_size = defaultdict(list)
        learning_rates = defaultdict(list)
        gamma = defaultdict(list)
        val_loss, val_acc = [], []
        for branch in inp.keys():
            nod, drop, batch, lr, gam = inp[branch].basename.replace(".json", "").split("_")
            # print("\n",inp[branch].basename)
            stats = inp[branch].load()
            # # print latest entry for everything
            # print("train_acc", stats['accuracy_stats']['train'][-1])
            # print("val_acc", stats['accuracy_stats']['val'][-1])
            # print("train_loss", stats['loss_stats']['train'][-1])
            # print("val_loss", stats['loss_stats']['val'][-1])
            val_loss.append(stats["loss_stats"]["val"][-1])
            val_acc.append(stats["accuracy_stats"]["val"][-1])
            # filling each parameter with the respective values
            n_nodes[nod].append(stats["accuracy_stats"]["val"][-1])
            dropout[drop].append(stats["accuracy_stats"]["val"][-1])
            batch_size[batch].append(stats["accuracy_stats"]["val"][-1])
            learning_rates[lr].append(stats["accuracy_stats"]["val"][-1])
            gamma[gam].append(stats["accuracy_stats"]["val"][-1])

        # fill nans so that they don't mess up investigation
        val_loss = np.nan_to_num(val_loss, nan=np.max(val_loss))
        val_acc = np.nan_to_num(val_acc, nan=np.min(val_acc))
        print("\nmean acc", np.mean(val_acc))
        print("Max val acc {} at training {} with parameters {}".format(np.max(val_acc), np.argmax(val_acc), inp[np.argmax(val_acc)].basename))
        print("\nmean loss", np.mean(val_loss))
        print("Min val loss {} at training {} with parameters {}".format(np.min(val_loss), np.argmin(val_loss), inp[np.argmin(val_loss)].basename))

        ### respective values per hyperparam
        print("\nnodes")
        for val, acc in n_nodes.items():
            print(val, np.mean(acc))
        print("\ndropout")
        for val, acc in dropout.items():
            print(val, np.mean(acc))
        print("\nbatch_size")
        for val, acc in batch_size.items():
            print(val, np.mean(acc))
        print("\nlearning_rate")
        for val, acc in learning_rates.items():
            print(val, np.mean(acc))
        print("\ngamma")
        for val, acc in gamma.items():
            print(val, np.mean(acc))

        for dic in [n_nodes, dropout, batch_size, learning_rates, gamma]:
            self.find_highest_value(dic)

        # latex table
        print("\multicolumn{2}{c|}{$n_\\text{nodes}$}", " & ", "\multicolumn{2}{c|}{dropout}", " & ", "\multicolumn{2}{c|}{batch size}", " & ", "\multicolumn{2}{c|}{learning rate}", " & ", "\multicolumn{2}{c}{$\gamma$}", "\\\\ \hline")
        for i in range(self.nParameters):
            print(
                list(n_nodes.keys())[i],
                " & ",
                np.round(np.mean(n_nodes[list(n_nodes.keys())[i]]), 3),
                " & ",
                list(dropout.keys())[i],
                "  & ",
                np.round(np.mean(dropout[list(dropout.keys())[i]]), 3),
                " & ",
                list(batch_size.keys())[i],
                " & ",
                np.round(np.mean(batch_size[list(batch_size.keys())[i]]), 3),
                " & ",
                list(learning_rates.keys())[i],
                " & ",
                np.round(np.mean(learning_rates[list(learning_rates.keys())[i]]), 3),
                " & ",
                list(gamma.keys())[i],
                " & ",
                np.round(np.mean(gamma[list(gamma.keys())[i]]), 3),
                " \\\\",
            )

        from IPython import embed

        embed()
