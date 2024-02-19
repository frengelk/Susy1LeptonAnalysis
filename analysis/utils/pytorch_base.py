import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import MulticlassAccuracy
import numpy as np
import random

from time import time

"""
# help class for torch data loading
class ClassifierDataset(data.Dataset):
    def __init__(self, epoch, steps_per_epoch, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

        self.epoch = epoch
        self.iter = steps_per_epoch

    def __len__(self):
        #return len(self.X_data)
        return self.iter

    def __getitem__(self, idx):

        new_idx = idx + (self.iter*self.epoch)

         if new_idx >= len(self.X_data):
            new_idx = new_idx % len(self.X_data)

        return self.X_data[new_idx], self.y_data[new_idx]
"""


class ClassifierDataset(data.Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class ClassifierDatasetWeight(data.Dataset):
    def __init__(self, X_data, y_data, weight_data):  #
        self.X_data = X_data
        self.y_data = y_data
        self.weight_data = weight_data

    def __getitem__(self, index):
        # print(f"Index: {index}")
        # item = self.X_data[index]
        # print(f"Item type: {type(item)}")
        return self.X_data[index], self.y_data[index], self.weight_data[index]

    def __len__(self):
        return len(self.X_data)


# Custom dataset collecting all numpy arrays and bundles them for training
class DataModuleClass(pl.LightningDataModule):
    def __init__(self, X_train, y_train, weight_train, X_val, y_val, weight_val, batch_size, n_processes):  # , steps_per_epoch
        super().__init__()
        # define data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.weight_train = weight_train
        self.weight_val = weight_val
        # self.X_test=X_test , X_test, y_test
        # self.y_test=y_test
        # define parameters
        self.batch_size = batch_size
        self.n_processes = n_processes
        # self.steps_per_epoch = steps_per_epoch

    # def prepare_data(self):

    # Define steps that should be done
    # on only one GPU, like getting data.

    def setup(self, stage=None):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.
        # self.train_dataset = ClassifierDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float())
        self.train_dataset = ClassifierDatasetWeight(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float(), torch.from_numpy(self.weight_train).float())
        self.val_dataset = ClassifierDatasetWeight(torch.from_numpy(self.X_val).float(), torch.from_numpy(self.y_val).float(), torch.from_numpy(self.weight_val).float())

    def train_dataloader(self):
        # Use BalancedWeightedSampler directly here
        # sampler = BalancedWeightedSampler(weights=abs(self.weight_train),

        # setting all weights to positive
        weight_train = abs(self.weight_train)
        tot_weight = np.sum(weight_train)
        for i in range(len(self.y_train[0])):
            mask = self.y_train[:, i] == 1
            sum = np.sum(weight_train[mask])
            # overall weight tensor should yield num_classes
            # FIXME the sqrt is just a test np.sqrt(sum)
            weight_train[mask] /= sum  # * len(self.y_train[0]) / tot_weight

        # FIXME oversample signal, or undersample? undersampling seems to be better
        mask = self.y_train[:, -1] == 1
        # weight_train[mask] *= 1/1.5

        sampler = data.WeightedRandomSampler(weight_train, len(weight_train), replacement=True)
        # dataloader = data.DataLoader(
        #     dataset=self.train_dataset,
        #     sampler=sampler,
        #     batch_size=self.batch_size,
        #     num_workers=8,
        # )
        # FIXME train without sampling
        dataloader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

        return dataloader

    def val_dataloader(self):
        return data.DataLoader(
            dataset=self.val_dataset,
            batch_size=10 * self.batch_size,  # 10 *  , shuffle=True  # len(val_dataset
            num_workers=8,
        )

    # def test_dataloader(self):
    # return data.DataLoader(
    # dataset=self.test_dataset,
    # batch_size=10 * self.batch_size,
    # num_workers=8,  # , shuffle=True  # self.batch_size
    # )


# torch Multiclassifer
class MulticlassClassification(pl.LightningModule):  # nn.Module core.lightning.LightningModule
    def __init__(self, num_feature, num_class, means, stds, dropout, class_weights, n_nodes, learning_rate=1e-3):
        super(MulticlassClassification, self).__init__()

        # Attribute failure
        # self.prepare_data_per_node = True
        self.learning_rate = learning_rate

        # custom normalisation layer
        self.norm = NormalizeInputs(means, stds)

        self.layer_1 = nn.Linear(num_feature, n_nodes)  # // 2
        self.layer_2 = nn.Linear(n_nodes, n_nodes)  # // 2
        self.layer_3 = nn.Linear(n_nodes, n_nodes)
        self.layer_4 = nn.Linear(n_nodes, n_nodes)

        self.layer_out = nn.Linear(n_nodes, num_class)
        self.softmax = nn.Softmax(dim=1)  # log_
        self.class_weights = class_weights
        self.loss = nn.CrossEntropyLoss(reduction="mean")  # weight=class_weights,

        self.relu = nn.ELU()  # FIXME nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm1 = nn.BatchNorm1d(n_nodes)  # // 2
        self.batchnorm2 = nn.BatchNorm1d(n_nodes)
        self.batchnorm3 = nn.BatchNorm1d(n_nodes)
        self.batchnorm4 = nn.BatchNorm1d(n_nodes)
        self.accuracy = MulticlassAccuracy(num_classes=num_class)

        # define global curves
        self.accuracy_stats = {"train": [], "val": []}
        self.loss_stats = {"train": [], "val": []}
        self.epoch = 0
        self.loss_per_node = {"MC": [], "T5": [], "val_MC": [], "val_T5": [], "val_MC_weight": [], "val_T5_weight": []}
        self.signal_batch_fraction = []

        # defining outputs for later calls
        self.validation_step_outputs = []
        self.training_step_outputs = []

        # lazy timing
        self.start = time()

    def forward(self, x):
        x = self.norm(x)
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x = self.layer_4(x)
        # x = self.batchnorm4(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        x = self.layer_out(x)
        x = self.softmax(x)

        return x

    def validation_step(self, batch, batch_idx):
        x, y, weight = batch[0].squeeze(0), batch[1].squeeze(0), batch[2].squeeze(0)
        self.eval()
        logits = self(x)
        # loss = nn.functional.nll_loss()
        # loss = nn.CrossEntropyLoss()
        weighted_loss = self.weighted_loss(logits, y, weight)
        loss_step = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc_step = self.accuracy(preds, y.argmax(dim=1))

        self.validation_step_outputs.append({"val_loss": weighted_loss, "val_acc": acc_step})  # loss_step

        # print("val_loss", loss_step, "val_acc", acc_step)
        return {"val_loss": loss_step, "val_acc": acc_step, "weighted_val_loss": weighted_loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def weighted_loss(self, y, y_hat, weight):
        # loss should be equally important to all classes.
        loss_fn = nn.CrossEntropyLoss(reduction="none")  # weight=self.class_weights,
        resp_class_weights = torch.matmul(y_hat.float(), self.class_weights.float())
        mask_T5 = y_hat[:, -1] == 1
        mask_MC = y_hat[:, -1] == 0
        loss = loss_fn(y, y_hat)
        loss_weight = loss * weight * resp_class_weights
        # print("MC_loss sum", sum(loss[mask_MC]))
        # print("T5_loss sum", sum(loss[mask_T5]))
        self.loss_per_node["val_MC"].append(loss[mask_MC].mean())
        self.loss_per_node["val_T5"].append(loss[mask_T5].mean())
        self.loss_per_node["val_MC_weight"].append(loss_weight[mask_MC].mean())
        self.loss_per_node["val_T5_weight"].append(loss_weight[mask_T5].mean())
        return (loss_weight).mean()

    def training_step(self, batch, batch_idx):
        x, y, weight = batch[0].squeeze(0), batch[1].squeeze(0), batch[2].squeeze(0)
        self.train()
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc_step = self.accuracy(preds, y.argmax(dim=1))
        # fraction of signal events
        frac = sum(y[:, -1] == 1) / len(y)
        self.signal_batch_fraction.append(frac)
        # maybe we do this and a softmax layer at the end
        # nll_loss = nn.functional.nll_loss(logits, y)
        loss_step = self.loss(logits, y)
        # loss_step = self.weighted_loss(logits, y, weight)
        # not necessary here, done during ? optimizer I guess
        # loss_step.backward(retain_graph=True)
        self.log("train_loss", loss_step, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc_step, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # HAS to be called loss!!!
        self.training_step_outputs.append({"loss": loss_step, "acc": acc_step})
        return {"loss": loss_step, "acc": acc_step}

    def on_train_epoch_end(self):  # , outputs
        # aggregating information over complete training
        acc_mean = np.mean([out["acc"].item() for out in self.training_step_outputs])
        loss_mean = np.mean([out["loss"].item() for out in self.training_step_outputs])

        # save epoch wise metrics for later
        self.loss_stats["train"].append(loss_mean)
        self.accuracy_stats["train"].append(acc_mean)
        self.epoch += 1
        print(
            "Epoch:",
            self.epoch,
            " time:",
            np.round(time() - self.start, 5),
            "\ntrain loss acc:",
            loss_mean,
            acc_mean,
        )
        self.training_step_outputs.clear()  # free memory
        # Has to return NONE
        # return outputs

    def on_validation_epoch_end(self):  # , outputs)
        # average over batches, and save extra computed values
        loss_mean = np.mean([out["val_loss"].item() for out in self.validation_step_outputs])
        acc_mean = np.mean([out["val_acc"].item() for out in self.validation_step_outputs])
        # save epoch wise metrics for later
        self.loss_stats["val"].append(loss_mean)
        self.accuracy_stats["val"].append(acc_mean)

        print("val loss acc:", loss_mean, acc_mean)

        # epoch_average = torch.stack(self.validation_step_outputs).mean()
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss_mean, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc_mean, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # def get_metrics(self):
    # # don't show the version number, does not really seem to work
    # # maybe do it in the progressbar
    # items = super().get_metrics()
    # items.pop("v_num", None)
    # return items


class NormalizeInputs(nn.Module):
    def __init__(self, means, stds):
        super(NormalizeInputs, self).__init__()
        self.mean = torch.tensor(means)
        self.std = torch.tensor(stds)

    def forward(self, input):
        x = input - self.mean
        x = x / self.std
        # cast x to correct dtype
        x = x.to(torch.float32)
        return x


class EventBatchSampler(data.Sampler):
    def __init__(self, y_data, batch_size, n_processes, steps_per_epoch):
        self.y_data = y_data
        self.batch_size = batch_size
        self.n_processes = n_processes
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        # return len(self.y_data)
        return (self.batch_size // self.n_processes) * self.n_processes

    def __iter__(self):
        sub_batch_size = self.batch_size // self.n_processes
        # arr_list = []
        for i in range(self.steps_per_epoch):
            try:
                batch_counter
            except:
                batch_counter = [0, 0, 0]

            indices_for_batch = []
            for j in range(self.n_processes):
                batch_counter[j] += 1

                # get correct indices for process
                check = self.y_data[:, j] == 1

                # prohibit creating batch if sample space is over
                if batch_counter[j] * sub_batch_size > np.sum(check):
                    batch_counter[j] = 0

                indices = np.where(check)[0]

                # return random indices for each process in same amount
                # choice = np.random.choice(
                #    np.arange(0, len(indices), 1), size=sub_batch_size, replace=False
                # )
                # indices_for_batch.append(indices[choice])

                # append next sliced batch
                indices_for_batch.append(indices[sub_batch_size * batch_counter[j] : sub_batch_size * (batch_counter[j] + 1)])

                # check[indices[choice]] all True

            # shuffle indices so network does not get all events from one category in a big chunk
            array = np.concatenate(indices_for_batch)
            np.random.shuffle(array)
            yield array

            # arr_list.append(array)
        # a = np.nditer(np.concatenate(arr_list))
        # return a


class MyPrintingCallback(Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("do something when training ends")
