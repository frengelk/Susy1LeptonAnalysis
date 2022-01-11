import law
import numpy as np
import pickle
import luigi.util
from luigi import BoolParameter, IntParameter, FloatParameter, ChoiceParameter
from law.target.collection import TargetCollection
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection as skm
from rich.console import Console

from tasks.basetasks import ConfigTask, HTCondorWorkflow
from tasks.arraypreparation import ArrayNormalisation

"""
DNN Stuff
"""

law.contrib.load("tensorflow")


class DNNTrainer(ConfigTask, HTCondorWorkflow, law.LocalWorkflow):

    channel = luigi.Parameter(default="0b", description="channel to train on")
    epochs = IntParameter(default=100)
    batch_size = IntParameter(default=10000)
    learning_rate = FloatParameter(default=0.01)
    debug = BoolParameter(default=False)
    n_layers = IntParameter(default=3)
    n_nodes = IntParameter(default=256)
    dropout = FloatParameter(default=0.2)
    norm_class_weights = FloatParameter(default=1)
    job_number = IntParameter(
        default=1, description="how many HTCondo jobs are started"
    )

    def create_branch_map(self):
        # overwrite branch map
        n = self.job_number
        return list(range(n))

        # return {i: i for i in range(n)}

    def requires(self):
        # return PrepareDNN.req(self)
        return ArrayNormalisation.req(self, channel="N0b_CR")

    def output(self):
        return {
            "saved_model": self.local_target("saved_model"),
            "history_callback": self.local_target("history.pckl"),
            # test data for plotting
            "test_data": self.local_target("test_data.npy"),
            "test_labels": self.local_target("test_labels.npy"),
            "test_acc": self.local_target("test_acc.json"),
        }

    def store_parts(self):
        # debug_str = ''
        if self.debug:
            debug_str = "debug"
        else:
            debug_str = ""

        # put hyperparameters in path to make an easy optimization search
        return (
            super(DNNTrainer, self).store_parts()
            + (self.analysis_choice,)
            + (self.channel,)
            + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (debug_str,)
        )

    def build_model_sequential(self, n_variables, n_processes):
        # try selu or elu https://keras.io/api/layers/activations/?
        # simple sequential model to start
        model = keras.Sequential(
            [
                # normalise input and keep the values
                keras.layers.BatchNormalization(
                    axis=1, trainable=False, input_shape=(n_variables,)
                ),
                keras.layers.Dense(256, activation=tf.nn.elu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256, activation=tf.nn.elu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256, activation=tf.nn.elu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(n_processes, activation=tf.nn.softmax),
            ]
        )

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

    def build_model_functional(
        self, n_variables, n_processes, n_layers=3, n_nodes=256, dropout=0.2
    ):
        inp = keras.Input((n_variables,))
        x = keras.layers.BatchNormalization(
            axis=1, trainable=False, input_shape=(n_variables,)
        )(inp)

        for i in range(n_layers):
            x = keras.layers.Dense(n_nodes, activation=tf.nn.relu)(x)
            x = keras.layers.Dropout(dropout)(x)
        # x = keras.layers.Dense(256, activation=tf.nn.relu)(x)
        # x = keras.layers.Dropout(0.2)(x)
        # x = keras.layers.Dense(256, activation=tf.nn.relu)(x)
        # x = keras.layers.Dropout(0.2)(x)

        out = keras.layers.Dense(n_processes, activation=tf.nn.softmax)(x)
        model = keras.models.Model(inputs=inp, outputs=out)

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

    def calc_norm_parameter(self, data):

        dat = np.swapaxes(data, 0, 1)
        means, stds = [], []
        for var in dat:
            means.append(var.mean())
            stds.append(var.std())

        return np.array(means), np.array(stds)

    def calc_class_weights(self, y_train, norm=1, sqrt=False):
        # calc class weights to battle imbalance
        # norm to tune down huge factors, sqrt to smooth the distribution
        from sklearn.utils import class_weight

        weight_array = norm * class_weight.compute_class_weight(
            "balanced",
            np.unique(np.argmax(y_train, axis=-1)),
            np.argmax(y_train, axis=-1),
        )

        if sqrt:
            # return dict(enumerate(np.sqrt(weight_array)))
            return dict(enumerate((weight_array) ** 0.9))
        if not sqrt:
            # set at minimum to 1.0
            # return dict(enumerate([a if a>1.0 else 1.0 for a in weight_array]))
            return dict(enumerate(weight_array))

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):

        # TENSORBOARD_PATH = (
        # self.output()["saved_model"].dirname
        # + "/logs/fit/"
        # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # )

        # all_processes = self.config_inst.get_aux("process_groups")["default"]

        # as luigi ints with default?
        batch_size = self.batch_size
        max_epochs = self.epochs

        # load data
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.processes) - 2  # substract data

        """
        train_data = np.load(self.input()[self.channel]["train"]["data"].path)
        train_labels = np.load(self.input()[self.channel]["train"]["label"].path)
        train_weights = np.load(self.input()[self.channel]["train"]["weight"].path)
        val_data = np.load(self.input()[self.channel]["validation"]["data"].path)
        val_labels = np.load(self.input()[self.channel]["validation"]["label"].path)
        test_data = np.load(self.input()[self.channel]["test"]["data"].path)
        test_labels = np.load(self.input()[self.channel]["test"]["label"].path)
        data = np.append(train_data, val_data, axis=0)
        data = np.append(data, test_data, axis=0)
        """

        # kinda bad solution, but I want to keep all processes as separate npy files
        arr_list = []

        for key in self.input().keys():
            arr_list.append((self.input()[key].load()))

        # dont concatenate, parallel to each other
        arr_conc = np.concatenate(list(a for a in arr_list[:-1]))
        labels = arr_list[-1]

        labels = np.swapaxes(labels, 0, 1)

        # split up test set 95
        Trainset, X_test, Trainlabel, y_test = skm.train_test_split(
            arr_conc, labels, test_size=0.05, random_state=42
        )

        # train and validation set 80:20
        X_train, X_val, y_train, y_val = skm.train_test_split(
            Trainset, Trainlabel, test_size=0.2, random_state=42
        )

        # configure the norm layer. Give it mu/sigma, layer is frozen
        # gamma, beta are for linear activations, so set them to unity transfo
        means, stds = self.calc_norm_parameter(arr_conc)
        print(means.shape)
        gamma = np.ones((n_variables,))
        beta = np.zeros((n_variables,))

        # initiliazemodel and set first layer
        model = self.build_model_functional(
            n_variables=n_variables,
            n_processes=n_processes,
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            dropout=self.dropout,
        )
        # model_seq = self.build_model_sequential(n_variables=n_variables, n_processes=n_processes)
        model.layers[1].set_weights([gamma, beta, means, stds])

        # display model summary
        model.summary()

        # define callbacks to be used during the training
        # tensorboard = keras.callbacks.TensorBoard(
        # log_dir=TENSORBOARD_PATH, histogram_freq=0, write_graph=True
        # )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=10, min_lr=0.001
        )
        stop_of = keras.callbacks.EarlyStopping(
            monitor="accuracy",  # val_accuracy
            verbose=1,
            min_delta=0.0,
            patience=20,
            restore_best_weights=True,  # for now
        )

        # calc class weights for training so all classes get some love
        # scale up by abundance, additional factor to tone the values down a little
        class_weights = self.calc_class_weights(
            y_train, norm=self.norm_class_weights, sqrt=False
        )
        print("\nclass weights", class_weights)

        # check if everything looks fines, exit by crtl+D
        if self.debug:
            self.output()["history_callback"].parent.touch()
            from IPython import embed

            embed()

        # plot schematic model graph
        self.output()["history_callback"].parent.touch()
        keras.utils.plot_model(
            #    model, to_file=self.output()["saved_model"].path + "/dnn_graph.png"
            model,
            to_file=self.output()["history_callback"].parent.path + "/dnn_scheme.png",
        )

        history_callback = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=max_epochs,
            verbose=2,
            callbacks=[stop_of, reduce_lr],  # tensorboard],
            # catch uneven input distributions
            class_weight=class_weights,
            # sample_weight=np.array(train_weights),
        )

        # save model
        self.output()["saved_model"].parent.touch()
        model.save(self.output()["saved_model"].path)

        # save callback for plotting
        with open(self.output()["history_callback"].path, "wb") as f:
            pickle.dump(history_callback.history, f)

        # test data
        self.output()["test_data"].dump(X_test)
        self.output()["test_labels"].dump(y_test)

        console = Console()
        # load test dta/labels and evaluate on unseen data
        test_loss, test_acc = model.evaluate(X_test, y_test)
        console.print(
            "\n[u][bold magenta]Test accuracy on channel {}:[/bold magenta][/u]".format(
                self.channel
            )
        )
        console.print(test_acc, "\n")
        self.output()["test_acc"].dump({"test_acc": test_acc})


class DNNHyperParameterOpt(ConfigTask, HTCondorWorkflow, law.LocalWorkflow):
    def create_branch_map(self):
        # overwrite branch map
        n = 1
        return list(range(n))

    def requires(self):
        # require DNN trainer with different configurations and monitor performance
        layers = [2, 3]  # ,3,4]
        nodes = [128, 256]  # ,256,512]
        dropout = [0.2, 0.3]  # ,0.4]
        grid_search = {
            "{}_{}_{}".format(lay, nod, drop): DNNTrainer.req(
                self,
                n_layers=lay,
                n_nodes=nod,
                dropout=drop,
                epochs=25,
                # workflow="local",
                # job_number=len(layers)*len(nodes)*len (dropout),
                # workers=len(layers)*len(nodes)*len(dropout),
            )
            for lay in layers
            for nod in nodes
            for drop in dropout
        }

        return grid_search

    def output(self):
        return self.local_target("performance.json")

    def store_parts(self):

        return super(DNNHyperParameterOpt, self).store_parts() + (self.analysis_choice,)

    def run(self):
        from IPython import embed

        # embed()

        performance = {}

        for key in self.input().keys():
            test_acc = self.input()[key]["test_acc"].load()["test_acc"]
            # ["collection"].targets[0]
            performance.update({key: test_acc})

        self.output().dump(performance)
