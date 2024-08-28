import tensorflow as tf
from model import ModelDispatcher
from config import Config
from datasets import DataLoaderDispatcher
import json
import numpy as np


class CustomTrainer:

    def __init__(
        self,
        config: Config,
        name="A module to train a model using custom training loop",
    ) -> None:
        self.name = name
        self.config = config

        self.model = self.get_model(info=self.config.model_info)

        self.data_loader = self.get_data_loader(info=self.config.dataset_info)
        self.train_params: dict = self.config.train_params
        self.train_dataset = self.data_loader.train_dataset
        self.validation_dataset = self.data_loader.validation_dataset
        self.epochs = self.train_params["epochs"]

        self.loss_function = self.get_loss_function()
        self.optimizer = self.get_optimizer()

        self.history = {
            "train": {"per_step": {"loss": [], "accuracy": []}, "per_epoch": {"loss": [], "accuracy": []}},
            "validation": {"per_step": {"loss": [], "accuracy": []}, "per_epoch": {"loss": [], "accuracy": []}},
        }

    def get_loss_function(self):
        return tf.keras.losses.CategoricalCrossentropy()

    def get_optimizer(self):
        return tf.keras.optimizers.Adam()

    def get_model(self, info: dict):
        model_type = info["type"]
        dispatcher = ModelDispatcher(model_type=model_type)
        model_utils = dispatcher.dispatch()
        # --------------------------------------------------
        initialize = info["initialize"]
        if initialize:
            layers_info = info["layers_info"]
            return model_utils.crete_model(layers_info=layers_info)
        else:
            filepath = info["filepath"]
            return model_utils.load_model(filepath=filepath)

    def get_data_loader(self, info: dict):
        dataset = info["name"]
        file_path = info["filepath"]
        dispatcher = DataLoaderDispatcher(dataset=dataset)
        return dispatcher.dispatch(filepath=file_path)

    def train(self):
        print("INFO: Start training...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            # Initialize Progbar
            progbar = tf.keras.utils.Progbar(len(self.train_dataset))

            # Initialize metrics
            train_loss_metric = tf.keras.metrics.Mean()
            train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

            # Training loop
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch_train, training=True)
                    loss = self.loss_function(y_batch_train, y_pred)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Update metrics
                train_loss_metric.update_state(loss)
                train_accuracy_metric.update_state(y_batch_train, y_pred)

                # Update Progbar with current loss and accuracy
                progbar.update(
                    step + 1,
                    [
                        ("loss", train_loss_metric.result().numpy()),
                        ("accuracy", train_accuracy_metric.result().numpy()),
                    ],
                )
                self.history["train"]["per_step"]["loss"].append(float(train_loss_metric.result().numpy()))
                self.history["train"]["per_step"]["accuracy"].append(float(train_accuracy_metric.result().numpy()))

            self.history["train"]["per_epoch"]["loss"].append(float(train_loss_metric.result().numpy()))
            self.history["train"]["per_epoch"]["accuracy"].append(float(train_accuracy_metric.result().numpy()))
            # Validation loop
            progbar = tf.keras.utils.Progbar(len(self.validation_dataset))
            val_loss_metric = tf.keras.metrics.Mean()
            val_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
            for val_step, (x_batch_val, y_batch_val) in enumerate(self.validation_dataset):
                val_pred = self.model(x_batch_val, training=False)
                v_loss = self.loss_function(y_batch_val, val_pred)

                # Update validation metrics
                val_loss_metric.update_state(v_loss)
                val_accuracy_metric.update_state(y_batch_val, val_pred)

                progbar.update(
                    val_step + 1,
                    [
                        ("loss", val_loss_metric.result().numpy()),
                        ("accuracy", val_accuracy_metric.result().numpy()),
                    ],
                )
                self.history["validation"]["per_step"]["loss"].append(float(val_loss_metric.result().numpy()))
                self.history["validation"]["per_step"]["accuracy"].append(float(val_accuracy_metric.result().numpy()))

            self.history["validation"]["per_epoch"]["loss"].append(float(val_loss_metric.result().numpy()))
            self.history["validation"]["per_epoch"]["accuracy"].append(float(val_accuracy_metric.result().numpy()))

            # # Print the validation metrics at the end of each epoch
            # print(
            #     f" - val_loss: {val_loss_metric.result().numpy()} - val_accuracy: {val_accuracy_metric.result().numpy()}\n"
            # )
            # save model and history data
            self.model.save(f"IMDB_LSTM_Base_epoch#{epoch + 1}.h5")

            with open(f"IMDB_LSTM_Base_epoch#{epoch + 1}_Training_History.json", "w") as jf:
                json.dump(self.history, jf)
        # save model and history data
        self.model.save("IMDB_LSTM_Base_Final.h5")

        with open("IMDB_LSTM_Base_Final_Training_History.json", "w") as jf:
            json.dump(self.history, jf)

    def retrain(self, model, masks: list, inq_step: int):
        print("INFO: Start retraining...")
        masks_tf = [tf.convert_to_tensor(m) for m in masks]
        optimizer = self.get_optimizer()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            # Initialize Progbar
            progbar = tf.keras.utils.Progbar(len(self.train_dataset))

            # Initialize metrics
            train_loss_metric = tf.keras.metrics.Mean()
            train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

            # Training loop
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch_train, training=True)
                    loss = self.loss_function(y_batch_train, y_pred)

                gradients = tape.gradient(loss, model.trainable_variables)
                masked_gradients = [grad * mask for grad, mask in zip(gradients, masks_tf)]

                optimizer.apply_gradients(zip(masked_gradients, model.trainable_variables))

                # Update metrics
                train_loss_metric.update_state(loss)
                train_accuracy_metric.update_state(y_batch_train, y_pred)

                # Update Progbar with current loss and accuracy
                progbar.update(
                    step + 1,
                    [
                        ("loss", train_loss_metric.result().numpy()),
                        ("accuracy", train_accuracy_metric.result().numpy()),
                    ],
                )
            model.save(f"IMDB_LSTM_INQ_step#{inq_step}_epoch#{epoch + 1}.h5")
        return model
