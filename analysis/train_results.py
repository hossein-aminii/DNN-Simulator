import os
import json
import tensorflow as tf
from datasets import DataLoaderDispatcher
import copy
import pickle


def read_json(path):
    with open(path, "r") as jf:
        return json.load(jf)


def read_pickle(path):
    with open(path, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def get_data_loader(info: dict, params: dict):
    dataset = info["name"]
    file_path = info["filepath"]
    dispatcher = DataLoaderDispatcher(dataset=dataset)
    return dispatcher.dispatch(filepath=file_path, params=params)


def get_loss_function():
    return tf.keras.losses.CategoricalCrossentropy()


def inference(model, dataset):
    progbar = tf.keras.utils.Progbar(len(dataset))
    loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    loss_function = get_loss_function()
    for step, (x_batch, y_batch) in enumerate(dataset):
        pred = model(x_batch, training=False)
        loss = loss_function(y_batch, pred)

        # Update validation metrics
        loss_metric.update_state(loss)
        accuracy_metric.update_state(y_batch, pred)

        progbar.update(
            step + 1,
            [
                ("loss", loss_metric.result().numpy()),
                ("accuracy", accuracy_metric.result().numpy()),
            ],
        )
    return float(loss_metric.result().numpy()), float(accuracy_metric.result().numpy())


results_directory = "results\\models"
expriments_directory_name = os.listdir(results_directory)
experiments_directories = []
for d in expriments_directory_name:
    experiments_directories.append(os.path.join(results_directory, d))

experiments_data = {}
epochs = 5

model_file_names = [{"epoch": epoch + 1, "filename": f"IMDB_LSTM_Base_epoch#{epoch + 1}.h5"} for epoch in range(epochs)]
model_file_names.append({"epoch": 6, "filename": "IMDB_LSTM_Base_Final.h5"})
config_filename = "config.json"

# ----------------------------------------------------------------------------
dataset_info = {"name": "IMDB", "filepath": "datasets\\IMDB\\data\\IMDB-Dataset.csv"}
# ----------------------------------------------------------------------------
for idx, d in enumerate(experiments_directories):

    configs = read_json(path=os.path.join(d, config_filename))
    print(f"INFO: start loading data for exp# {idx + 1}")
    tokenizer = read_pickle(os.path.join(d, "tokenizer.pickle"))
    params = {
        "MAX_NB_WORDS": configs["embedding"]["input_dim"],
        "MAX_SEQUENCE_LENGTH": configs["embedding"]["input_length"],
        "EMBEDDING_DIM": configs["embedding"]["output_dim"],
        "tokenizer": tokenizer,
    }
    data_loader = get_data_loader(info=dataset_info, params=params)
    train_dataset = data_loader.train_dataset
    validation_dataset = data_loader.validation_dataset
    print(f"INFO: loading data for exp# {idx + 1} done successfully!")

    this_exp_data = copy.deepcopy(configs)
    for model_info in model_file_names:
        epoch = model_info["epoch"]
        model_path = os.path.join(d, model_info["filename"])
        model = tf.keras.models.load_model(model_path)
        print(f"INFO: start inference on train data set on model {model_path}")
        train_loss, train_accuracy = inference(model=model, dataset=train_dataset)
        print(f"INFO: start inference on validation data set on model {model_path}")
        validation_loss, validation_accuracy = inference(model=model, dataset=validation_dataset)
        print(f"INFO: inference on model {model_path} done successfully")
        print("-" * 80)
        this_exp_data[f"train_results_epoch#{epoch}"] = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": validation_loss,
            "val_accuracy": validation_accuracy,
        }
    experiments_data[f"exp#{idx + 1}"] = this_exp_data

with open("train_results.json", "w") as jf:
    json.dump(experiments_data, jf)
