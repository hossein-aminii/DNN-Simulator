from model import ModelDispatcher
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def get_model(info: dict):
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


def load_tokenizer(path: str):
    with open(path, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


model_info = {
    "type": "LSTM",
    "initialize": False,
    "filepath": "results\\models\\1_1\\IMDB_LSTM_Base_epoch#1.h5",  # nedded when initialize is False (load model from a file)
}

model = get_model(info=model_info)
print(model.summary())


def preprocess_single_sentence(sentence, tokenizer, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence


tokenizer_path = "results/models/1_1/tokenizer.pickle"
tokenizer = load_tokenizer(path=tokenizer_path)

test_sentence = "this id a Great movie!"
# test_sentence = "It is not good and not bad movie, but it does not worth to watch"
X_test = preprocess_single_sentence(test_sentence, tokenizer, max_sequence_length=250)

predictions = model.predict(X_test)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_class)
