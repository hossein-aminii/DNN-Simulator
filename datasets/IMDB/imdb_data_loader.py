from nltk.corpus import stopwords
import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle


class IMDBDatasetLoader:

    def __init__(
        self,
        filepath: str,
        load: bool = True,
        MAX_NB_WORDS=10000,
        MAX_SEQUENCE_LENGTH=128,
        EMBEDDING_DIM=32,
        tokenizer=None,
        name="A module to work with IMDB data and load them",
    ) -> None:

        self.name = name
        self.filepath = filepath
        self.stop_words = set(stopwords.words("english"))
        # The maximum number of words to be used. (most frequent)
        self.MAX_NB_WORDS = MAX_NB_WORDS
        # Max number of words in each complaint.
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH  # mean of sequence length is 126
        # This is fixed.
        self.EMBEDDING_DIM = EMBEDDING_DIM

        self.tokenizer = tokenizer

        self.train_dataset = None
        self.validation_dataset = None

        if load:
            self.load()

    def clean_text(self, text: str):
        text = text.lower()  # lowercase text
        text = " ".join(word for word in text.split() if word not in self.stop_words)  # remove stopwords from text
        text = re.sub("<.*?>", "", text)
        text = re.sub(r"\W", " ", text)  # Remove all the special characters
        text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)  # remove all single characters
        text = re.sub(r"\^[a-zA-Z]\s+", " ", text)  # Remove single characters from the start
        text = re.sub(r"\s+", " ", text, flags=re.I)  # Substituting multiple spaces with single space
        return text

    def save_tokenizer(self, tokenizer: Tokenizer):
        with open("tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def tokenizing(self, data: pd.DataFrame):
        if self.tokenizer is None:
            tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        else:
            tokenizer = self.tokenizer
        tokenizer.fit_on_texts(data["review"].values)
        word_index = tokenizer.word_index
        print("INFO: Found %s unique tokens." % len(word_index))
        X = tokenizer.texts_to_sequences(data["review"].values)
        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        print("INFO: Shape of input data tensor:", X.shape)
        # Converting categorical labels to numbers.
        Y = pd.get_dummies(data["sentiment"]).values
        self.save_tokenizer(tokenizer=tokenizer)
        return X, Y

    @staticmethod
    def train_test_split(X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"INFO: train dataset shapes -> x.shape: {X_train.shape}, y.shape: {Y_train.shape}")
        print(f"INFO: test dataset shapes -> x.shape: {X_test.shape}, y.shape: {Y_test.shape}")
        return (X_train, Y_train), (X_test, Y_test)

    def create_dataset(self, X, Y, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        return dataset

    @staticmethod
    def read_data_from_file(filepath: str):
        print(f"INFO: reading data from {filepath}")
        return pd.read_csv(filepath)

    def load(self):
        data_df = self.read_data_from_file(filepath=self.filepath)
        print("INFO: Statrt data preprocessing...")
        data_df["review"] = data_df["review"].apply(self.clean_text)
        text_values = data_df["review"].values
        min_words = 100000
        max_words = 0
        num_words = []
        for text in text_values:
            words = text.split()
            if len(words) < min_words:
                min_words = len(words)
            if len(words) > max_words:
                max_words = len(words)
            num_words.append(len(words))
        mean_words = float(sum(num_words)) / float(len(num_words))
        print(f"min words: {min_words}, max words: {max_words}, mean words: {mean_words}")
        X, Y = self.tokenizing(data=data_df)
        (x_train, y_train), (x_test, y_test) = self.train_test_split(X=X, Y=Y)
        print("INFO: Data preprocessing done successfully!")
        self.train_dataset = self.create_dataset(X=x_train, Y=y_train)
        self.validation_dataset = self.create_dataset(X=x_test, Y=y_test)
        print(f"INFO: train dataset shape -> num_batch: {len(self.train_dataset)}")
        print(f"INFO: validation dataset shape -> num_batch: {len(self.validation_dataset)}")
        print("INFO: Loading IMDB data done successfully!")
