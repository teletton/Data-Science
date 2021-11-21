from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

if __name__ == "__main__":
    max_features = 10000;
    maxlen = 20;

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    