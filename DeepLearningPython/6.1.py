# Word-level one-hot encoding (toy example)
import numpy as np

if __name__ == "__main__":
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_length = 10

    results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1
            print(sample)
            print(word)
            print(str(i) + " " + str(j) + " " + str(index))

    print(results)
