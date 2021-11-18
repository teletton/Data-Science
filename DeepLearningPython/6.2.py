import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework']

characters = string.printable

token_index = dict(zip(characters, range(1, len(characters) + 1)))
print(token_index)
max_length = 50

results = np.zeros((len(samples), max_length, 101))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        print(index)
        results[i, j, index] = 1

print(results[0, 1])