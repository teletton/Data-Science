import string

samples = ['The cat sat on the mat.', 'The dog ate my homework']

characters = string.printable

token_index = dict(zip(range(1, len(characters) + 1), characters))

print(token_index)