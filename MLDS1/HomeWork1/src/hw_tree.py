import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node:
    def __init__(self, threshold, feature, value):
        self.left = None
        self.right = None
        self.entropy = None
        self.threshold = threshold
        self.feature = feature
        self.value

class Tree:
    def __init__(self, rand, get_candidate_columns, min_samples, data):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples
        self.data = data
        self.start_node = Node()

    def build(self, X, y, node):
        if len(X) < self.min_samples:
            return 
        
        
        
        
    def predict_instance(self, instance):
        pass

        
    def predict(self, X):
        y = []
        for instance in X:
            y.append(self.predict_instance(instance))
        return y
    

if __name__ == "__main__":
    data = pd.read_csv("./data/housing3.csv")

    for i in range(len(data['Class'])):
        if data['Class'][i] == "C1":
            data['Class'][i] = 1
        else:
            data['Class'][i] = 2
    
    print(data.head())