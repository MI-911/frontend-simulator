"""
    Simulates the frontend using probabilities.
"""
import json

import pandas as pd
import numpy as np
from numpy.random import choice


entity_counts = None
entity_count_path = './data/entity_counts.json'


def rate_entity(uri):
    global entity_counts

    if entity_counts is None:
        with open(entity_count_path, 'r') as f:
            entity_counts = json.load(f)

    if uri not in entity_counts:
        return 0

    counts = entity_counts[uri]
    weights = np.array(list(counts.values()))
    s = sum(weights)
    return choice([1, -1, 0], p=weights/s)


if __name__ == '__main__':
    ratings = pd.read_csv('./data/ratings.csv', index_col=0)
    ratings = ratings.loc[:, ~ratings.columns.str.contains('^Unnamed')]

    correct = 0
    for _, (_, uri, _, sentiment) in ratings.iterrows():
        pred = rate_entity(uri)
        if sentiment == pred:
            correct += 1

    print(f'Acc: {correct / len(ratings)}')
