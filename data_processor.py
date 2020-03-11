"""
    Process the MindReader data and stores probabilities for all entities.
"""
import json
from collections import Counter

import pandas as pd


def calculate_probabilities(input_path, output_path, entity_file, ratings_file):
    # Load data
    entities_df = pd.read_csv(input_path + entity_file)

    ratings = pd.read_csv(input_path + ratings_file, index_col=0)
    ratings = ratings.loc[:, ~ratings.columns.str.contains('^Unnamed')]

    entity_types = set()
    entities = set()
    entity_type_map = {}
    entity_super_types = set()

    # Find all entity types, entities and map entities to type.
    for _, (uri, labels) in entities_df[['uri', 'labels']].iterrows():
        if uri not in entity_type_map:
            entity_type_map[uri] = set()
        entities.add(uri)
        labels = labels.split('|')
        entity_super_types.add(labels[0])
        for label in labels:
            entity_types.add(label)
            entity_type_map[uri].add(label)

    # Method for increasing count per sentiment.
    def count_up(dictionary, label, sent):
        if sent == 1:
            dictionary[label]['l'] += 1
        elif sent == -1:
            dictionary[label]['d'] += 1
        else:
            dictionary[label]['u'] += 1

    entity_count = {uri: {'l': 0, 'd': 0, 'u': 0} for uri in entities}
    type_count = {label: {'l': 0, 'd': 0, 'u': 0} for label in entity_types}

    # Find number of like, dislike, dont know, for each entity and entity type
    for _, (_, uri, _, sentiment) in ratings.iterrows():
        if uri not in entities:
            continue

        count_up(entity_count, uri, sentiment)

        for label in entity_type_map[uri]:
            count_up(type_count, label, sentiment)

    for uri, counts in entity_count.items():
        cumulative = sum(counts.values())

        if cumulative > 5:
            continue

        # Use super type entity has more than one subtype
        types = list(entity_type_map[uri])
        super_type = [label for label in types if label in entity_super_types][0]
        types.remove(super_type)
        if len(types) != 1:
            real_type = super_type
        else:
            real_type = types[0]

        entity_count[uri] = type_count[real_type]

    with open(output_path + 'entity_counts.json', 'w') as f:
        json.dump(entity_count, f)


if __name__ == '__main__':
    calculate_probabilities('./data/', './data/', 'entities.csv', 'ratings.csv')
