"""
    Simulates the frontend using probabilities.
"""
import json
from concurrent.futures._base import wait
from concurrent.futures.thread import ThreadPoolExecutor
from functools import reduce
from time import sleep, time

import pandas as pd
import numpy as np
import requests
import tqdm as tqdm
from numpy.random import choice
from uuid import uuid4
from loguru import logger

entity_counts = None
entity_count_path = './data/entity_counts.json'

rating_types = {-1, 0, 1}
# Do not change the API base unless you have a good reason to do so
api_base = 'http://localhost:5000/api'


def generate_token():
    return f'{uuid4()}+simulation'


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


def rate_entities(uris):
    return {uri: rate_entity(uri) for uri in uris}


def generate_feedback(questions, prediction=False):
    if prediction:
        feedback = {key: _generate_feedback(questions[key]) for key in questions.keys() if key != 'prediction'}
        keys = reduce(lambda a, b: set(a).intersection(set(b)), feedback.values())

        return {key: reduce(lambda a, b: a[key] + b[key], feedback.values()) for key in keys}
    else:
        return _generate_feedback(questions)


def _generate_feedback(questions):
    uri_ratings = rate_entities([question['uri'] for question in questions])
    category_uris = {cat: [uri for uri, rating in uri_ratings.items() if rating == cat] for cat in rating_types}

    return {
        'liked': category_uris[1],
        'disliked': category_uris[-1],
        'unknown': category_uris[0]
    }


class Simulation:
    def __init__(self):
        self.token = generate_token()

    def _headers(self):
        return {'Authorization': self.token}

    def _post(self, url, data):
        pass

    def _get(self, url):
        return requests.get(url, headers=self._headers())

    def _get_movies(self):
        return self._get(f'{api_base}/movies')

    def _feedback(self, feedback):
        return requests.post(f'{api_base}/feedback', json=feedback, headers=self._headers())

    def run(self):
        feedback = generate_feedback(self._get_movies().json())

        while questions := self._feedback(feedback).json():
            prediction = 'prediction' in questions
            feedback = generate_feedback(questions, prediction)

            if prediction:
                break


def run_simulation():
    Simulation().run()


if __name__ == '__main__':
    if int(requests.get(f'{api_base}/sessions').text):
        logger.error('There are existing sessions in the target API. Please clear those sessions before the simulation')
        exit(1)

    ratings = pd.read_csv('./data/ratings.csv', index_col=0)
    ratings = ratings.loc[:, ~ratings.columns.str.contains('^Unnamed')]

    executor = ThreadPoolExecutor(max_workers=10)
    futures = list()
    for i in range(1000):
        futures.append(executor.submit(run_simulation))

    for future in tqdm.tqdm(futures):
        future.result()
