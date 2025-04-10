# utils.py
import numpy as np
import tensorflow as tf
import random

def set_seed_tf(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # TF 2.x

class Timer:
    def __init__(self, name='timer'):
        self.name = name

    def logging(self, content):
        print(f"{self.name}: {content}")

def bpr_neg_samp(warm_user, n_samp, user_nb_dict, warm_item):
    pos_item = []
    neg_item = []
    warm_item = set(warm_item)
    for u in warm_user:
        pos_nb = user_nb_dict[u]
        pos_item.extend(pos_nb)
        neg_nb = np.random.choice(list(warm_item - set(pos_nb)), len(pos_nb), replace=False)
        neg_item.extend(neg_nb)
    user_batch = np.repeat(warm_user, [len(user_nb_dict[u]) for u in warm_user])
    return np.stack([user_batch, pos_item, neg_item], axis=1)
