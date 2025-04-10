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
        available_neg = warm_item - set(pos_nb)
        neg_count = len(pos_nb)
        if len(available_neg) == 0:
            # Handle case where no negative items are available (rare)
            neg_nb = np.array([0] * neg_count)  # Default to 0 or skip this user
        elif neg_count > len(available_neg):
            # Cap negative samples to available population
            neg_nb = np.random.choice(list(available_neg), len(available_neg), replace=False)
            neg_nb = np.pad(neg_nb, (0, neg_count - len(available_neg)), mode='constant', constant_values=0)
        else:
            neg_nb = np.random.choice(list(available_neg), neg_count, replace=False)
        neg_item.extend(neg_nb)
    user_batch = np.repeat(warm_user, [len(user_nb_dict[u]) for u in warm_user])
    return np.stack([user_batch, pos_item, neg_item], axis=1)
