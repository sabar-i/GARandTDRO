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

def bpr_neg_samp(users, item_num, user_dict, warm_item):
    neg_items = []
    for u in users:
        pos_items = user_dict.get(u, [])
        for _ in range(len(pos_items)):
            while True:
                neg_id = np.random.randint(0, item_num)  # Ensure within item_num
                if neg_id not in pos_items and neg_id not in warm_item:
                    break
            neg_items.append(neg_id)
    return np.array(neg_items).reshape(-1, 1)
