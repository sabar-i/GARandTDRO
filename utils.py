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
    train_data = []
    for u in users:
        pos_items = user_dict.get(u, [])
        for pos_i in pos_items:
            while True:
                neg_id = np.random.randint(0, item_num)  # Ensure within item_num
                if neg_id not in pos_items and neg_id not in warm_item:
                    break
            train_data.append([u, pos_i, neg_id])  # [user_id, positive_item, negative_item]
    return np.array(train_data, dtype=np.int64)
