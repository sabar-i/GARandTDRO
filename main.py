# main.py
import os
from metric import ndcg
import utils
import time
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from pprint import pprint
from GAR.GAR import GAR
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--datadir', type=str, default="/kaggle/input/amazon")
parser.add_argument('--dataset', type=str, default="Amazon")
parser.add_argument('--val_interval', type=float, default=1)
parser.add_argument('--val_start', type=int, default=0)
parser.add_argument('--test_batch_us', type=int, default=200)
parser.add_argument('--Ks', nargs='?', default='[20]')
parser.add_argument('--n_test_user', type=int, default=2000)
parser.add_argument('--embed_meth', type=str, default='ncf')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--train_set', type=str, default='map')
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--restore', type=str, default="")
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--model', type=str, default='GAR')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.6)
args = parser.parse_args([])
args.datadir = "/kaggle/input/amazon"
args.Ks = eval(args.Ks)
args.model = args.model.upper()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
utils.set_seed_tf(args.seed)
pprint(vars(args))
timer = utils.Timer(name='main')
ndcg.init(args)

# Load dataset
data_dir = args.datadir
content_data = np.load(os.path.join(data_dir, 'all_item_feature.npy'))
content_data = np.concatenate([np.zeros([1, content_data.shape[-1]]), content_data], axis=0)
emb = np.load(os.path.join(data_dir, 'all_item_embedding.npy'))
user_map = np.load(os.path.join(data_dir, 'user_map.npy'), allow_pickle=True).item()
item_map = np.load(os.path.join(data_dir, 'item_map.npy'), allow_pickle=True).item()
warm_items_raw = np.load(os.path.join(data_dir, 'warm_item.npy'), allow_pickle=True)
cold_items_raw = np.load(os.path.join(data_dir, 'cold_item.npy'), allow_pickle=True)

# Extract content from 0-dimensional arrays
warm_items_set = warm_items_raw.item() if warm_items_raw.shape == () else warm_items_raw
cold_items_set = cold_items_raw.item() if cold_items_raw.shape == () else cold_items_raw

# Debug: Check types and contents
print(f"warm_items_raw type: {type(warm_items_raw)}, content: {warm_items_set if hasattr(warm_items_set, '__len__') else [warm_items_set]}")
print(f"item_map type: {type(item_map)}, first 10 keys: {list(item_map.keys())[:10]}")

# Apply CLCRec+TDRO offset (num_user = 21607 for Amazon)
num_user = 21607
num_item = 93755  # From CLCRec+TDRO Amazon config
max_valid_index = num_item - 1  # 93754
warm_items = np.array([i + num_user for i in warm_items_set if 0 <= i < max_valid_index], dtype=np.int64)
cold_items = np.array([i + num_user for i in cold_items_set if 0 <= i < max_valid_index], dtype=np.int64)

# Debug: Check mapping results
print(f"Mapped warm_items: {warm_items[:10]} (length: {len(warm_items)})")
if len(warm_items) == 0:
    print(f"Warning: warm_items is empty after mapping. Raw items: {list(warm_items_set)[:10] if hasattr(warm_items_set, '__len__') else [warm_items_set]}")
    warm_items = np.array([num_user])  # Fallback to first offset item
elif len(warm_items) < 3:
    print(f"Warning: warm_items has {len(warm_items)} samples, less than n_clusters=3. Using all available.")

# Load and remap training_dict and other dictionaries with offset
training_dict = np.load(os.path.join(data_dir, 'training_dict.npy'), allow_pickle=True).item()
validation_cold_dict = np.load(os.path.join(data_dir, 'validation_cold_dict.npy'), allow_pickle=True).item()
validation_warm_dict = np.load(os.path.join(data_dir, 'validation_warm_dict.npy'), allow_pickle=True).item()
testing_cold_dict = np.load(os.path.join(data_dir, 'testing_cold_dict.npy'), allow_pickle=True).item()
testing_warm_dict = np.load(os.path.join(data_dir, 'testing_warm_dict.npy'), allow_pickle=True).item()
interaction_timestamp_dict = np.load(os.path.join(data_dir, 'interaction_timestamp_dict.npy'), allow_pickle=True).item()

def remap_items_dict(dict_data):
    return {k: [i + num_user for i in v if 0 <= i < max_valid_index] for k, v in dict_data.items()}

training_dict = remap_items_dict(training_dict)
validation_cold_dict = remap_items_dict(validation_cold_dict)
validation_warm_dict = remap_items_dict(validation_warm_dict)
testing_cold_dict = remap_items_dict(testing_cold_dict)
testing_warm_dict = remap_items_dict(testing_warm_dict)

# Validate para_dict before passing to bpr_neg_samp
para_dict = {
    'user_array': np.arange(len(user_map)),
    'item_array': np.arange(num_user, num_user + num_item),
    'warm_item': warm_items,
    'cold_item': cold_items,
    'warm_user': np.array(list(training_dict.keys())),
    'emb_user_nb': training_dict,
    'pos_user_nb': training_dict,
    'cold_val_user': np.array(list(validation_cold_dict.keys())),
    'cold_val_user_nb': validation_cold_dict,
    'warm_test_user': np.array(list(testing_warm_dict.keys())),
    'warm_test_user_nb': testing_warm_dict,
    'cold_test_user': np.array(list(testing_cold_dict.keys())),
    'cold_test_user_nb': testing_cold_dict,
    'hybrid_test_user': np.array(list(testing_warm_dict.keys()) + list(testing_cold_dict.keys())),
    'hybrid_test_user_nb': {**testing_warm_dict, **testing_cold_dict}
}

user_node_num = len(user_map) + 1
item_node_num = num_user + num_item  # 115361 for Amazon
user_emb = emb[:user_node_num]
item_emb = emb[user_node_num:user_node_num + item_node_num]
timer.logging('Data loaded from {}'.format(data_dir))

# TDRO Preprocessing
K = 3
E = 3
warm_features = content_data[warm_items - num_user]  # Adjust for original IDs
if len(warm_features) == 0:
    print("Warning: warm_features is empty. Skipping clustering or using fallback.")
    group_labels = np.zeros(len(warm_items), dtype=int)  # Fallback: assign all to cluster 0
elif len(warm_features) < K:
    print(f"Warning: warm_features has {len(warm_features)} samples, less than n_clusters={K}. Reducing clusters to {len(warm_features)}.")
    kmeans = KMeans(n_clusters=len(warm_features), random_state=args.seed, n_init='auto')
    group_labels = kmeans.fit_predict(warm_features)
else:
    kmeans = KMeans(n_clusters=K, random_state=args.seed, n_init='auto')
    group_labels = kmeans.fit_predict(warm_features)
group_map = {int(item): label for item, label in zip(warm_items, group_labels)}

# Handle empty interactions
interactions = []
for uid, items in training_dict.items():
    timestamps = interaction_timestamp_dict.get(uid, [0] * len(items))
    for iid, ts in zip(items, timestamps):
        interactions.append((uid, iid, ts))
interactions = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'timestamp'])
interactions = interactions.sort_values('timestamp')
if len(interactions) > 0:
    period_size = len(interactions) // E
    periods = [interactions.iloc[i:i + period_size] for i in range(0, len(interactions), period_size)]
else:
    print("Warning: interactions is empty. Skipping TDRO period division.")
    periods = []  # Empty list to avoid range error

def get_exclude_pair(u_pair, ts_nei):
    pos_item = np.array(sorted(list(set(para_dict['pos_user_nb'][u_pair[0]]) - set(ts_nei[u_pair[0]]))), dtype=np.int64)
    pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
    return np.stack([pos_user, pos_item], axis=1)

def get_exclude_pair_count(ts_user, ts_nei, batch):
    exclude_pair_list = []
    exclude_count = [0]
    for i, beg in enumerate(range(0, len(ts_user), batch)):
        end = min(beg + batch, len(ts_user))
        batch_user = ts_user[beg:end]
        batch_range = list(range(end - beg))
        batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))
        exclude_pair = list(map(lambda x: get_exclude_pair(x, ts_nei), batch_u_pair))
        exclude_pair = np.concatenate(exclude_pair, axis=0)
        exclude_pair_list.append(exclude_pair)
        exclude_count.append(exclude_count[i] + len(exclude_pair))
    return [np.concatenate(exclude_pair_list, axis=0), exclude_count]

exclude_val_cold = get_exclude_pair_count(para_dict['cold_val_user'][:args.n_test_user], para_dict['cold_val_user_nb'], args.test_batch_us)
exclude_test_warm = get_exclude_pair_count(para_dict['warm_test_user'][:args.n_test_user], para_dict['warm_test_user_nb'], args.test_batch_us)
exclude_test_cold = get_exclude_pair_count(para_dict['cold_test_user'][:args.n_test_user], para_dict['cold_test_user_nb'], args.test_batch_us)
exclude_test_hybrid = get_exclude_pair_count(para_dict['hybrid_test_user'][:args.n_test_user], para_dict['hybrid_test_user_nb'], args.test_batch_us)

# Model setup
model = GAR(emb.shape[-1], content_data.shape[-1], args)
save_dir = '/kaggle/working/GAR/model_save/'
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir + args.dataset + '-' + args.model + '-'
param_file = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
save_file = save_path + param_file
args.param_file = param_file

# Training loop
patience_count = 0
va_metric_max = 0
train_time = 0
val_time = 0
batch_count = 0
item_index = np.arange(item_node_num)

timer.logging("Training model...")
for epoch in range(1, args.max_epoch + 1):
    train_input = utils.bpr_neg_samp(para_dict['warm_user'], item_node_num, para_dict['emb_user_nb'], para_dict['warm_item'], num_user)
    # Debug: Check train_input shape and content
    print(f"train_input shape: {train_input.shape}, sample: {train_input[:5]}")
    if train_input.size > 0:
        invalid_indices = np.any((train_input < num_user) | (train_input >= item_node_num), axis=1)
        if np.any(invalid_indices):
            print(f"Warning: Invalid indices found in train_input: {train_input[invalid_indices]}")
            train_input = train_input[~invalid_indices]  # Filter out invalid indices
        if train_input.shape[1] < 3:
            print(f"Warning: train_input has {train_input.shape[1]} columns, expected 3. Padding with zeros.")
            train_input = np.pad(train_input, ((0, 0), (0, 3 - train_input.shape[1])), mode='constant')
    n_batch = len(train_input) // args.batch_size if len(train_input) > 0 else 0
    for beg in range(0, len(train_input) - args.batch_size, args.batch_size):
        end = beg + args.batch_size
        batch_count += 1
        t_train_begin = time.time()
        batch_lbs = train_input[beg:end]
        # Clip or validate item indices to prevent out-of-bounds errors
        batch_lbs[:, 1] = np.clip(batch_lbs[:, 1], num_user, item_node_num - 1)  # Positive items
        batch_lbs[:, 2] = np.clip(batch_lbs[:, 2], num_user, item_node_num - 1)  # Negative items
        # Debug: Verify clipped indices
        if np.any(batch_lbs[:, 1] >= item_node_num) or np.any(batch_lbs[:, 2] >= item_node_num):
            print(f"Warning: Clipping failed for batch_lbs: {batch_lbs[batch_lbs[:, 1] >= item_node_num]}")
        batch_group_ids = np.array([group_map.get(int(item), 0) for item in batch_lbs[:, 1]])
        period_grads = np.zeros(model.emb_dim)

        d_loss = model.train_d(user_emb[batch_lbs[:, 0]], item_emb[batch_lbs[:, 1] - num_user], item_emb[batch_lbs[:, 2] - num_user],
                               content_data[batch_lbs[:, 1] - num_user], batch_group_ids, period_grads)
        g_loss = model.train_g(user_emb[batch_lbs[:, 0]], item_emb[batch_lbs[:, 1] - num_user], content_data[batch_lbs[:, 1] - num_user],
                               batch_group_ids, period_grads)
        loss = sum(d_loss + g_loss)
        t_train_end = time.time()
        train_time += t_train_end - t_train_begin

        if (batch_count % int(n_batch * args.val_interval) == 0) and (epoch >= args.val_start):
            t_val_begin = time.time()
            gen_user_emb = model.get_user_emb(user_emb)
            gen_item_emb = model.get_item_emb(content_data, item_emb, para_dict['warm_item'], para_dict['cold_item'])
            va_metric, _ = ndcg.test(model.get_ranked_rating,
                                     lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                                     ts_nei=para_dict['cold_val_user_nb'],
                                     ts_user=para_dict['cold_val_user'][:args.n_test_user],
                                     masked_items=para_dict['warm_item'],
                                     exclude_pair_cnt=exclude_val_cold)
            va_metric_current = va_metric['ndcg'][0]
            if va_metric_current > va_metric_max:
                va_metric_max = va_metric_current
                model.save_weights(save_file)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count > args.patience:
                    break
            t_val_end = time.time()
            val_time += t_val_end - t_val_begin
            timer.logging('Epo%d(%d/%d) Loss:%.4f|va_metric:%.4f|Best:%.4f|Time_Tr:%.2fs|Val:%.2fs' %
                          (epoch, patience_count, args.patience, loss, va_metric_current, va_metric_max, train_time, val_time))

# Testing
model.load_weights(save_file)
gen_user_emb = model.get_user_emb(user_emb)
gen_item_emb = model.get_item_emb(content_data, item_emb, para_dict['warm_item'], para_dict['cold_item'])
cold_res, _ = ndcg.test(model.get_ranked_rating,
                        lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                        ts_nei=para_dict['cold_test_user_nb'],
                        ts_user=para_dict['cold_test_user'][:args.n_test_user],
                        masked_items=para_dict['warm_item'],
                        exclude_pair_cnt=exclude_test_cold)
timer.logging('Cold-start result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}'.format(
    args.Ks[0], cold_res['precision'][0], cold_res['recall'][0], cold_res['ndcg'][0]))

warm_res, _ = ndcg.test(model.get_ranked_rating,
                        lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                        ts_nei=para_dict['warm_test_user_nb'],
                        ts_user=para_dict['warm_test_user'][:args.n_test_user],
                        masked_items=para_dict['cold_item'],
                        exclude_pair_cnt=exclude_test_warm)
timer.logging("Warm result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], warm_res['precision'][0], warm_res['recall'][0], warm_res['ndcg'][0]))

hybrid_res, _ = ndcg.test(model.get_ranked_rating,
                          lambda u: model.get_user_rating(u, item_index, gen_user_emb, gen_item_emb),
                          ts_nei=para_dict['hybrid_test_user_nb'],
                          ts_user=para_dict['hybrid_test_user'][:args.n_test_user],
                          masked_items=None,
                          exclude_pair_cnt=exclude_test_hybrid)
timer.logging("Hybrid result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], hybrid_res['precision'][0], hybrid_res['recall'][0], hybrid_res['ndcg'][0]))

# Save results
result_file = '/kaggle/working/GAR/result/'
os.makedirs(result_file, exist_ok=True)
with open(result_file + f'{args.model}.txt', 'a') as f:
    f.write(str(vars(args)))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (cold_res['precision'][i], cold_res['recall'][i], cold_res['ndcg'][i]))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (warm_res['precision'][i], warm_res['recall'][i], warm_res['ndcg'][i]))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (hybrid_res['precision'][i], hybrid_res['recall'][i], hybrid_res['ndcg'][i]))
    f.write('\n')
