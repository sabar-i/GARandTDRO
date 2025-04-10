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
from GAR import GAR
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

# Convert sets to integer arrays
warm_items = np.array(list(warm_items_raw), dtype=np.int64)  # Convert set to list, then to int64 array
cold_items = np.array(list(cold_items_raw), dtype=np.int64)  # Same for cold items
print("warm_items_raw type:", type(warm_items_raw), "content:", warm_items_raw)
print("warm_items type:", type(warm_items), "content:", warm_items)
print("cold_items_raw type:", type(cold_items_raw), "content:", cold_items_raw)
print("cold_items type:", type(cold_items), "content:", cold_items)

training_dict = np.load(os.path.join(data_dir, 'training_dict.npy'), allow_pickle=True).item()
validation_cold_dict = np.load(os.path.join(data_dir, 'validation_cold_dict.npy'), allow_pickle=True).item()
validation_warm_dict = np.load(os.path.join(data_dir, 'validation_warm_dict.npy'), allow_pickle=True).item()
testing_cold_dict = np.load(os.path.join(data_dir, 'testing_cold_dict.npy'), allow_pickle=True).item()
testing_warm_dict = np.load(os.path.join(data_dir, 'testing_warm_dict.npy'), allow_pickle=True).item()
interaction_timestamp_dict = np.load(os.path.join(data_dir, 'interaction_timestamp_dict.npy'), allow_pickle=True).item()

user_node_num = len(user_map) + 1
item_node_num = len(item_map) + 1
user_emb = emb[:user_node_num]
item_emb = emb[user_node_num:user_node_num + item_node_num]
para_dict = {
    'user_array': np.arange(user_node_num - 1),
    'item_array': np.arange(item_node_num - 1),
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
timer.logging('Data loaded from {}'.format(data_dir))

# TDRO Preprocessing
K = 3
E = 3
warm_features = content_data[warm_items]
kmeans = KMeans(n_clusters=K, random_state=args.seed)
group_labels = kmeans.fit_predict(warm_features)
group_map = {int(item): label for item, label in zip(warm_items, group_labels)}  # Ensure int keys

interactions = []
for uid, items in training_dict.items():
    timestamps = interaction_timestamp_dict.get(uid, [0] * len(items))
    for iid, ts in zip(items, timestamps):
        interactions.append((uid, iid, ts))
interactions = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'timestamp'])
interactions = interactions.sort_values('timestamp')
period_size = len(interactions) // E
periods = [interactions.iloc[i:i + period_size] for i in range(0, len(interactions), period_size)]

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
model = eval(args.model)(emb.shape[-1], content_data.shape[-1], args)
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
    train_input = utils.bpr_neg_samp(para_dict['warm_user'], len(interactions), para_dict['emb_user_nb'], para_dict['warm_item'])
    n_batch = len(train_input) // args.batch_size
    for beg in range(0, len(train_input) - args.batch_size, args.batch_size):
        end = beg + args.batch_size
        batch_count += 1
        t_train_begin = time.time()
        batch_lbs = train_input[beg:end]
        batch_group_ids = np.array([group_map.get(int(item), 0) for item in batch_lbs[:, 1]])  # Ensure int keys
        period_grads = np.zeros(model.emb_dim)

        d_loss = model.train_d(user_emb[batch_lbs[:, 0]], item_emb[batch_lbs[:, 1]], item_emb[batch_lbs[:, 2]],
                               content_data[batch_lbs[:, 1]], batch_group_ids, period_grads)
        g_loss = model.train_g(user_emb[batch_lbs[:, 0]], item_emb[batch_lbs[:, 1]], content_data[batch_lbs[:, 1]],
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
