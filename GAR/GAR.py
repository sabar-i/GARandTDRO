import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

def build_mlp(mlp_in, hidden_dims, act, drop_rate, is_training, scope_name, bn_first=True):
    with tf.variable_scope(scope_name):
        hidden = mlp_in
        if bn_first:
            hidden = tf.layers.batch_normalization(hidden, training=is_training, scale=False, name='mlp_bn_1')
        hidden = tf.layers.dense(hidden, hidden_dims[0], name="mlp_fc_1",
                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        for i in range(2, len(hidden_dims) + 1):
            if act == 'relu':
                hidden = tf.nn.leaky_relu(hidden, alpha=0.01)
            hidden = tf.layers.batch_normalization(hidden, training=is_training, name='mlp_bn_' + str(i))
            if act == 'tanh':
                hidden = tf.nn.tanh(hidden)
            hidden = tf.layers.dropout(hidden, rate=drop_rate, training=is_training, name='mlp_drop_' + str(i))
            hidden = tf.layers.dense(hidden, hidden_dims[i - 1],
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     name='mlp_fc_' + str(i))
        return hidden

class GAR(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.g_lr = 1e-3
        self.d_lr = 1e-3
        self.g_drop = 0.1
        self.d_drop = 0.5
        self.g_layer = [200, 200]
        self.d_layer = [200, 200]
        self.g_act = 'tanh'
        self.d_act = 'tanh'
        # TDRO hyperparameters for Amazon
        self.K = 3  # Number of groups (TDRO paper)
        self.E = 3  # Number of time periods (TDRO paper)
        self.lambda_ = 0.9  # Trade-off parameter (TDRO paper)
        self.p = 0.2  # Shifting factor exponent (TDRO paper)
        self.eta_w = 0.1  # Group weight learning rate
        self.mu = 0.2  # Loss smoothing factor

        # Placeholders
        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.real_emb = tf.placeholder(tf.float32, [None, emb_dim], name='real_emb')
        self.neg_emb = tf.placeholder(tf.float32, [None, emb_dim], name='neg_emb')
        self.opp_emb = tf.placeholder(tf.float32, [None, emb_dim], name='opp_emb')
        self.g_training = tf.placeholder(tf.bool, name='G_is_training')
        self.d_training = tf.placeholder(tf.bool, name='D_is_training')
        self.group_ids = tf.placeholder(tf.int32, [None], name='group_ids')  # Group assignments for items
        self.period_grads = tf.placeholder(tf.float32, [emb_dim], name='period_grads')  # Temporal gradients

        # Generator
        self.gen_emb = self.build_generator(self.content, self.g_layer, self.g_act, self.g_drop, self.g_training, False)

        # Discriminator loss
        uemb = tf.tile(self.opp_emb, [3, 1])
        iemb = tf.concat([self.real_emb, self.neg_emb, self.gen_emb], axis=0)
        D_out = self.build_discriminator(uemb, iemb, self.d_layer, self.d_act, self.d_drop, self.d_training, False)
        self.D_out = tf.transpose(tf.reshape(D_out, [3, -1]))
        self.real_logit = tf.gather(self.D_out, indices=[0], axis=1)
        self.neg_logit = tf.gather(self.D_out, indices=[1], axis=1)
        self.d_fake_logit = tf.gather(self.D_out, indices=[2], axis=1)
        self.d_loss_base = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.real_logit - (1 - args.beta) * self.d_fake_logit - args.beta * self.neg_logit,
            labels=tf.ones_like(self.real_logit))

        # Group-wise D loss
        self.d_group_losses = []
        for j in range(self.K):
            mask = tf.equal(self.group_ids, j)
            group_loss = tf.reduce_mean(tf.boolean_mask(self.d_loss_base, mask))
            self.d_group_losses.append(group_loss)
        self.d_loss = tf.reduce_mean(self.d_loss_base) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='D'))

        # Generator loss
        self.g_out = self.build_discriminator(self.opp_emb, self.gen_emb, self.d_layer, self.d_act, self.d_drop, self.d_training, True)
        self.d_out = self.build_discriminator(self.opp_emb, self.real_emb, self.d_layer, self.d_act, self.d_drop, self.d_training, True)
        self.g_loss_base = (1.0 - args.alpha) * tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.g_out - self.d_out, labels=tf.ones_like(self.g_out))
        self.sim_loss = args.alpha * tf.reduce_mean(tf.abs(self.gen_emb - self.real_emb))
        
        # Group-wise G loss
        self.g_group_losses = []
        for j in range(self.K):
            mask = tf.equal(self.group_ids, j)
            group_loss = tf.reduce_mean(tf.boolean_mask(self.g_loss_base, mask))
            self.g_group_losses.append(group_loss)
        self.g_loss = tf.reduce_mean(self.g_loss_base) + self.sim_loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G'))

        # TDRO variables
        self.group_weights = tf.Variable(tf.ones([self.K]) / self.K, trainable=False, name='group_weights')
        self.prev_group_losses_G = [tf.Variable(0.0, trainable=False) for _ in range(self.K)]
        self.prev_group_losses_D = [tf.Variable(0.0, trainable=False) for _ in range(self.K)]

        # Optimizers
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(d_update_ops):
            self.d_optimizer = tf.train.AdamOptimizer(self.d_lr).minimize(self.d_loss,
                                                                         var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self.g_optimizer = tf.train.AdamOptimizer(self.g_lr).minimize(self.g_loss,
                                                                         var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))

        # User rating and ranking
        self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
        self.user_rating = tf.matmul(self.uemb, self.iemb, transpose_b=True)
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        # Transformed embeddings
        self.user_ori_emb = tf.placeholder(tf.float32, [None, emb_dim], name='ori_user_emb')
        self.item_ori_emb = tf.placeholder(tf.float32, [None, emb_dim], name='ori_item_emb')
        with tf.variable_scope("D", reuse=True):
            self.warm_user_emb = build_mlp(self.user_ori_emb, self.d_layer, self.d_act, self.d_drop, False, 'user_emb', bn_first=True)
            self.warm_item_emb = build_mlp(self.item_ori_emb, self.d_layer, self.d_act, self.d_drop, False, 'item_emb', bn_first=True)

        self.sess.run(tf.global_variables_initializer())
        print([v.name for v in tf.trainable_variables()])

    def build_generator(self, condition, hid_dims, act, drop_rate, training, reuse):
        with tf.variable_scope('G', reuse=reuse):
            gen_emb = build_mlp(condition, hid_dims, act, drop_rate, training, 'E0', False)
        return gen_emb

    def build_discriminator(self, uembs, iembs, hid_dims, act, drop_rate, training, reuse):
        with tf.variable_scope("D", reuse=reuse):
            out_uemb = build_mlp(uembs, hid_dims, act, drop_rate, training, 'user_emb', bn_first=True)
            out_iemb = build_mlp(iembs, hid_dims, act, drop_rate, training, 'item_emb', bn_first=True)
            out = tf.reduce_sum(out_uemb * out_iemb, axis=-1)
        return out

    def train_d(self, batch_uemb, batch_iemb, batch_neg_iemb, batch_content, batch_group_ids, period_grads):
        # Compute group losses and gradients
        feed_dict = {
            self.opp_emb: batch_uemb,
            self.real_emb: batch_iemb,
            self.neg_emb: batch_neg_iemb,
            self.content: batch_content,
            self.group_ids: batch_group_ids,
            self.period_grads: period_grads,
            self.d_training: True,
            self.g_training: False
        }
        d_group_losses, d_loss = self.sess.run([self.d_group_losses, self.d_loss], feed_dict=feed_dict)
        
        # TDRO: Update group weights
        scores = []
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        for j in range(self.K):
            g_j = tf.gradients(d_group_losses[j], d_vars)[0]
            shift = tf.reduce_sum(g_j * self.period_grads) if g_j is not None else 0.0
            score = (1 - self.lambda_) * d_group_losses[j] + self.lambda_ * shift
            scores.append(score)
        scores = tf.stack(scores)
        w_new = tf.nn.softmax(tf.log(self.group_weights + 1e-9) + self.eta_w * scores)
        self.sess.run(self.group_weights.assign(w_new))

        # Smooth losses and update
        for j in range(self.K):
            smoothed_loss = (1 - self.mu) * self.prev_group_losses_D[j] + self.mu * d_group_losses[j]
            self.sess.run(self.prev_group_losses_D[j].assign(smoothed_loss))
        self.sess.run(self.d_optimizer, feed_dict=feed_dict)
        return [d_loss]

    def train_g(self, batch_uemb, batch_iemb, batch_content, batch_group_ids, period_grads):
        # Compute group losses and gradients
        feed_dict = {
            self.opp_emb: batch_uemb,
            self.real_emb: batch_iemb,
            self.content: batch_content,
            self.group_ids: batch_group_ids,
            self.period_grads: period_grads,
            self.d_training: False,
            self.g_training: True
        }
        g_group_losses, g_loss, sim_loss = self.sess.run([self.g_group_losses, self.g_loss, self.sim_loss], feed_dict=feed_dict)
        
        # TDRO: Update group weights
        scores = []
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        for j in range(self.K):
            g_j = tf.gradients(g_group_losses[j], g_vars)[0]
            shift = tf.reduce_sum(g_j * self.period_grads) if g_j is not None else 0.0
            score = (1 - self.lambda_) * g_group_losses[j] + self.lambda_ * shift
            scores.append(score)
        scores = tf.stack(scores)
        w_new = tf.nn.softmax(tf.log(self.group_weights + 1e-9) + self.eta_w * scores)
        self.sess.run(self.group_weights.assign(w_new))

        # Smooth losses and update
        for j in range(self.K):
            smoothed_loss = (1 - self.mu) * self.prev_group_losses_G[j] + self.mu * g_group_losses[j]
            self.sess.run(self.prev_group_losses_G[j].assign(smoothed_loss))
        self.sess.run(self.g_optimizer, feed_dict=feed_dict)
        return [g_loss, sim_loss]

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        out_emb = np.copy(item_emb)
        out_emb[cold_item] = self.sess.run(self.gen_emb, feed_dict={self.content: content[cold_item], self.g_training: False})
        out_emb = self.sess.run(self.warm_item_emb, feed_dict={self.item_ori_emb: out_emb})
        return out_emb

    def get_user_emb(self, user_emb):
        trans_user_emb = self.sess.run(self.warm_user_emb, feed_dict={self.user_ori_emb: user_emb})
        return trans_user_emb

    def get_user_rating(self, uids, iids, uemb, iemb):
        user_rat = self.sess.run(self.user_rating, feed_dict={self.uemb: uemb[uids], self.iemb: iemb[iids]})
        return user_rat

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_index = self.sess.run([self.top_score, self.top_item_index], feed_dict={self.rat: ratings, self.k: k})
        return ranked_score, ranked_index
