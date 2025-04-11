# GAR.py
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

class GAR(tf.keras.Model):
    def __init__(self, emb_dim, content_dim, args):
        super(GAR, self).__init__()
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
        self.K = 3  # TDRO: Amazon-specific
        self.E = 3
        self.lambda_ = 0.9
        self.p = 0.2
        self.eta_w = 0.1
        self.mu = 0.2
        self.alpha = args.alpha  # 0.5
        self.beta = args.beta   # 0.6

        # Generator layers
        self.g_layers = [tf.keras.layers.Dense(units, activation=self.g_act,
                                               kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                               kernel_initializer='glorot_uniform')
                         for units in self.g_layer]
        self.g_output_layer = tf.keras.layers.Dense(self.emb_dim, activation=None,
                                                    kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                                    kernel_initializer='glorot_uniform')
        self.g_dropout = tf.keras.layers.Dropout(self.g_drop)
        
        # Discriminator layers
        self.d_user_layers = [tf.keras.layers.Dense(units, activation=self.d_act,
                                                    kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                                    kernel_initializer='glorot_uniform')
                              for units in self.d_layer]
        self.d_item_layers = [tf.keras.layers.Dense(units, activation=self.d_act,
                                                    kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                                    kernel_initializer='glorot_uniform')
                              for units in self.d_layer]
        self.d_dropout = tf.keras.layers.Dropout(self.d_drop)

        # TDRO variables
        self.group_weights = tf.Variable(tf.ones([self.K]) / self.K, trainable=False)
        self.prev_group_losses_G = [tf.Variable(0.0, trainable=False) for _ in range(self.K)]
        self.prev_group_losses_D = [tf.Variable(0.0, trainable=False) for _ in range(self.K)]

        # Optimizers
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_lr)
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_lr)

    def build_generator(self, content, training):
        hidden = content
        for layer in self.g_layers:
            hidden = layer(hidden)
            hidden = self.g_dropout(hidden, training=training)
        return self.g_output_layer(hidden)

    def build_discriminator(self, uembs, iembs, training):
        u_hidden = uembs
        i_hidden = iembs
        for u_layer, i_layer in zip(self.d_user_layers, self.d_item_layers):
            u_hidden = u_layer(u_hidden)
            i_hidden = i_layer(i_hidden)
            u_hidden = self.d_dropout(u_hidden, training=training)
            i_hidden = self.d_dropout(i_hidden, training=training)
        return tf.reduce_sum(u_hidden * i_hidden, axis=-1)

    @tf.function
    def train_d(self, batch_uemb, batch_iemb, batch_neg_iemb, batch_content, batch_group_ids, period_grads):
        gen_emb = self.build_generator(batch_content, training=True)
        uemb = tf.tile(batch_uemb, [3, 1])  # [3072, emb_dim]
        iemb = tf.concat([batch_iemb, batch_neg_iemb, gen_emb], axis=0)  # [3072, emb_dim]
        period_grads_broadcast = tf.reshape(period_grads, [self.emb_dim, 1])  # Reshape to [128, 1] for broadcasting
    
        with tf.GradientTape() as tape:
            d_out = self.build_discriminator(uemb, iemb, training=True)  # [3072,]
            batch_size = tf.shape(batch_uemb)[0]  # 1024
            real_logit = d_out[:batch_size]  # [1024,]
            neg_logit = d_out[batch_size:2*batch_size]  # [1024,]
            d_fake_logit = d_out[2*batch_size:3*batch_size]  # [1024,]
            
            d_loss_base = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_logit - (1 - self.beta) * d_fake_logit - self.beta * neg_logit,
                labels=tf.ones_like(real_logit)
            )  # [1024,]
            
            # Group-wise loss
            d_group_losses = [tf.reduce_mean(tf.boolean_mask(d_loss_base, tf.equal(batch_group_ids, j)))
                              for j in range(self.K)]
            d_loss = tf.reduce_mean(d_loss_base) + tf.reduce_sum(self.losses)
    
            # TDRO update
            scores = []
            d_vars = self.trainable_variables  # All trainable vars (approximation)
            for j in range(self.K):
                g_j = tf.gradients(d_group_losses[j], d_vars)[0]  # Shape might be [128, 200] for a layer
                shift = tf.reduce_sum(g_j * period_grads_broadcast) if g_j is not None else 0.0
                score = (1 - self.lambda_) * d_group_losses[j] + self.lambda_ * shift
                scores.append(score)
            scores = tf.stack(scores)
            w_new = tf.nn.softmax(tf.math.log(self.group_weights + 1e-9) + self.eta_w * scores)
            self.group_weights.assign(w_new)
    
        grads = tape.gradient(d_loss, self.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Smooth losses
        for j in range(self.K):
            smoothed_loss = (1 - self.mu) * self.prev_group_losses_D[j] + self.mu * d_group_losses[j]
            self.prev_group_losses_D[j].assign(smoothed_loss)
        
        return [d_loss]

    @tf.function
    def train_g(self, batch_uemb, batch_iemb, batch_content, batch_group_ids, period_grads):
        with tf.GradientTape() as tape:
            gen_emb = self.build_generator(batch_content, training=True)
            g_out = self.build_discriminator(batch_uemb, gen_emb, training=True)
            d_out = self.build_discriminator(batch_uemb, batch_iemb, training=True)
            g_loss_base = (1.0 - self.alpha) * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=g_out - d_out, labels=tf.ones_like(g_out))
            sim_loss = self.alpha * tf.reduce_mean(tf.abs(gen_emb - batch_iemb))
            
            # Group-wise loss
            g_group_losses = [tf.reduce_mean(tf.boolean_mask(g_loss_base, tf.equal(batch_group_ids, j)))
                              for j in range(self.K)]
            g_loss = tf.reduce_mean(g_loss_base) + sim_loss + tf.reduce_sum(self.losses)

            # TDRO update
            scores = []
            g_vars = self.trainable_variables
            for j in range(self.K):
                g_j = tf.gradients(g_group_losses[j], g_vars)[0]
                shift = tf.reduce_sum(g_j * period_grads) if g_j is not None else 0.0
                score = (1 - self.lambda_) * g_group_losses[j] + self.lambda_ * shift
                scores.append(score)
            scores = tf.stack(scores)
            w_new = tf.nn.softmax(tf.math.log(self.group_weights + 1e-9) + self.eta_w * scores)
            self.group_weights.assign(w_new)

        grads = tape.gradient(g_loss, self.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Smooth losses
        for j in range(self.K):
            smoothed_loss = (1 - self.mu) * self.prev_group_losses_G[j] + self.mu * g_group_losses[j]
            self.prev_group_losses_G[j].assign(smoothed_loss)
        
        return [g_loss, sim_loss]

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        out_emb = np.copy(item_emb)
        out_emb[cold_item] = self.build_generator(content[cold_item], training=False).numpy()
        return self.build_discriminator(out_emb, out_emb, training=False).numpy()  # Simplified

    def get_user_emb(self, user_emb):
        return self.build_discriminator(user_emb, user_emb, training=False).numpy()  # Simplified

    def get_user_rating(self, uids, iids, uemb, iemb):
        return tf.matmul(uemb[uids], iemb[iids], transpose_b=True).numpy()

    def get_ranked_rating(self, ratings, k):
        top_score, top_item_index = tf.nn.top_k(ratings, k=k)
        return top_score.numpy(), top_item_index.numpy()
