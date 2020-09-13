# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
import time
import math
import evaluate
from keras.layers import Lambda, Input, Dense
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from preprocessor import *
from test import parse_args
import threading
import os
from tensorflow.python.client import device_lib
import numpy as np

args = parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
data_generator = ml1m(args.batch_size)
intermediate_dim = 512
latent_dim = 64


class JoVA():
    def __init__(self, args, data):
        self.args = args
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.train_R = data.train_R
        self.train_R_U = data.train_R_U
        self.train_R_I = data.train_R_I
        self.train_R_U_norm = tf.nn.l2_normalize(self.train_R_U, 1)
        self.train_R_I_norm = tf.nn.l2_normalize(self.train_R_I, 1)
        self.test_R = data.test_R
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size

        self.train_num = self.train_R.sum()
        self.num_batch = int(math.ceil(self.train_num / float(self.batch_size)))

        self.lr = args.lr

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.mean_u, self.log_std_u, self.u_embeddings = self.user_vae()
        self.mean_i, self.log_std_i, self.i_embeddings = self.item_vae()

        self.u_g_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.neg_items)



        self.u_g = tf.nn.embedding_lookup(self.train_R_U, self.users)
        self.pos_i_g = tf.nn.embedding_lookup(self.train_R_I, self.pos_items)
        self.neg_i_g = tf.nn.embedding_lookup(self.train_R_I, self.neg_items)

        self.mean_u_g = tf.nn.embedding_lookup(self.mean_u, self.users)
        self.pos_mean_i_g = tf.nn.embedding_lookup(self.mean_i, self.pos_items)
        self.neg_mean_i_g = tf.nn.embedding_lookup(self.mean_i, self.neg_items)

        self.log_std_u_g = tf.nn.embedding_lookup(self.log_std_u, self.users)
        self.pos_log_std_i_g = tf.nn.embedding_lookup(self.log_std_i, self.pos_items)
        self.neg_log_std_i_g = tf.nn.embedding_lookup(self.log_std_i, self.neg_items)

        self.loss, self.bpr_loss, self.vae_loss = self.get_loss()
        self.rating = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

    def user_vae(self):
        # encoder

        self.W1_u = tf.get_variable('W1_u', [self.num_items, latent_dim], tf.float32, xavier_initializer())
        # self.W1_u = tf.nn.dropout(W1_u, keep_prob=0.8)
        # self.b1_u = tf.get_variable('b1_u', [1, latent_dim], tf.float32, xavier_initializer())
        self.W2_u = tf.get_variable('W2_u', [self.num_items, latent_dim], tf.float32, xavier_initializer())
        # self.W2_u = tf.nn.dropout(W2_u, keep_prob=0.8)
        # self.b2_u = tf.get_variable('b2_u', [1, latent_dim], tf.float32, xavier_initializer())

        mean_u = tf.matmul(self.train_R_U_norm, self.W1_u)
        log_std_u = tf.matmul(self.train_R_U_norm, self.W2_u)
        u_embedding_vae = mean_u + tf.exp(log_std_u) * tf.random_normal(shape=(self.num_users, latent_dim))



        self.W4_u = tf.get_variable('W4_u', [self.num_items, latent_dim], tf.float32, xavier_initializer())
        u_embedding_ae =  tf.matmul(self.train_R_U_norm, self.W4_u)
        u_embedding = (u_embedding_vae + u_embedding_ae)/2

        return mean_u, log_std_u, u_embedding

    def item_vae(self):
        # encoder

        self.W1_i = tf.get_variable('W1_i', [self.num_users, latent_dim], tf.float32, xavier_initializer())

        self.W2_i = tf.get_variable('W2_i', [self.num_users, latent_dim], tf.float32, xavier_initializer())

        mean_i = tf.matmul(self.train_R_I_norm, self.W1_i)
        log_std_i = tf.matmul(self.train_R_I_norm, self.W2_i)
        i_embedding_vae = mean_i + tf.exp(log_std_i) * tf.random_normal(shape=(self.num_items, latent_dim))

        self.W4_i = tf.get_variable('W4_i', [self.num_users, latent_dim], tf.float32, xavier_initializer())
        i_embedding_ae = tf.matmul(self.train_R_I_norm, self.W4_i)
        i_embedding = (i_embedding_vae + i_embedding_ae)/2


        return mean_i, log_std_i, i_embedding

    def get_loss(self):
        pos_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings), axis=1)
        bpr_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores - neg_scores) + 1e-10)))
        #bpr_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores-neg_scores)))


        kl_loss_u = tf.reduce_sum(
            tf.square(self.mean_u_g) + tf.square(tf.exp(self.log_std_u_g)) - 1 - 2 * self.log_std_u_g, axis=1)
        vae_loss_u = tf.reduce_mean( 0.0000 * kl_loss_u)


        pos_kl_loss_i = tf.reduce_sum(
            tf.square(self.pos_mean_i_g) + tf.square(tf.exp(self.pos_log_std_i_g)) - 1 - 2 * self.pos_log_std_i_g,
            axis=1)
        pos_vae_loss_i = tf.reduce_mean( 0.0000 * pos_kl_loss_i)


        neg_kl_loss_i = tf.reduce_sum(
            tf.square(self.neg_mean_i_g) + tf.square(tf.exp(self.neg_log_std_i_g)) - 1 - 2 * self.neg_log_std_i_g, axis=1)
        neg_vae_loss_i = tf.reduce_mean(0.0000 * neg_kl_loss_i)
        vae_loss = (vae_loss_u + pos_vae_loss_i + neg_vae_loss_i) * 0
        loss = bpr_loss + vae_loss + 1e-4 * (
                    tf.nn.l2_loss(self.u_g_embeddings) + tf.nn.l2_loss(self.pos_i_g_embeddings) + tf.nn.l2_loss(
                self.neg_i_g_embeddings)) / self.batch_size
        return loss, bpr_loss, vae_loss

    '''
    def prepare_model(self):
        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R_U")
        self.input_R_I = tf.placeholder(dtype=tf.float32, shape=[None, self.num_users], name="input_R_I")
        self.input_R_N_I = tf.placeholder(dtype=tf.float32, shape=[None, self.num_users], name="input_R_N_I")

        intermediate_dim = 512
        latent_dim = 128
        # encoder
        initializer = tf.contrib.layers.xavier_initializer()
        W1_u = tf.get_variable('W1_u', [self.num_items, intermediate_dim], tf.float32, xavier_initializer())
        b1_u = tf.Variable(initializer([1, intermediate_dim]))
        L1_u = tf.nn.relu(tf.add(tf.matmul(self.input_R_U, W1_u), b1_u))
        L1_u = tf.nn.dropout(L1_u, keep_prob=0.7)
        W2_u = tf.get_variable('W2_u', [intermediate_dim, latent_dim], tf.float32, xavier_initializer())
        b2_u = tf.Variable(initializer([1, latent_dim]))
        mean_u = tf.add(tf.matmul(L1_u, W2_u), b2_u)
        W3_u = tf.get_variable('W3_u', [intermediate_dim, latent_dim], tf.float32, xavier_initializer())
        b3_u = tf.Variable(initializer([1, latent_dim]))
        var_u = tf.nn.relu(tf.add(tf.matmul(L1_u, W3_u), b3_u)) + 1
        self.u_embedding = mean_u + var_u * K.random_normal(shape=(K.shape(mean_u)[0], latent_dim))
        #self.u_embedding = self.u_embedding / tf.sqrt(tf.reduce_sum(tf.square(self.u_embedding), axis=1, keepdims=True))

        # decoder
        W4_u = tf.get_variable('W4_u', [latent_dim, intermediate_dim], tf.float32, xavier_initializer())
        b4_u = tf.Variable(initializer([1, intermediate_dim]))
        L2_u = tf.nn.relu(tf.add(tf.matmul(self.u_embedding, W4_u), b4_u))
        L2_u = tf.nn.dropout(L2_u, keep_prob=0.7)
        W5_u = tf.get_variable('W5_u', [intermediate_dim, self.num_items], tf.float32, xavier_initializer())
        b5_u = tf.Variable(initializer([1, self.num_items]))
        output_user = tf.nn.sigmoid(tf.add(tf.matmul(L2_u, W5_u), b5_u))
        reconstruction_loss_u = tf.reduce_sum(tf.square(tf.multiply(self.input_R_U, self.input_R_U-output_user)), axis=1)
        kl_loss_u = tf.reduce_sum(tf.square(mean_u) + tf.squre(1 - var_u), axis = 1)
        #reconstruction_loss_u = binary_crossentropy(self.input_R_U, output_user) * self.num_items
        vae_loss_u = K.mean(reconstruction_loss_u + 0.5*kl_loss_u)
        W1_i = tf.get_variable('W1_i', [self.num_users, intermediate_dim], tf.float32, xavier_initializer())
        b1_i = tf.Variable(initializer([1, intermediate_dim]))
        L1_i = tf.nn.relu(tf.add(tf.matmul(self.input_R_I, W1_i), b1_i))
        L1_i_N = tf.nn.relu(tf.add(tf.matmul(self.input_R_N_I, W1_i), b1_i))
        L1_i = tf.nn.dropout(L1_i, keep_prob=0.7)
        L1_i_N = tf.nn.dropout(L1_i_N, keep_prob=0.7)
        W2_i = tf.get_variable('W2_i', [intermediate_dim, latent_dim], tf.float32, xavier_initializer())
        b2_i = tf.Variable(initializer([1, latent_dim]))
        mean_i = tf.add(tf.matmul(L1_i, W2_i), b2_i)
        mean_i_N = tf.add(tf.matmul(L1_i_N, W2_i), b2_i)
        W3_i = tf.get_variable('W3_i', [intermediate_dim, latent_dim], tf.float32, xavier_initializer())
        b3_i = tf.Variable(initializer([1, latent_dim]))
        var_i = tf.nn.relu(tf.add(tf.matmul(L1_i, W3_i), b3_i)) + 1
        var_i_N = tf.nn.relu(tf.add(tf.matmul(L1_i_N, W3_i), b3_i)) + 1
        self.i_embedding = mean_i + var_i * K.random_normal(shape=(K.shape(mean_i)[0], latent_dim))
        #self.i_embedding = self.i_embedding / tf.sqrt(tf.reduce_sum(tf.square(self.i_embedding), axis=1, keepdims=True))

        self.i_embedding_N = mean_i_N + var_i_N * K.random_normal(shape=(K.shape(mean_i_N)[0], latent_dim))
        #self.i_embedding_N = self.i_embedding_N / tf.sqrt( tf.reduce_sum(tf.square(self.i_embedding_N), axis=1, keepdims=True))

        # decoder

        W4_i = tf.get_variable('W4_i', [latent_dim, intermediate_dim], tf.float32, xavier_initializer())
        b4_i = tf.Variable(initializer([1, intermediate_dim]))
        L2_i = tf.nn.relu(tf.add(tf.matmul(self.i_embedding, W4_i), b4_i))
        L2_i_N = tf.nn.relu(tf.add(tf.matmul(self.i_embedding_N, W4_i), b4_i))
        L2_i = tf.nn.dropout(L2_i, keep_prob=0.7)
        L2_i_N = tf.nn.dropout(L2_i_N, keep_prob=0.7)
        W5_i = tf.get_variable('W5_i', [intermediate_dim, self.num_users], tf.float32, xavier_initializer())
        b5_i = tf.Variable(initializer([1, self.num_users]))
        output_item = tf.nn.sigmoid(tf.add(tf.matmul(L2_i, W5_i), b5_i))
        output_item_N = tf.nn.sigmoid(tf.add(tf.matmul(L2_i_N, W5_i), b5_i))
        reconstruction_loss_i = tf.reduce_sum(tf.square(tf.multiply(self.input_R_I, self.input_R_I-output_item)), axis=1)
        #reconstruction_loss_i = binary_crossentropy(self.input_R_I, output_item) * self.num_users
        #reconstruction_loss_i_N = binary_crossentropy(self.input_R_N_I, output_item_N) * self.num_users
        reconstruction_loss_i_N = tf.reduce_sum(tf.square(tf.multiply(self.input_R_N_I, self.input_R_N_I-output_item_N)), axis=1)
        kl_loss_i = tf.reduce_sum(tf.square(mean_i) + tf.squre(1 - var_i), axis=1)
        kl_loss_i_N = tf.reduce_sum(tf.square(mean_i_N) + tf.squre(1 - var_i_N), axis=1)
        vae_loss_i = K.mean(reconstruction_loss_i + 0.5*kl_loss_i)
        vae_loss_i_N = K.mean(reconstruction_loss_i_N + 0.5*kl_loss_i_N)

        pos_scores = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding_N), axis=1)
        # pre_cost = tf.maximum(neg_scores - pos_scores + 0.01,
        # tf.zeros(tf.shape(neg_scores)[0]))
        bpr_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores - neg_scores) + 1e-9)))
        # bpr_loss = tf.reduce_mean(pre_cost)
        self.Decoder = tf.matmul(self.u_embedding, self.i_embedding, transpose_a=False, transpose_b=True)

        self.cost = 0 * (vae_loss_i + vae_loss_u + vae_loss_i_N) + bpr_loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimizer = optimizer.minimize(self.cost)
    '''


class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample()


class train_thread(threading.Thread):
    def __init__(self, model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample

    def run(self):
        users, pos_items, neg_items = self.sample.data
        self.data = self.sess.run([self.model.optimizer, self.model.loss, self.model.bpr_loss, self.model.vae_loss],
                                  feed_dict={self.model.users: users, self.model.pos_items: pos_items,
                                             self.model.neg_items: neg_items})


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model = JoVA(args, data_generator)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # tf.set_random_seed(777)

    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    n_batch = model.num_batch
    train_R = model.train_R
    for epoch in range(1, args.train_epoch + 1):
        t1 = time.time()
        loss, bpr_loss, vae_loss = 0., 0., 0.
        '''
        *********************************************************
        parallelized sampling
        '''
        sample_last = sample_thread()
        sample_last.start()
        sample_last.join()
        for idx in range(n_batch):
            train_cur = train_thread(model, sess, sample_last)
            sample_next = sample_thread()

            train_cur.start()
            sample_next.start()

            sample_next.join()
            train_cur.join()

            row_id, col_idx, col_N_idx = sample_last.data
            _, batch_loss, batch_bpr_loss, batch_vae_loss = train_cur.data
            sample_last = sample_next

            loss += batch_loss / n_batch
            bpr_loss += batch_bpr_loss / n_batch
            vae_loss += batch_vae_loss / n_batch

        print("Epoch %d //" % (epoch), " cost = {:.8f}".format(loss), " bpr_cost = {:.8f}".format(bpr_loss),
              " vae_cost = {:.5f}".format(vae_loss)
              , "Elapsed time : %d sec" % (time.time() - t1))
        evaluate.test_all(sess, model)
        print("=" * 100)
