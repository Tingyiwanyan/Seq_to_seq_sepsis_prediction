from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

semantic_step_global = 6
semantic_positive_sample = 8
unsupervised_cluster_num = 10
latent_dim_global = 100
positive_sample_size = 5
batch_size = 128
unsupervised_neg_size = 5


class projection(keras.layers.Layer):
    def __init__(self, units=unsupervised_cluster_num, input_dim=latent_dim_global):
        super(projection, self).__init__()
        # w_init = tf.random_normal_initializer()
        w_init = tf.keras.initializers.Orthogonal()
        self.w = tf.Variable(
            initial_value=w_init(shape=(units, input_dim), dtype="float32"),
            trainable=True,
        )
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(
        # initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        # )

    def call(self, inputs):
        return tf.math.multiply(inputs, self.w)


class protatype_ehr():
    def __init__(self, projection):
        #self.read_d = read_d
        self.projection_model = projection
        #self.train_data = read_d.train_data
        #self.test_data = read_d.test_data
        #self.validate_data = read_d.val_data
        #self.length_train = len(self.train_data)
        #self.length_test = len(self.test_data)
        #self.length_val = len(self.validate_data)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")


        """
        define hyper-parameters
        """
        self.gaussian_mu = 0
        self.gaussian_sigma = 0.0001
        self.batch_size = batch_size
        self.neg_size = self.batch_size
        self.pos_size = positive_sample_size
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 20
        self.feature_num = 34
        self.pre_train_epoch = 15
        self.latent_dim = latent_dim_global
        self.tau = 1
        self.time_sequence = 48#self.read_d.time_sequence
        self.tcn_filter_size = 3
        self.semantic_time_step = semantic_step_global
        self.unsupervised_cluster_num = unsupervised_cluster_num
        self.unsupervised_neg_size = unsupervised_neg_size
        self.start_sampling_index = 5
        self.sampling_interval = 5
        self.converge_threshold_E = 200
        self.semantic_positive_sample = semantic_positive_sample
        self.max_value_projection = np.zeros((self.batch_size, self.semantic_time_step))
        self.basis_input = np.ones((self.unsupervised_cluster_num, self.latent_dim))

        self.create_memory_bank()
        self.length_train = len(self.train_data)

        """
        initialize orthogonal projection basis
        """
        self.initializer_basis = tf.keras.initializers.Orthogonal()
        self.init_projection_basis = tf.Variable(
            self.initializer_basis(shape=(self.unsupervised_cluster_num, self.latent_dim)))

        self.steps = self.length_train // self.batch_size
        self.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0003, decay_steps=self.steps)

        """
        initialize position encoding vectors
        """
        self.position_embedding = np.zeros((self.semantic_time_step, self.latent_dim))
        self.generate_orthogonal = ortho_group.rvs(self.latent_dim)
        for i in range(self.semantic_time_step):
            # self.position_embedding[i,:] = self.position_encoding(i)
            self.position_embedding[i, :] = self.generate_orthogonal[i]

        # self.batch_position_embedding = np.expand_dims(self.position_embedding,0)
        # self.batch_position_embedding = np.broadcast_to(self.batch_position_embedding,[self.batch_size,
        # self.semantic_time_step,
        # self.latent_dim])

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0003,
            decay_steps=self.steps,
            decay_rate=0.3)

    def create_memory_bank(self):
        # self.train_data, self.train_logit,self.train_sofa,self.train_sofa_score = self.aquire_data(0, self.train_data, self.length_train)
        # self.test_data, self.test_logit,self.test_sofa,self.test_sofa_score = self.aquire_data(0, self.test_data, self.length_test)
        # self.val_data, self.val_logit,self.val_sofa,self.val_sofa_score = self.aquire_data(0, self.validate_data, self.length_val)

        #file_path = '/home/tingyi/physionet_data/Interpolate_data/'
        file_path = '/athena/penglab/scratch/tiw4003/Interpolate_data/'
        with open(file_path + 'train.npy', 'rb') as f:
            self.train_data = np.load(f)
        with open(file_path + 'train_logit.npy', 'rb') as f:
            self.train_logit = np.load(f)
        with open(file_path + 'train_on_site_time.npy', 'rb') as f:
            self.train_on_site_time = np.load(f)

        # with open(file_path + 'test.npy', 'rb') as f:
        # self.test_data = np.load(f)
        # with open(file_path + 'test_logit.npy', 'rb') as f:
        # self.test_logit = np.load(f)
        with open(file_path + 'val.npy', 'rb') as f:
            self.val_data = np.load(f)
        with open(file_path + 'val_logit.npy', 'rb') as f:
            self.val_logit = np.load(f)
        with open(file_path + 'val_on_site_time.npy', 'rb') as f:
            self.val_on_site_time = np.load(f)
        # with open(file_path + 'train_sofa_score.npy', 'rb') as f:
        # self.train_sofa_score = np.load(f)
        # with open(file_path + 'train_sofa.npy', 'rb') as f:
        # self.train_sofa = np.load(f)

        with open(file_path + 'train_origin.npy', 'rb') as f:
            self.train_data_origin = np.load(f)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_data, self.train_logit, self.train_on_site_time, self.train_data_origin))  # ,self.train_sofa_score))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        cohort_index = np.where(self.train_logit == 1)[0]
        control_index = np.where(self.train_logit == 0)[0]
        self.memory_bank_cohort = self.train_data[cohort_index, :, :]
        self.memory_bank_control = self.train_data[control_index, :, :]
        self.memory_bank_cohort_on_site = self.train_on_site_time[cohort_index]
        self.memory_bank_control_on_site = self.train_on_site_time[control_index]
        self.memory_bank_cohort_origin = self.train_data_origin[cohort_index]
        self.memory_bank_control_origin = self.train_data_origin[control_index]
        self.num_cohort = self.memory_bank_cohort.shape[0]
        self.num_control = self.memory_bank_control.shape[0]