from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.cluster import KMeans
from tcn import TCN
from tensorflow import keras
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
import numpy_indexed as npi
from numpy import savetxt

import matplotlib.pyplot as plt
import numpy as np
import random

semantic_step_global = 6
unsupervised_cluster_num = 10
latent_dim_global = 100
positive_sample_size = 5


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
    def __init__(self, read_d, projection):
        self.read_d = read_d
        self.projection_model = projection
        self.train_data = read_d.train_data
        self.test_data = read_d.test_data
        self.validate_data = read_d.val_data
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.length_val = len(self.validate_data)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        """
        define hyper-parameters
        """
        self.gaussian_mu = 0
        self.gaussian_sigma = 0.0001
        self.batch_size = 128
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
        self.time_sequence = self.read_d.time_sequence
        self.tcn_filter_size = 5
        self.semantic_time_step = semantic_step_global
        self.unsupervised_cluster_num = unsupervised_cluster_num
        self.start_sampling_index = 5
        self.sampling_interval = 5
        self.converge_threshold_E = 200
        self.semantic_positive_sample = 5
        self.max_value_projection = np.zeros((self.batch_size, self.semantic_time_step))
        self.basis_input = np.ones((self.unsupervised_cluster_num, self.latent_dim))

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

        file_path = '/home/tingyi/physionet_data/Interpolate_data/'
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
        self.num_cohort = self.memory_bank_cohort.shape[0]
        self.num_control = self.memory_bank_control.shape[0]

    """
    def compute_positive_pair(self, z, p):
        z = tf.math.l2_normalize(z, axis=1)
        p = tf.math.l2_normalize(p, axis=1)

        positive_dot_prod = tf.multiply(z, p)
        positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(positive_dot_prod, 1) / self.tau)

        return positive_dot_prod_sum

    def compute_negative_paris(self, z, p):
        z = tf.math.l2_normalize(z, axis=1)
        p = tf.math.l2_normalize(p, axis=1)

        similarity_matrix = tf.matmul(z, tf.transpose(p))
        mask = tf.linalg.diag(tf.zeros(p.shape[0]), padding_value=1)

        negative_dot_prods = tf.multiply(similarity_matrix, mask)
        negative_dot_prods_sum = tf.reduce_sum(tf.math.exp(negative_dot_prods / self.tau), 1)

        return negative_dot_prods_sum

    def info_nce_loss(self, z, p):
        positive_dot_prod_sum = self.compute_positive_pair(z,p)
        negative_dot_prod_sum = self.compute_negative_paris(z,p)

        denominator = tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator))
        loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum, denominator), 0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss
    """

    def compute_positive_pair(self, z, p):
        z = tf.math.l2_normalize(z, axis=-1)
        p = tf.math.l2_normalize(p, axis=-1)

        z = tf.expand_dims(z,1)

        positive_dot_prod = tf.multiply(z, p)
        self.check_positive_dot_prod = positive_dot_prod
        positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(tf.reduce_sum(positive_dot_prod, -1),-1) / self.tau)

        return positive_dot_prod_sum

    def compute_negative_paris(self, z, global_pull_cohort, global_pull_control, label):
        z = tf.math.l2_normalize(z, axis=1)

        global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=-1)
        global_pull_control = tf.math.l2_normalize(global_pull_control, axis=-1)

        global_pull_cohort = tf.reshape(global_pull_cohort,(global_pull_cohort.shape[0]*global_pull_cohort.shape[1],
                                                            global_pull_cohort.shape[2]))

        global_pull_control = tf.reshape(global_pull_control,(global_pull_control.shape[0]*global_pull_control.shape[1],
                                                              global_pull_control.shape[2]))

        similarity_matrix_cohort = tf.matmul(z, tf.transpose(global_pull_cohort))
        similarity_matrix_control = tf.matmul(z, tf.transpose(global_pull_control))

        neg_cohort_sum = tf.reduce_sum(tf.math.exp(similarity_matrix_cohort / self.tau), 1)
        self.check_neg_cohort_sum = neg_cohort_sum
        neg_control_sum = tf.reduce_sum(tf.math.exp(similarity_matrix_control / self.tau), 1)
        self.check_neg_control_sum = neg_control_sum
        label = tf.cast(label, tf.int32)
        self.check_label = label

        neg_sum_both = tf.stack((neg_cohort_sum, neg_control_sum), 1)
        self.check_neg_sum_both = neg_sum_both
        negative_dot_prods_sum = [neg_sum_both[i, label[i]] for i in range(z.shape[0])]
        self.check_negative_dot_prods_sum = negative_dot_prods_sum

        return negative_dot_prods_sum

    def info_nce_loss(self, z, p, global_pull_cohort, global_pull_control, label):
        positive_dot_prod_sum = self.compute_positive_pair(z, p)
        negative_dot_prod_sum = self.compute_negative_paris(z, global_pull_cohort, global_pull_control, label)

        denominator = tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator))
        loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum, denominator), 0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss

    def compute_positive_pair_un(self, z, p):
        z = tf.math.l2_normalize(z, axis=-1)
        p = tf.math.l2_normalize(p, axis=-1)

        positive_dot_prod = tf.multiply(z, p)
        positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(positive_dot_prod, -1) / self.tau)

        return positive_dot_prod_sum

    def unsupervised_prototype_loss(self, extract_time, projection_basis, order_input):
        extract_time = tf.math.l2_normalize(extract_time, axis=1)
        # extract_time_order = tf.reshape(extract_time,
        # [extract_time.shape[0]*extract_time.shape[1],extract_time.shape[2]])
        projection_basis = tf.math.l2_normalize(projection_basis, axis=-1)
        projection_basis_expand = tf.expand_dims(projection_basis, axis=0)
        projection_basis_expand = tf.expand_dims(projection_basis_expand, axis=0)
        projection_basis_broad = tf.broadcast_to(projection_basis_expand,
                                                 [extract_time.shape[0], extract_time.shape[1],
                                                  projection_basis.shape[0], projection_basis.shape[1]])

        extract_time_expand = tf.expand_dims(extract_time, axis=2)
        extract_time_broad = tf.broadcast_to(extract_time_expand, [extract_time.shape[0], extract_time.shape[1],
                                                                   projection_basis.shape[0], extract_time.shape[2]])

        denominator = tf.multiply(projection_basis_broad, extract_time_broad)

        negative_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(denominator, -1) / self.tau), 2)

        negative_dot_prod_sum = tf.reshape(negative_dot_prod_sum,
                                           [negative_dot_prod_sum.shape[0] * negative_dot_prod_sum.shape[1]])

        projection_basis = tf.expand_dims(projection_basis, 0)

        self.total_sementic_un = []
        for i in range(self.unsupervised_cluster_num):
            check = order_input == i
            check = tf.cast(check, tf.float64)
            #check = tf.cast(check, tf.float32)
            check = tf.expand_dims(check, 2)
            self.check_un = check
            # projection_single = tf.broadcast_to(tf.expand_dims(projection_basis[0,i,:],0)
            # ,projection_basis.shape)

            projection_basis_single = tf.expand_dims(projection_basis[:, i, :], 1)
            projection_single = tf.broadcast_to(projection_basis_single, shape=(check.shape[0],
                                                                                check.shape[1],
                                                                                projection_basis.shape[2]))

            self.check_projection_single_un = projection_single
            batch_semantic_embedding_single = tf.math.multiply(projection_single,
                                                               check)
            #self.check_batch_semantic_embedding_single = batch_semantic_embedding_single
            # batch_semantic_embedding_single = tf.reduce_sum(batch_semantic_embedding_single, axis=1)
            # batch_semantic_embedding_single = tf.expand_dims(batch_semantic_embedding_single, axis=1)
            self.total_sementic_un.append(batch_semantic_embedding_single)

        batch_semantic_embedding_whole = tf.stack(self.total_sementic_un)
        batch_semantic_embedding_whole = tf.reduce_sum(batch_semantic_embedding_whole, axis=0)

        pos_prod_sum = self.compute_positive_pair_un(extract_time, batch_semantic_embedding_whole)

        pos_prod_sum = tf.reshape(pos_prod_sum, [pos_prod_sum.shape[0] * pos_prod_sum.shape[1]])

        nomalized_prob_log = tf.math.log(tf.math.divide(pos_prod_sum, negative_dot_prod_sum))

        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss

    def projection_regularize_loss(self, projection_basis):
        projection_basis = projection_basis[0, :, :]

        similarity_matrix = tf.matmul(projection_basis, tf.transpose(projection_basis))
        mask = tf.linalg.diag(tf.zeros(projection_basis.shape[0]), padding_value=1)

        negative_dot_prods = tf.math.abs(tf.multiply(similarity_matrix, mask))
        projection_regular_loss = tf.reduce_mean(tf.reduce_sum(negative_dot_prods, 1))

        return projection_regular_loss

    def first_lvl_resolution_deconv(self):
        inputs = layers.Input((1, self.latent_dim))

        tcn_deconv1 = tf.keras.layers.Conv1DTranspose(self.feature_num, self.tcn_filter_size)

        output = tcn_deconv1(inputs)

        return tf.keras.Model(inputs, output, name='tcn_deconv1')

    def one_h_resolution_deconv(self):
        inputs = layers.Input((1, self.latent_dim))

        tcn_deconv1 = tf.keras.layers.Conv1DTranspose(2*self.feature_num, 1)

        tcn_deconv1_ = tf.keras.layers.Conv1DTranspose(self.feature_num, 1)

        output = tcn_deconv1(inputs)
        output = tcn_deconv1_(output)

        return tf.keras.Model(inputs, output, name='tcn_deconv1')

    def tcn_encoder_second_last_level(self):
        """
        Implement tcn encoder
        """
        """
        define dilation for each layer(24 hours)
        """
        dilation1 = 1  # 3 hours
        dilation2 = 2  # 7hours
        dilation3 = 4  # 15hours
        dilation4 = 8  # with filter size 3, 8x3=24, already covers the whole time sequence
        # dilation5 = 16

        """
        define the identical resolution
        """
        inputs = layers.Input((self.time_sequence, self.feature_num))
        # tcn_conv0 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
        # dilation_rate=dilation1, padding='valid')
        tcn_conv0 = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu', dilation_rate=1, padding='valid')
        layernorm1 = tf.keras.layers.BatchNormalization()
        # padding_1 = (self.tcn_filter_size - 1) * dilation1
        # inputs1 = tf.pad(inputs, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_1)
        self.outputs0 = tcn_conv0(inputs)
        # self.outputs1 = conv1_identity(self.outputs1)

        """
        define the first tcn layer, dilation=1
        """
        # inputs = layers.Input((self.time_sequence,self.feature_num))
        tcn_conv1 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
                                           dilation_rate=dilation1, padding='valid')
        conv1_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu', dilation_rate=1)
        layernorm1 = tf.keras.layers.BatchNormalization()
        padding_1 = (self.tcn_filter_size - 1) * dilation1
        # inputs1 = tf.pad(inputs, tf.constant([[0,0],[1,0],[0,0]]) * padding_1)

        inputs1 = tf.pad(self.outputs0, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_1)
        self.outputs1 = tcn_conv1(inputs1)
        self.outputs1 = conv1_identity(self.outputs1)
        # self.outputs1 = layernorm1(self.outputs1)

        """
        define the second tcn layer, dilation=2
        """
        tcn_conv2 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
                                           dilation_rate=dilation2,
                                           padding='valid')
        conv2_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu',
                                                dilation_rate=1)
        layernorm2 = tf.keras.layers.BatchNormalization()
        padding_2 = (self.tcn_filter_size - 1) * dilation2
        inputs2 = tf.pad(self.outputs1, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_2)
        self.outputs2 = tcn_conv2(inputs2)
        self.outputs2 = conv2_identity(self.outputs2)
        # self.outputs2 = layernorm2(self.outputs2)

        """
        define the third tcn layer, dilation=4
        """
        tcn_conv3 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
                                           dilation_rate=dilation3,
                                           padding='valid')
        conv3_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu',
                                                dilation_rate=1)
        layernorm3 = tf.keras.layers.BatchNormalization()
        padding_3 = (self.tcn_filter_size - 1) * dilation3
        inputs3 = tf.pad(self.outputs2, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_3)
        self.outputs3 = tcn_conv3(inputs3)
        self.outputs3 = conv3_identity(self.outputs3)
        # self.outputs3 = layernorm3(self.outputs3)

        """
        fourth layer
        """
        tcn_conv4 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
                                           dilation_rate=dilation4,
                                           padding='valid')
        conv4_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu',
                                                dilation_rate=1)
        layernorm4 = tf.keras.layers.BatchNormalization()
        padding_4 = (self.tcn_filter_size - 1) * dilation4
        inputs4 = tf.pad(self.outputs3, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_4)
        self.outputs4 = tcn_conv4(inputs4)
        self.outputs4 = conv4_identity(self.outputs4)
        # self.outputs4 = layernorm4(self.outputs4)

        return tf.keras.Model(inputs,
                              [inputs, self.outputs4, self.outputs3, self.outputs2, self.outputs1, self.outputs0],
                              name='tcn_encoder')

    def lstm_split_multi(self):
        inputs = layers.Input((self.time_sequence, self.latent_dim))
        inputs2 = layers.Input((self.time_sequence, self.latent_dim))
        inputs3 = layers.Input((self.time_sequence, self.latent_dim))
        inputs4 = layers.Input((self.time_sequence, self.latent_dim))
        output1 = inputs[:, -1, :]
        output_query1 = tf.expand_dims(output1, 1)

        output2 = inputs[:, 0:-1, :]
        output3 = inputs2[:, 0:-1, :]
        output4 = inputs3[:, 0:-1, :]
        output5 = inputs4[:, 0:-1, :]
        # output2 = inputs

        return tf.keras.Model([inputs, inputs2, inputs3, inputs4],
                              [output_query1, output2, output3, output4, output5], name='lstm_split')


    def position_project_layer(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                # layers.Input((50)),
                layers.Dense(
                    self.latent_dim,
                    # use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )
            ],
            name="position_projection",
        )
        return model



    def position_encoding(self, pos):
        pos_embedding = np.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            if i % 2 == 0:
                pos_embedding[i] = np.sin(pos / (np.power(2, 2 * i / self.latent_dim)))
            else:
                pos_embedding[i] = np.cos(pos / (np.power(2, 2 * i / self.latent_dim)))

        return tf.math.l2_normalize(pos_embedding)


    def E_step_initial(self,batch_embedding,projection_basis):
        batch_embedding = tf.math.l2_normalize(batch_embedding, axis=1)

        #projection_basis = projection_basis[0]
        projection_basis = tf.expand_dims(projection_basis, 0)
        projection_basis = tf.broadcast_to(projection_basis,
                                           shape=(batch_embedding.shape[0],
                                                  projection_basis.shape[1], projection_basis.shape[2]))

        self.first_check_projection = projection_basis

        batch_embedding_whole = tf.reshape(batch_embedding,(batch_embedding.shape[0]*batch_embedding.shape[1],
                                                            batch_embedding.shape[2]))
        self.check_batch_embedding_whole = batch_embedding_whole

        batch_embedding = tf.expand_dims(batch_embedding, 2)
        batch_embedding = tf.broadcast_to(batch_embedding, [batch_embedding.shape[0],
                                                            batch_embedding.shape[1],
                                                            self.unsupervised_cluster_num,
                                                            self.latent_dim])

        self.check_batch_embedding_E = batch_embedding

        check_converge = 100 * np.ones((batch_embedding.shape[0] * batch_embedding.shape[1]))

        self.check_check_converge = check_converge

        check_converge_num = 1000
        self.check_converge_num = check_converge_num

        max_value_projection = 0

        while(check_converge_num > self.converge_threshold_E):
            print(check_converge_num)
            basis = tf.math.l2_normalize(projection_basis, axis=-1)
            self.check_basis = basis

            basis = tf.expand_dims(basis, 1)
            basis = tf.broadcast_to(basis, [batch_embedding.shape[0], batch_embedding.shape[1],
                                            self.unsupervised_cluster_num,
                                            self.latent_dim])
            basis = tf.cast(basis,tf.float64)
            self.check_basis_E = basis

            projection = tf.multiply(batch_embedding, basis)
            projection = tf.reduce_sum(projection, 3)

            self.check_projection_E = projection
            max_value_projection = np.argmax(projection, axis=2)
            self.check_max_value_projection = max_value_projection

            projection_basis_whole = tf.reshape(max_value_projection,
                                                (max_value_projection.shape[0]*max_value_projection.shape[1]))

            self.projection_basis_whole = projection_basis_whole

            semantic_cluster = []

            for i in range(self.unsupervised_cluster_num):
                semantic_index = np.where(self.projection_basis_whole == i)[0]
                semantic = tf.gather(batch_embedding_whole, semantic_index)
                semantic = tf.reduce_mean(semantic, 0)
                semantic_cluster.append(semantic)

            semantic_cluster = tf.stack(semantic_cluster, 0)

            self.check_semantic_cluster = semantic_cluster

            projection_basis = semantic_cluster

            projection_basis = tf.expand_dims(projection_basis, 0)
            projection_basis = tf.broadcast_to(projection_basis,
                                                shape=(batch_embedding.shape[0],
                                                      projection_basis.shape[1], projection_basis.shape[2]))



            #max_value_projection_reshape = tf.reshape(max_value_projection,
                                                     # (max_value_projection.shape[0]*max_value_projection.shape[1]))
            cluster_diff = projection_basis_whole - check_converge
            check_converge = projection_basis_whole

            self.check_cluster_diff = cluster_diff

            check_converge_num = len(np.where(cluster_diff !=0)[0])

        return max_value_projection, projection_basis


    def E_step(self, batch_embedding,projection_basis):
        batch_embedding = tf.math.l2_normalize(batch_embedding, axis=1)
        basis = tf.math.l2_normalize(projection_basis, axis=1)

        batch_embedding = tf.expand_dims(batch_embedding,2)
        batch_embedding = tf.broadcast_to(batch_embedding,[batch_embedding.shape[0],
                                                           batch_embedding.shape[1],
                                                           self.unsupervised_cluster_num,
                                                           self.latent_dim])
        self.check_batch_embedding_E = batch_embedding

        basis = tf.expand_dims(basis,0)
        basis = tf.expand_dims(basis,1)
        basis = tf.broadcast_to(basis,
                                [batch_embedding.shape[0],batch_embedding.shape[1],
                                 self.unsupervised_cluster_num,self.latent_dim])

        self.check_basis_E = basis

        projection = tf.multiply(batch_embedding, basis)
        projection = tf.reduce_sum(projection,3)

        self.check_projection_E = projection
        max_value_projection = np.argmax(projection, axis=2)
        self.check_max_value_projection = max_value_projection

        return max_value_projection


    def extract_temporal_semantic(self,x_batch_feature,on_site_time,x_batch_origin):
        temporal_semantic = \
            np.zeros((x_batch_feature.shape[0], self.semantic_positive_sample, 2, self.latent_dim))

        temporal_semantic_origin = \
            np.zeros((x_batch_feature.shape[0], self.semantic_positive_sample, 2, x_batch_origin.shape[2]))

        sample_sequence_batch = np.zeros((x_batch_feature.shape[0], self.semantic_positive_sample))
        for k in range(x_batch_feature.shape[0]):
            single_on_site = on_site_time[k]
            sample_sequence_feature = np.zeros((self.semantic_positive_sample, 2, self.latent_dim))
            sample_sequence_origin = np.zeros((self.semantic_positive_sample, 2, x_batch_origin.shape[2]))
            if single_on_site == 0:
                sample_sequence = np.zeros(self.semantic_positive_sample)
            elif single_on_site < self.semantic_positive_sample:
                sample_sequence = np.zeros(self.semantic_positive_sample)
                for j in range(self.semantic_positive_sample):
                    sample_sequence[j] = random.randint(0, int(single_on_site))
            else:
                if single_on_site > 46:
                    single_on_site = 46
                sample_sequence = random.sample(range(0, int(single_on_site)), self.semantic_positive_sample)

            sample_sequence_batch[k, :] = sample_sequence

            for j in range(self.semantic_positive_sample):
                sample_sequence_feature[j, 0, :] = x_batch_feature[k, int(sample_sequence[j]), :]
                sample_sequence_feature[j, 1, :] = x_batch_feature[k, int(sample_sequence[j]) + 1, :]
                sample_sequence_origin[j, 0, :] = x_batch_origin[k, int(sample_sequence[j]), :]
                sample_sequence_origin[j, 1, :] = x_batch_origin[k, int(sample_sequence[j]) + 1, :]


            temporal_semantic[k, :, :, :] = sample_sequence_feature
            temporal_semantic_origin[k, :, :, :] = sample_sequence_origin
            #self.check_temporal_semantic = temporal_semantic

        return temporal_semantic,sample_sequence_batch, temporal_semantic_origin


    def train_semantic_basis(self):
        self.tcn = self.tcn_encoder_second_last_level()
        self.auc_all = []
        self.loss_track = []
        self.loss_track_mse = []
        self.projection_layer = self.project_logit()
        self.position_project = self.position_project_layer()
        self.bceloss = tf.keras.losses.BinaryCrossentropy()
        #self.basis_model = self.projection_model()
        self.tcn_1_lvl = self.one_h_resolution_deconv()
        position_encode_1 = self.position_encoding(1)
        position_encode_2 = self.position_encoding(2)

        position_encode_whole = np.zeros((2,self.latent_dim))
        position_encode_whole[0,:] = position_encode_1
        position_encode_whole[1,:] = position_encode_2

        position_encode_whole = tf.expand_dims(position_encode_whole,0)
        position_encode_whole = tf.expand_dims(position_encode_whole,0)

        #projection_basis = self.init_projection_basis
        #projection_basis = tf.cast(projection_basis,tf.float64)

        #projection_basis = tf.expand_dims(projection_basis, 0)

        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            # extract_val, global_val,k = self.model_extractor(self.val_data)
            tcn_temporal_output_val = self.tcn(self.val_data)
            last_layer_output_val = tcn_temporal_output_val[1]
            on_site_extract_val = [last_layer_output_val[i, np.abs(int(self.val_on_site_time[i]) - 1), :] for i in
                                   range(self.val_on_site_time.shape[0])]
            on_site_extract_array_val = tf.stack(on_site_extract_val)
            prediction_val = self.projection_layer(on_site_extract_array_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit, prediction_val)
            print("auc")
            print(val_acc)
            self.auc_all.append(val_acc)

            tcn_temporal_output_whole = self.tcn(self.train_data)
            output_1h_resolution_whole = tcn_temporal_output_whole[5]
            #on_site_extract_whole = [last_layer_output[i, np.abs(int(self.train_on_site_time[i] - 1)), :] for i in
                               #range(self.train_on_site_time.shape[0])]

            temporal_semantic_whole, sample_sequence_batch_whole, temporal_semantic_origin_whole = \
                self.extract_temporal_semantic(output_1h_resolution_whole, self.train_on_site_time, self.train_data)

            temporal_semantic_whole_ = tf.reshape(temporal_semantic_whole,
                                            (temporal_semantic_whole.shape[0],
                                             temporal_semantic_whole.shape[1] *
                                             temporal_semantic_whole.shape[2],
                                             temporal_semantic_whole.shape[3]))

           # projection_basis = self.init_projection_basis
            #projection_basis = tf.expand_dims(projection_basis, 0)
            if epoch == 0:
                order_input_total_init, projection_basis_total = \
                    self.E_step_initial(temporal_semantic_whole_, self.init_projection_basis)
            else:
                order_input_total_init, projection_basis_total = \
                    self.E_step_initial(temporal_semantic_whole_, self.check_projection_basis)

            order_input_total_init = tf.reshape(order_input_total_init,
                                           (temporal_semantic_whole.shape[0],
                                           temporal_semantic_whole.shape[1],
                                           temporal_semantic_whole.shape[2]))

            self.check_order_input_total_init = order_input_total_init

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.train_data, self.train_logit, self.train_on_site_time,
                 #order_input_total, temporal_semantic_whole,
                 projection_basis_total))
            self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)


            for step, (x_batch_train, y_batch_train, on_site_time, projection_basis) in enumerate(self.train_dataset):
                self.check_x_batch = x_batch_train
                self.check_on_site_time = on_site_time
                self.check_label = y_batch_train

                random_indices_cohort = np.random.choice(self.num_cohort, size=x_batch_train.shape[0], replace=False)
                random_indices_control = np.random.choice(self.num_control, size=x_batch_train.shape[0], replace=False)

                x_batch_train_cohort = self.memory_bank_cohort[random_indices_cohort, :, :]
                x_batch_train_control = self.memory_bank_control[random_indices_control, :, :]
                on_site_time_cohort = self.memory_bank_cohort_on_site[random_indices_cohort]
                on_site_time_control = self.memory_bank_control_on_site[random_indices_control]

                input_projection_batch = np.ones((self.unsupervised_cluster_num, self.latent_dim))


                with tf.GradientTape() as tape:
                    tcn_temporal_output = self.tcn(x_batch_train)
                    tcn_temporal_output_cohort = self.tcn(x_batch_train_cohort)
                    tcn_temporal_output_control = self.tcn(x_batch_train_control)

                    #projection_basis = self.basis_model(input_projection_batch)
                    projection_basis = tf.cast(projection_basis[0],tf.float64)
                    self.check_output = tcn_temporal_output
                    last_layer_output = tcn_temporal_output[1]
                    out_put_1h_resolution = tcn_temporal_output[5]
                    out_put_1h_resolution_cohort = tcn_temporal_output_cohort[5]
                    out_put_1h_resolution_control = tcn_temporal_output_control[5]
                    on_site_extract = [last_layer_output[i, np.abs(int(on_site_time[i] - 1)), :] for i in
                                       range(on_site_time.shape[0])]
                    temporal_semantic, sample_sequence_batch, temporal_semantic_origin = \
                        self.extract_temporal_semantic(out_put_1h_resolution,on_site_time,x_batch_train)
                    temporal_semantic_cohort, sample_sequence_batch_cohort, temporal_semantic_origin_cohort = \
                        self.extract_temporal_semantic(out_put_1h_resolution_cohort, on_site_time_cohort,x_batch_train_cohort)
                    temporal_semantic_control, sample_sequence_batch_control, temporal_semantic_origin_control = \
                        self.extract_temporal_semantic(out_put_1h_resolution_control, on_site_time_control,x_batch_train_control)

                    temporal_semantic_ = tf.reshape(temporal_semantic,
                                                    (temporal_semantic.shape[0],
                                                     temporal_semantic.shape[1]*
                                                     temporal_semantic.shape[2],
                                                     temporal_semantic.shape[3]))

                    temporal_semantic_cohort_ = tf.reshape(temporal_semantic_cohort,
                                                    (temporal_semantic_cohort.shape[0],
                                                     temporal_semantic_cohort.shape[1] *
                                                     temporal_semantic_cohort.shape[2],
                                                     temporal_semantic_cohort.shape[3]))

                    temporal_semantic_control_ = tf.reshape(temporal_semantic_control,
                                                           (temporal_semantic_control.shape[0],
                                                            temporal_semantic_control.shape[1]*
                                                            temporal_semantic_control.shape[2],
                                                            temporal_semantic_control.shape[3]))

                    self.check_temporal_semantic = temporal_semantic
                    self.check_temporal_semantic_ = temporal_semantic_

                    self.reconstruct_temporal_semantic = tf.reshape(temporal_semantic_,
                                                                    (temporal_semantic.shape[0] *
                                                                     temporal_semantic.shape[1] *
                                                                     temporal_semantic.shape[2],
                                                                     temporal_semantic.shape[3]))

                    self.reconstruct_temporal_semantic_origin = tf.reshape(temporal_semantic_origin,
                                                                           (temporal_semantic_origin.shape[0] *
                                                                            temporal_semantic_origin.shape[1] *
                                                                            temporal_semantic_origin.shape[2],
                                                                            temporal_semantic_origin.shape[3]))

                    order_input_total = self.E_step(temporal_semantic_, projection_basis)

                    order_input_total_cohort = self.E_step(temporal_semantic_cohort_, projection_basis)

                    order_input_total_control = self.E_step(temporal_semantic_control_, projection_basis)

                    self.check_projection_basis = projection_basis

                    #order_input_total_ = tf.reshape(order_input_total,(temporal_semantic_control.shape[0],
                                                                      #temporal_semantic_control.shape[1],
                                                                      #temporal_semantic_control.shape[2]))

                    self.check_order_input_total = order_input_total

                    self.check_sample_sequence_batch = sample_sequence_batch

                    projection_basis = tf.expand_dims(projection_basis,0)

                    self.total_sementic = []
                    self.total_semantic_cohort = []
                    self.total_semantic_control = []
                    for i in range(self.unsupervised_cluster_num):
                        check = order_input_total == i
                        check = tf.cast(check, tf.float64)
                        check = tf.expand_dims(check, 2)
                        self.check_check = check

                        check_cohort = order_input_total_cohort == i
                        check_cohort = tf.cast(check_cohort, tf.float64)
                        check_cohort = tf.expand_dims(check_cohort, 2)
                        self.check_check_cohort = check_cohort

                        check_control = order_input_total_control == i
                        check_control = tf.cast(check_control, tf.float64)
                        check_control = tf.expand_dims(check_control, 2)
                        self.check_check_control = check_control

                        projection_basis_single = tf.expand_dims(projection_basis[:, i, :], 1)
                        projection_single = tf.broadcast_to(projection_basis_single, shape=(check.shape[0],
                                                                                            check.shape[1],
                                                                                            projection_basis.shape[2]))

                        projection_single_cohort = tf.broadcast_to(projection_basis_single, shape=(check_cohort.shape[0],
                                                                                            check_cohort.shape[1],
                                                                                            projection_basis.shape[2]))

                        projection_single_control = tf.broadcast_to(projection_basis_single,
                                                                   shape=(check_control.shape[0],
                                                                          check_control.shape[1],
                                                                          projection_basis.shape[2]))

                        self.check_projection_single = projection_single
                        batch_semantic_embedding_single = tf.math.multiply(projection_single,
                                                                           check)
                        self.check_batch_semantic_embedding_single = batch_semantic_embedding_single

                        self.total_sementic.append(batch_semantic_embedding_single)

                        batch_semantic_embedding_single_cohort = tf.math.multiply(projection_single_cohort,
                                                                           check_cohort)

                        self.total_semantic_cohort.append(batch_semantic_embedding_single_cohort)

                        batch_semantic_embedding_single_control = tf.math.multiply(projection_single_control,
                                                                                  check_control)

                        self.total_semantic_control.append(batch_semantic_embedding_single_control)

                    batch_semantic_embedding_whole_ = tf.stack(self.total_sementic)
                    batch_semantic_embedding_whole = tf.reduce_sum(batch_semantic_embedding_whole_,axis=0)
                    self.check_batch_semantic_embedding_whole = batch_semantic_embedding_whole

                    batch_semantic_embedding_whole_cohort_ = tf.stack(self.total_semantic_cohort)
                    batch_semantic_embedding_whole_cohort = tf.reduce_sum(batch_semantic_embedding_whole_cohort_,axis=0)
                    self.check_batch_semantic_embedding_whole_cohort = batch_semantic_embedding_whole_cohort

                    batch_semantic_embedding_whole_control_ = tf.stack(self.total_semantic_control)
                    batch_semantic_embedding_whole_control = tf.reduce_sum(batch_semantic_embedding_whole_control_, axis=0)
                    self.check_batch_semantic_embedding_whole_control = batch_semantic_embedding_whole_control

                    batch_semantic_embedding_whole = tf.reshape(batch_semantic_embedding_whole,(batch_semantic_embedding_whole.shape[0],
                                                                                                self.semantic_positive_sample,
                                                                                                2,
                                                                                                self.latent_dim))

                    position_encode_batch = tf.broadcast_to(position_encode_whole,shape=(x_batch_train.shape[0],
                                                                                         self.semantic_positive_sample,
                                                                                         2,
                                                                                         self.latent_dim))

                    semantic_input = tf.math.add(batch_semantic_embedding_whole,position_encode_batch)

                    batch_semantic_temporal_feature_seperate = self.position_project(semantic_input)
                    batch_semantic_temporal_feature = tf.reduce_sum(batch_semantic_temporal_feature_seperate,2)
                    self.check_semantic_separate = batch_semantic_temporal_feature_seperate
                    self.check_batch_semantic_temporal_feature = batch_semantic_temporal_feature

                    """
                    Semantic for cohort
                    """
                    batch_semantic_embedding_whole_cohort = tf.reshape(batch_semantic_embedding_whole_cohort,
                                                                       (batch_semantic_embedding_whole_cohort.shape[0],
                                                                                                self.semantic_positive_sample,
                                                                                                2,
                                                                                                self.latent_dim))


                    semantic_input_cohort = tf.math.add(batch_semantic_embedding_whole_cohort,position_encode_batch)

                    batch_semantic_temporal_feature_seperate_cohort = self.position_project(semantic_input_cohort)
                    batch_semantic_temporal_feature_cohort = tf.reduce_sum(batch_semantic_temporal_feature_seperate_cohort,2)


                    """
                    Semantic for control
                    """
                    batch_semantic_embedding_whole_control = tf.reshape(batch_semantic_embedding_whole_control,
                                                                       (batch_semantic_embedding_whole_control.shape[0],
                                                                        self.semantic_positive_sample,
                                                                        2,
                                                                        self.latent_dim))

                    semantic_input_control = tf.math.add(batch_semantic_embedding_whole_control, position_encode_batch)

                    batch_semantic_temporal_feature_seperate_control = self.position_project(semantic_input_control)
                    batch_semantic_temporal_feature_control = tf.reduce_sum(batch_semantic_temporal_feature_seperate_control,
                                                                           2)

                    on_site_extract_array = tf.stack(on_site_extract)
                    prediction = self.projection_layer(on_site_extract_array)
                    bceloss = self.bceloss(y_batch_train, prediction)
                    self.check_prediction = prediction

                    semantic_time_progression_loss = self.info_nce_loss(on_site_extract_array, batch_semantic_temporal_feature,
                                                                        batch_semantic_temporal_feature_cohort,
                                                                        batch_semantic_temporal_feature_control,
                                                                        y_batch_train)

                    unsupervised_loss = self.unsupervised_prototype_loss(self.check_temporal_semantic_,
                                                                         self.check_projection_basis,
                                                                         self.check_order_input_total)

                    self.check_un_loss = unsupervised_loss
                    self.check_bce_loss = bceloss

                    bceloss = tf.cast(bceloss,tf.float64)
                    semantic_time_progression_loss = tf.cast(semantic_time_progression_loss,tf.float64)



                    loss = 0.6*semantic_time_progression_loss# + 0.2*unsupervised_loss

                gradients = \
                    tape.gradient(loss,
                                  self.tcn.trainable_variables)# + self.projection_layer.trainable_weights+
                                  #self.position_project.trainable_weights) #+ self.basis_model.trainable_weights)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients,
                                            self.tcn.trainable_variables))#+ self.projection_layer.trainable_weights+
                                  #self.position_project.trainable_weights))# + self.basis_model.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)

                with tf.GradientTape() as tape:
                    self.reconstruct_temporal_semantic = tf.expand_dims(self.reconstruct_temporal_semantic,1)

                    self.reconstruct_temporal_semantic_origin = \
                        tf.expand_dims(self.reconstruct_temporal_semantic_origin,1)

                    reconstruct_1_lvl = self.tcn_1_lvl(self.reconstruct_temporal_semantic)

                    self.check_reconstruct_1_lvl = reconstruct_1_lvl

                    mseloss = tf.keras.losses.MeanSquaredError()(self.reconstruct_temporal_semantic_origin, reconstruct_1_lvl)

                trainable_weights = self.tcn_1_lvl.trainable_weights

                gradients = tape.gradient(mseloss, trainable_weights)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients, trainable_weights))
                #self.check_loss = loss

                if step % 10 == 0:
                    print("Training loss(for one batch, mse) at step %d: %.4f"
                          % (step, float(mseloss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track_mse.append(mseloss)

    def reconstruct_semantic_1_lvl(self):
        self.reconstruct_semantic_origin = np.zeros((self.unsupervised_cluster_num, 1, self.feature_num))
        projection_basis = tf.expand_dims(self.check_projection_basis,1)
        self.reconstruct_semantic_norm = self.tcn_1_lvl(projection_basis)

        for k in range(self.unsupervised_cluster_num):
            for i in range(self.feature_num):
                if self.read_d.std_all[i] == 0:
                    self.reconstruct_semantic_origin[:,:,i] = self.read_d.ave_all[i]
                else:
                    self.reconstruct_semantic_origin[k, 0, i] = \
                        (self.reconstruct_semantic_norm[k, 0, i] * self.read_d.std_all[i]) + self.read_d.ave_all[i]


    def semantic_combination_recommandation(self):
        position_encode_1 = self.position_encoding(1)
        position_encode_2 = self.position_encoding(2)

        position_encode_whole = np.zeros((2,self.latent_dim))
        position_encode_whole[0,:] = position_encode_1
        position_encode_whole[1,:] = position_encode_2

        position_encode_whole = tf.expand_dims(position_encode_whole,0)
        position_encode_whole = tf.expand_dims(position_encode_whole,0)

        tcn_temporal_output_cohort = self.tcn(self.memory_bank_cohort)[1]
        tcn_temporal_output_control = self.tcn(self.memory_bank_control)[1]
        sequence = np.array(range(self.unsupervised_cluster_num))

        semantic_sequence_projection = []
        semantic_projection = []
        for i in range(self.unsupervised_cluster_num):
            projection_single = tf.broadcast_to(tf.expand_dims(self.check_projection_basis[i,:],0),
                                                (self.unsupervised_cluster_num,
                                                 self.latent_dim))
            sequence_single = tf.broadcast_to(tf.expand_dims(sequence[i],0),(self.unsupervised_cluster_num,))

            semantic_projection_single = tf.stack([projection_single,self.check_projection_basis],1)

            semantic_sequence_projection_single = tf.stack([sequence_single,sequence])

            semantic_sequence_projection.append(semantic_sequence_projection_single)

            semantic_projection.append(semantic_projection_single)

        self.check_semantic_projection_single = semantic_projection_single
        semantic_sequence_projection = tf.stack(semantic_sequence_projection)
        semantic_sequence_projection = tf.transpose(semantic_sequence_projection,(0,2,1))
        semantic_projection = tf.stack(semantic_projection)

        position_encode_whole = tf.broadcast_to(position_encode_whole,(semantic_projection.shape[0],
                                                                       semantic_projection.shape[1],
                                                                       position_encode_whole.shape[2],
                                                                       position_encode_whole.shape[3]))

        temporal_semantic_encode_input = tf.math.add(semantic_projection,position_encode_whole)

        temporal_semantic_encode_feature = self.position_project(temporal_semantic_encode_input)

        temporal_semantic_encode_feature = tf.reduce_sum(temporal_semantic_encode_feature, 2)

        self.check_semantic_sequence_projection = tf.reshape(semantic_sequence_projection,
                                                             (semantic_sequence_projection.shape[0]*
                                                              semantic_sequence_projection.shape[1],
                                                              semantic_sequence_projection.shape[2]))
        self.check_semantic_projection = semantic_projection
        self.check_semantic_encode_feature = tf.reshape(temporal_semantic_encode_feature,
                                                        (temporal_semantic_encode_feature.shape[0]*
                                                         temporal_semantic_encode_feature.shape[1],
                                                         temporal_semantic_encode_feature.shape[2]))

        self.check_semantic_encode_feature = tf.math.l2_normalize(self.check_semantic_encode_feature, axis=-1)

        on_site_extract_cohort = [tcn_temporal_output_cohort[i,
                                  np.abs(int(self.memory_bank_cohort_on_site[i] - 1)), :] for i in
                           range(self.memory_bank_cohort_on_site.shape[0])]
        on_site_extract_control = [tcn_temporal_output_control[i,
                                   np.abs(int(self.memory_bank_control_on_site[i] - 1)), :] for i in
                           range(self.memory_bank_control_on_site.shape[0])]

        self.check_on_site_extract_cohort = tf.math.l2_normalize(tf.stack(on_site_extract_cohort),axis=-1)
        self.check_on_site_extract_control = tf.math.l2_normalize(tf.stack(on_site_extract_control),axis=-1)

        self.score_cohort = tf.matmul(self.check_semantic_encode_feature,
                                      tf.transpose(self.check_on_site_extract_cohort))
        self.score_control = tf.matmul(self.check_semantic_encode_feature,
                                       tf.transpose(self.check_on_site_extract_control))

        self.score_cohort = tf.reduce_mean(self.score_cohort,1)
        self.score_control = tf.reduce_mean(self.score_control,1)


    def train_standard(self):
        # input = layers.Input((self.time_sequence, self.feature_num))
        self.tcn = self.tcn_encoder_second_last_level()
        # tcn = self.tcn(input)
        self.auc_all = []
        self.loss_track = []
        # self.model_extractor = tf.keras.Model(input, tcn, name="time_extractor")
        self.projection_layer = self.project_logit()
        self.bceloss = tf.keras.losses.BinaryCrossentropy()

        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            # extract_val, global_val,k = self.model_extractor(self.val_data)
            tcn_temporal_output_val = self.tcn(self.val_data)
            last_layer_output_val = tcn_temporal_output_val[1]
            on_site_extract_val = [last_layer_output_val[i, np.abs(int(self.val_on_site_time[i]) - 1), :] for i in
                                   range(self.val_on_site_time.shape[0])]
            on_site_extract_array_val = tf.stack(on_site_extract_val)
            prediction_val = self.projection_layer(on_site_extract_array_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit, prediction_val)
            print("auc")
            print(val_acc)
            self.auc_all.append(val_acc)
            for step, (x_batch_train, y_batch_train, on_site_time, x_batch_origin) in enumerate(self.train_dataset):
                self.check_x_batch = x_batch_train
                self.check_on_site_time = on_site_time
                self.check_label = y_batch_train
                with tf.GradientTape() as tape:
                    tcn_temporal_output = self.tcn(x_batch_train)
                    self.check_output = tcn_temporal_output
                    last_layer_output = tcn_temporal_output[1]
                    on_site_extract = [last_layer_output[i, int(on_site_time[i] - 1), :] for i in
                                       range(on_site_time.shape[0])]
                    on_site_extract_array = tf.stack(on_site_extract)
                    prediction = self.projection_layer(on_site_extract_array)
                    loss = self.bceloss(y_batch_train, prediction)
                    self.check_prediction = prediction

                gradients = \
                    tape.gradient(loss,
                                  self.tcn.trainable_variables + self.projection_layer.trainable_weights)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients,
                                              self.tcn.trainable_variables + self.projection_layer.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)

    def project_logit(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                # layers.Input((50)),
                layers.Dense(
                    1,
                    # use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='sigmoid'
                )
            ],
            name="predictor",
        )
        return model



