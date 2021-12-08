from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.cluster import KMeans
from tcn import TCN

import matplotlib.pyplot as plt
import numpy as np

class seq_seq_ehr():
    def __init__(self, read_d):
        self.read_d = read_d
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
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 20
        self.pre_train_epoch = 3
        self.latent_dim = 100
        self.tau = 1
        self.time_sequence = self.read_d.time_sequence
        self.tcn_filter_size = 3

        self.steps = self.length_train // self.batch_size
        self.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, decay_steps=self.steps)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=self.steps,
            decay_rate=0.5)

        self.resp_rate_range = [12,16]
        self.heart_rate_range = [60,100]
        self.temp_range = [36,37]
        self.map_range = [70,100]
        self.sbp_range = [10,120]
        self.dbp_range = [10,80]
        self.relations = [[1, 1], [0, 1], [1, 0], [0.0], [1, 2], [2, 1], [2, 2], [0, 2], [2, 0]]

    def aquire_data(self, starting_index, data_set,length):
        data = np.zeros((length,self.time_sequence,35))
        data_sofa_score = np.zeros((length,4))
        data_sofa = np.zeros((length,4))
        logit_dp = np.zeros((length,1))
        for i in range(length):
            name = data_set[starting_index+i]
            self.read_d.read_table(name)
            one_data = self.read_d.one_data_tensor
            #one_data = np.mean(one_data,0)
            data[i,:,:] = one_data
            logit_dp[i,0] = self.read_d.one_data_logit
            data_sofa_score[i,:] = self.read_d.one_data_sofa_score
            data_sofa[i,:] = self.read_d.one_data_sofa

        logit = logit_dp[:,0]
        return (data, logit,data_sofa,data_sofa_score)

    def create_memory_bank(self):
        #self.train_data, self.train_logit,self.train_sofa,self.train_sofa_score = self.aquire_data(0, self.train_data, self.length_train)
        #self.test_data, self.test_logit,self.test_sofa,self.test_sofa_score = self.aquire_data(0, self.test_data, self.length_test)
        #self.val_data, self.val_logit,self.val_sofa,self.val_sofa_score = self.aquire_data(0, self.validate_data, self.length_val)

        file_path = '/home/tingyi/physionet_data/'
        with open(file_path + 'train.npy', 'rb') as f:
            self.train_data = np.load(f)
        with open(file_path + 'train_logit.npy', 'rb') as f:
            self.train_logit = np.load(f)
        with open(file_path + 'test.npy', 'rb') as f:
            self.test_data = np.load(f)
        with open(file_path + 'test_logit.npy', 'rb') as f:
            self.test_logit = np.load(f)
        with open(file_path + 'val.npy', 'rb') as f:
            self.val_data = np.load(f)
        with open(file_path + 'val_logit.npy', 'rb') as f:
            self.val_logit = np.load(f)
        with open(file_path + 'train_sofa_score.npy', 'rb') as f:
            self.train_sofa_score = np.load(f)
        with open(file_path + 'train_sofa.npy', 'rb') as f:
            self.train_sofa = np.load(f)

        with open(file_path + 'train_origin_35.npy', 'rb') as f:
            self.train_data_origin = np.load(f)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_logit,self.train_sofa_score))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        cohort_index = np.where(self.train_logit == 1)[0]
        control_index = np.where(self.train_logit == 0)[0]
        self.memory_bank_cohort = self.train_data[cohort_index,:,:]
        self.memory_bank_control = self.train_data[control_index,:,:]

    def create_memory_bank_prospect(self):
        #self.train_data, self.train_logit,self.train_sofa,self.train_sofa_score = self.aquire_data(0, self.train_data, self.length_train)
        #self.test_data, self.test_logit,self.test_sofa,self.test_sofa_score = self.aquire_data(0, self.test_data, self.length_test)
        #self.val_data, self.val_logit,self.val_sofa,self.val_sofa_score = self.aquire_data(0, self.validate_data, self.length_val)

        file_path = '/home/tingyi/physionet_data/'
        with open(file_path + 'train_34.npy', 'rb') as f:
            self.train_data = np.load(f)
        with open(file_path + 'train_logit_34.npy', 'rb') as f:
            self.train_logit = np.load(f)
        with open(file_path + 'train_origin_34.npy', 'rb') as f:
            self.train_data_origin = np.load(f)
        with open(file_path + 'train_mask_34.npy', 'rb') as f:
            self.train_mask = np.load(f)

        with open(file_path + 'val_34.npy', 'rb') as f:
            self.val_data = np.load(f)
        with open(file_path + 'val_logit_34.npy', 'rb') as f:
            self.val_logit = np.load(f)
        with open(file_path + 'val_origin_34.npy', 'rb') as f:
            self.val_data_origin = np.load(f)
        with open(file_path + 'val_mask_34.npy', 'rb') as f:
            self.val_mask = np.load(f)

        #index = np.array(self.read_d.missingness_95)
        #self.train_data = self.train_data[:,:,index]
        #self.val_data = self.val_data[:,:,index]


        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_logit))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        #cohort_index = np.where(self.train_logit == 1)[0]
        #control_index = np.where(self.train_logit == 0)[0]
        #self.memory_bank_cohort = self.train_data[cohort_index,:,:]
        #self.memory_bank_control = self.train_data[control_index,:,:]

    def compute_positive_pair(self,z,p):
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)

        positive_dot_prod = tf.multiply(z,p)
        positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(positive_dot_prod,1)/self.tau)

        return positive_dot_prod_sum

    def compute_negative_paris(self,z):
        z = tf.math.l2_normalize(z, axis=1)
        similarity_matrix = tf.matmul(z, tf.transpose(z))
        mask = tf.linalg.diag(tf.zeros(z.shape[0]),padding_value=1)

        negative_dot_prods = tf.multiply(similarity_matrix,mask)
        negative_dot_prods_sum =tf.reduce_sum(tf.math.exp(negative_dot_prods/self.tau),1)

        return negative_dot_prods_sum

    def info_nce_loss(self,z,p):
        positive_dot_prod_sum = self.compute_positive_pair(z,p)
        negative_dot_prod_sum = self.compute_negative_paris(z)

        denominator = tf.math.add(positive_dot_prod_sum,negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum,denominator))
        loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum,denominator),0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log,0))

        return loss, loss_prob

    def compute_positive_pair_infomax(self,z,p):
        z = tf.math.l2_normalize(z, axis=1)
        p = tf.math.l2_normalize(p, axis=1)

        z_expand = tf.expand_dims(z,1)
        z_expand_broad = tf.broadcast_to(z_expand, p.shape)

        positive_dot_prod = tf.multiply(z_expand_broad,p)
        positive_dot_prod_exp = tf.math.exp(tf.reduce_sum(positive_dot_prod,2)/self.tau)

        self.check_pos = positive_dot_prod_exp

        return positive_dot_prod_exp

    def compute_negative_pair_infomax(self,z,p):
        z = tf.math.l2_normalize(z, axis=1)
        p_reshape = tf.reshape(p,[p.shape[0]*p.shape[1],p.shape[2]])
        similarity_matrix = tf.matmul(z, tf.transpose(p_reshape))
        self.check_sim = similarity_matrix
        mask = tf.linalg.diag(tf.zeros(z.shape[0]), padding_value=1)
        mask_b = tf.broadcast_to(tf.expand_dims(mask,1),[z.shape[0],p.shape[1],z.shape[0]])
        mask_b_reshape = tf.transpose(tf.reshape(mask_b,[mask_b.shape[0]*mask_b.shape[1],mask_b.shape[2]]))
        self.check_mask_b = mask_b_reshape

        negative_dot_prods = tf.multiply(similarity_matrix, mask_b_reshape)
        negative_dot_prods_sum = tf.reduce_sum(tf.math.exp(negative_dot_prods / self.tau), 1)

        return negative_dot_prods_sum

    def info_nce_loss_infomax(self,z,p):
        positive_dot_prod_sum = self.compute_positive_pair_infomax(z, p)
        negative_dot_prod_sum = self.compute_negative_pair_infomax(z, p)

        negative_dot_prod_sum_extend = tf.expand_dims(negative_dot_prod_sum, 1)
        negative_dot_prod_sum_extend = tf.broadcast_to(negative_dot_prod_sum_extend, positive_dot_prod_sum.shape)
        self.negative_check_extend = negative_dot_prod_sum_extend
        denominator = tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum_extend)
        nomalized_prob_log = tf.reduce_sum(tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator)), 1)
        loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum, denominator), 0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss, loss_prob


    def compute_positive_pair_cluster(self,z,p,label):
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        similarity_matrix = tf.matmul(z, tf.transpose(p))
        mask = tf.linalg.diag(tf.zeros(z.shape[0]), padding_value=1)
        similarity_matrix = tf.multiply(similarity_matrix,mask)
        self.sim_check = similarity_matrix
        mask_label = np.zeros((z.shape[0],z.shape[0]))
        for i in range(z.shape[0]):
            data_cluster = label[i]
            cluster_index = list(np.where(label==data_cluster)[0])
            mask_label[i,:][cluster_index] = 1
            self.check_cluster = cluster_index

        self.check_mask_label = mask_label
        positive_dot_prod_sum_ = tf.multiply(similarity_matrix,mask_label)
        positive_dot_prod_sum = tf.math.exp(positive_dot_prod_sum_/self.tau)

        self.positive_check = positive_dot_prod_sum

        return positive_dot_prod_sum

    def compute_negative_paris_cluster(self,z,label):
        z = tf.math.l2_normalize(z, axis=1)
        similarity_matrix = tf.matmul(z, tf.transpose(z))
        mask = tf.linalg.diag(tf.zeros(z.shape[0]), padding_value=1)
        similarity_matrix = tf.multiply(similarity_matrix, mask)
        mask_label = np.zeros((z.shape[0],z.shape[0]))
        for i in range(z.shape[0]):
            data_cluster = label[i]
            cluster_index = list(np.where(label != data_cluster)[0])
            mask_label[i,:][cluster_index] = 1

        negative_dot_prod_sum_ = tf.multiply(similarity_matrix,mask_label)
        negative_dot_prod_sum = tf.reduce_sum(tf.math.exp(negative_dot_prod_sum_ / self.tau), 1)


        self.negative_check = negative_dot_prod_sum

        return negative_dot_prod_sum

    def info_nce_loss_cluster(self,z,p,label):
        positive_dot_prod_sum = self.compute_positive_pair_cluster(z,p,label)
        negative_dot_prod_sum = self.compute_negative_paris_cluster(z,label)

        negative_dot_prod_sum_extend = tf.expand_dims(negative_dot_prod_sum,1)
        negative_dot_prod_sum_extend = tf.broadcast_to(negative_dot_prod_sum_extend,positive_dot_prod_sum.shape)
        self.negative_check_extend = negative_dot_prod_sum_extend
        denominator = tf.math.add(positive_dot_prod_sum,negative_dot_prod_sum_extend)
        nomalized_prob_log = tf.reduce_sum(tf.math.log(tf.math.divide(positive_dot_prod_sum,denominator)),1)
        loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum,denominator),0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log,0))

        return loss, loss_prob

    def tcn_encoder_second_last_level(self):
        """
        Implement tcn encoder
        """
        """
        define dilation for each layer(24 hours)
        """
        dilation1 = 1
        dilation2 = 2
        dilation3 = 4
        dilation4 = 8 # with filter size 3, 8x3=24, already covers the whole time sequence

        """
        define the first tcn layer, dilation=1
        """
        inputs = layers.Input((self.time_sequence,35))
        tcn_conv1 = tf.keras.layers.Conv1D(self.latent_dim,self.tcn_filter_size,activation='relu',dilation_rate=dilation1,padding='valid')
        conv1_identity = tf.keras.layers.Conv1D(self.latent_dim,1,activation='relu',dilation_rate=1)
        layernorm1 = tf.keras.layers.BatchNormalization()
        padding_1 = (self.tcn_filter_size-1) * dilation1
        inputs1 = tf.pad(inputs, tf.constant([[0,0],[1,0],[0,0]]) * padding_1)
        self.outputs1 = tcn_conv1(inputs1)
        self.outputs1 = conv1_identity(self.outputs1)
        #self.outputs1 = layernorm1(self.outputs1)

        """
        define the second tcn layer, dilation=2
        """
        tcn_conv2 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu', dilation_rate=dilation2,
                                           padding='valid')
        conv2_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu',
                                                dilation_rate=1)
        layernorm2 = tf.keras.layers.BatchNormalization()
        padding_2 = (self.tcn_filter_size - 1) * dilation2
        inputs2 = tf.pad(self.outputs1, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_2)
        self.outputs2 = tcn_conv2(inputs2)
        self.outputs2 = conv2_identity(self.outputs2)
        #self.outputs2 = layernorm2(self.outputs2)

        """
        define the third tcn layer, dilation=4
        """
        tcn_conv3 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu', dilation_rate=dilation3,
                                           padding='valid')
        conv3_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu',
                                                dilation_rate=1)
        layernorm3 = tf.keras.layers.BatchNormalization()
        padding_3 = (self.tcn_filter_size - 1) * dilation3
        inputs3 = tf.pad(self.outputs2, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_3)
        self.outputs3 = tcn_conv3(inputs3)
        self.outputs3 = conv3_identity(self.outputs3)
        #self.outputs3 = layernorm3(self.outputs3)


        """
        fourth layer
        """
        tcn_conv4 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu', dilation_rate=dilation4,
                                           padding='valid')
        conv4_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu',
                                                dilation_rate=1)
        layernorm4 = tf.keras.layers.BatchNormalization()
        padding_4 = (self.tcn_filter_size - 1) * dilation4
        inputs4 = tf.pad(self.outputs3, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_4)
        self.outputs4 = tcn_conv4(inputs4)
        self.outputs4 = conv4_identity(self.outputs4)
        #self.outputs4 = layernorm4(self.outputs4)

        return tf.keras.Model(inputs, [self.outputs4, self.outputs3, self.outputs2,self.outputs1], name='tcn_encoder')


    def lstm_encoder(self):
        inputs = layers.Input((self.time_sequence,35))
        lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        lstm_2 = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        lstm_3 = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        dense_stack = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                            #activity_regularizer=tf.keras.regularizers.l2(0.01)
                                            )
        whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
        whole_seq_output, final_memory_state, final_carry_state = lstm_2(whole_seq_output)
        whole_seq_output, final_memory_state, final_carry_state = lstm_3(whole_seq_output)
        whole_seq_output = dense_stack(whole_seq_output)

        return tf.keras.Model(inputs, whole_seq_output, name="lstm_encoder")

    def self_att_layer(self):
        """
        implement self-attention layer
        """
        #inputs = layers.Input((self.time_sequence,self.latent_dim))
        query_input = layers.Input((1,self.latent_dim))
        value_input = layers.Input((self.time_sequence-1,self.latent_dim))
        #value_input = layers.Input((self.time_sequence, self.latent_dim))
        query_embedding = tf.keras.layers.Dense(
                    self.latent_dim,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )
        value_embedding = tf.keras.layers.Dense(
                    self.latent_dim,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )

        #query_seq_encoding = tf.math.l2_normalize(query_embedding(query_input))
        query_seq_encoding = query_embedding(query_input)
        #value_seq_encoding = tf.math.l2_normalize(value_embedding(value_input))
        value_seq_encoding = value_embedding(value_input)
        qv_attention = tf.matmul(value_seq_encoding,query_seq_encoding,transpose_b=True)
        attention = tf.nn.softmax(qv_attention,1)

        self.att = attention
        #attention_broad = tf.broadcast_to(attention, value_seq_encoding.shape)
        #output_val_local = tf.math.l2_normalize(tf.multiply(attention,value_seq_encoding))
        output_val_local = tf.multiply(attention, value_seq_encoding)
        self.output_val = output_val_local

        self.query_seq = query_seq_encoding

        local_info = tf.reduce_sum(output_val_local,1)
        self.local_info = local_info
        #global_encoding = tf.math.l2_normalize(tf.math.add(local_info,tf.squeeze(query_seq_encoding,1)))
        global_encoding = tf.math.add(local_info, tf.squeeze(query_seq_encoding, 1))
        #global_encoding = self.local_info
        attention_squeez = tf.squeeze(attention,2)

        return tf.keras.Model([query_input,value_input],[value_seq_encoding,global_encoding,attention_squeez],name="self_att_layer")


    def self_att_multi_layer(self):
        """
        implement self-attention multi-layer
        """
        #inputs = layers.Input((self.time_sequence,self.latent_dim))
        query_input = layers.Input((1,self.latent_dim))
        value_input = layers.Input((self.time_sequence-1,self.latent_dim))
        value_input_sec = layers.Input((self.time_sequence-1,self.latent_dim))
        value_input_3 = layers.Input((self.time_sequence-1,self.latent_dim))
        value_input_4 = layers.Input((self.time_sequence - 1, self.latent_dim))
        query_embedding = tf.keras.layers.Dense(
                    self.latent_dim,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )
        value_embedding = tf.keras.layers.Dense(
                    self.latent_dim,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )
        """
        value_embedding2 = tf.keras.layers.Dense(
            self.latent_dim,
            # use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )
        value_embedding3 = tf.keras.layers.Dense(
            self.latent_dim,
            # use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )
        value_embedding4 = tf.keras.layers.Dense(
            self.latent_dim,
             #use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )
        """
        #query_seq_encoding = tf.math.l2_normalize(query_embedding(query_input))
        query_seq_encoding = query_embedding(query_input)
        #value_seq_encoding = tf.math.l2_normalize(value_embedding(value_input))
        value_seq_encoding = value_embedding(value_input)
        value_seq_encoding2 = value_embedding(value_input_sec)
        value_seq_encoding3 = value_embedding(value_input_3)
        value_seq_encoding4 = value_embedding(value_input_4)
        qv_attention = tf.matmul(value_seq_encoding,query_seq_encoding,transpose_b=True)
        qv_attention2 = tf.matmul(value_seq_encoding2, query_seq_encoding, transpose_b=True)
        qv_attention3 = tf.matmul(value_seq_encoding3, query_seq_encoding, transpose_b=True)
        qv_attention4 = tf.matmul(value_seq_encoding4, query_seq_encoding, transpose_b=True)
        self.check_av = qv_attention2

        attention = tf.nn.softmax(qv_attention,1)
        attention2 = tf.nn.softmax(qv_attention2,1)
        attention3 = tf.nn.softmax(qv_attention3, 1)
        attention4 = tf.nn.softmax(qv_attention4, 1)
        #attention2 = tf.nn.softmax(tf.squeeze(tf.nn.softmax(qv_attention2, 1),2)*tf.squeeze(attention,2))

        self.att = attention
        self.att2 = attention2
        self.att3 = attention3
        self.att4 = attention4
        #attention_broad = tf.broadcast_to(attention, value_seq_encoding.shape)
        #output_val_local = tf.math.l2_normalize(tf.multiply(attention,value_seq_encoding))
        output_val_local = tf.multiply(attention, value_seq_encoding)
        output_val_local2 = tf.multiply(attention2, value_seq_encoding2)
        output_val_local3 = tf.multiply(attention3, value_seq_encoding3)
        output_val_local4 = tf.multiply(attention4, value_seq_encoding4)
        self.output_val = output_val_local

        self.query_seq = query_seq_encoding

        local_info = tf.reduce_sum(output_val_local,1)
        self.local_info = local_info
        local_info2 = tf.reduce_sum(output_val_local2, 1)
        self.local_info2 = local_info2
        local_info3 = tf.reduce_sum(output_val_local3, 1)
        self.local_info3 = local_info3
        local_info4 = tf.reduce_sum(output_val_local4, 1)
        self.local_info4 = local_info4
        local_all = tf.math.add(local_info,local_info2)
        local_all = tf.math.add(local_all,local_info3)
        local_all = tf.math.add(local_all,local_info4)
        #local_all = tf.math.add(local_info, local_info3)
        #global_encoding = tf.math.l2_normalize(tf.math.add(local_info,tf.squeeze(query_seq_encoding,1)))
        #global_encoding = tf.math.add((local_info, local_info2, 1))
        global_encoding = tf.math.add(local_all, tf.squeeze(query_seq_encoding, 1))
        self.query_seq = query_seq_encoding
        attention_squeez = tf.squeeze(attention,2)
        attention_squeez2 = tf.squeeze(attention2,2)
        attention_squeez3 = tf.squeeze(attention3,2)
        attention_squeez4 = tf.squeeze(attention4,2)

        return tf.keras.Model([query_input,value_input,value_input_sec,value_input_3,value_input_4],
                              [value_seq_encoding,global_encoding,
                               attention_squeez,attention_squeez2,attention_squeez3,attention_squeez4],name="self_att_layer")

    def self_att_multi_layer_prospect(self):
        """
        implement self-attention multi-layer
        """
        #inputs = layers.Input((self.time_sequence,self.latent_dim))
        query_input = layers.Input((self.time_sequence,self.latent_dim))
        value_input = layers.Input((self.time_sequence-1,self.latent_dim))
        value_input_sec = layers.Input((self.time_sequence-1,self.latent_dim))
        value_input_3 = layers.Input((self.time_sequence-1,self.latent_dim))
        value_input_4 = layers.Input((self.time_sequence - 1, self.latent_dim))
        query_embedding = tf.keras.layers.Dense(
                    self.latent_dim,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )
        value_embedding = tf.keras.layers.Dense(
                    self.latent_dim,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )

        value_embedding2 = tf.keras.layers.Dense(
            self.latent_dim,
            # use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )
        value_embedding3 = tf.keras.layers.Dense(
            self.latent_dim,
            # use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )
        value_embedding4 = tf.keras.layers.Dense(
            self.latent_dim,
             use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )

        #query_seq_encoding = tf.math.l2_normalize(query_embedding(query_input))
        query_seq_encoding = query_embedding(query_input)
        #value_seq_encoding = tf.math.l2_normalize(value_embedding(value_input))
        value_seq_encoding = value_embedding(value_input)
        value_seq_encoding2 = value_embedding2(value_input_sec)
        value_seq_encoding3 = value_embedding3(value_input_3)
        value_seq_encoding4 = value_embedding4(value_input_4)
        qv_attention = tf.matmul(value_seq_encoding,query_seq_encoding,transpose_b=True)
        qv_attention2 = tf.matmul(value_seq_encoding2, query_seq_encoding, transpose_b=True)
        qv_attention3 = tf.matmul(value_seq_encoding3, query_seq_encoding, transpose_b=True)
        qv_attention4 = tf.matmul(value_seq_encoding4, query_seq_encoding, transpose_b=True)
        self.check_av = qv_attention2

        attention = tf.nn.softmax(qv_attention,1)
        attention2 = tf.nn.softmax(qv_attention2,1)
        attention3 = tf.nn.softmax(qv_attention3, 1)
        attention4 = tf.nn.softmax(qv_attention4, 1)
        #attention2 = tf.nn.softmax(tf.squeeze(tf.nn.softmax(qv_attention2, 1),2)*tf.squeeze(attention,2))

        self.att = attention
        self.att2 = attention2
        self.att3 = attention3
        self.att4 = attention4
        #attention_broad = tf.broadcast_to(attention, value_seq_encoding.shape)
        #output_val_local = tf.math.l2_normalize(tf.multiply(attention,value_seq_encoding))
        output_val_local = tf.multiply(attention, value_seq_encoding)
        output_val_local2 = tf.multiply(attention2, value_seq_encoding2)
        output_val_local3 = tf.multiply(attention3, value_seq_encoding3)
        output_val_local4 = tf.multiply(attention4, value_seq_encoding4)
        self.output_val = output_val_local

        self.query_seq = query_seq_encoding

        local_info = tf.reduce_sum(output_val_local,1)
        self.local_info = local_info
        local_info2 = tf.reduce_sum(output_val_local2, 1)
        self.local_info2 = local_info2
        local_info3 = tf.reduce_sum(output_val_local3, 1)
        self.local_info3 = local_info3
        local_info4 = tf.reduce_sum(output_val_local4, 1)
        self.local_info4 = local_info4
        local_all = tf.math.add(local_info,local_info2)
        local_all = tf.math.add(local_all,local_info3)
        local_all = tf.math.add(local_all,local_info4)
        #local_all = tf.math.add(local_info, local_info4)
        #global_encoding = tf.math.l2_normalize(tf.math.add(local_info,tf.squeeze(query_seq_encoding,1)))
        #global_encoding = tf.math.add((local_info, local_info2, 1))
        global_encoding = tf.math.add(local_all, tf.squeeze(query_seq_encoding, 1))
        self.query_seq = query_seq_encoding
        attention_squeez = tf.squeeze(attention,2)
        attention_squeez2 = tf.squeeze(attention2,2)
        attention_squeez3 = tf.squeeze(attention3,2)
        attention_squeez4 = tf.squeeze(attention4,2)

        return tf.keras.Model([query_input,value_input,value_input_sec,value_input_3,value_input_4],
                              [value_seq_encoding,global_encoding,
                               attention_squeez,attention_squeez2,attention_squeez3,attention_squeez4],name="self_att_layer")


    def lstm_split(self):
        inputs = layers.Input((self.time_sequence,self.latent_dim))
        output1 = inputs[:,-1,:]
        output1 = tf.expand_dims(output1,1)
        output2 = inputs[:,0:-1,:]
        #output2 = inputs

        return tf.keras.Model(inputs,[output1,output2],name='lstm_split')

    def lstm_split_multi(self):
        inputs = layers.Input((self.time_sequence,self.latent_dim))
        inputs2 = layers.Input((self.time_sequence,self.latent_dim))
        inputs3 = layers.Input((self.time_sequence, self.latent_dim))
        inputs4 = layers.Input((self.time_sequence, self.latent_dim))
        output1 = inputs[:,-1,:]
        output_query1 = tf.expand_dims(output1,1)

        output2 = inputs[:,0:-1,:]
        output3 = inputs2[:,0:-1,:]
        output4 = inputs3[:,0:-1,:]
        output5 = inputs4[:, 0:-1, :]
        #output2 = inputs

        return tf.keras.Model([inputs,inputs2,inputs3,inputs4],
                              [output_query1,output2,output3,output4,output5],name='lstm_split')

    def lstm_merge(self):
        input1 = layers.Input((self.time_sequence-1,self.latent_dim))
        #input1 = layers.Input((self.time_sequence, self.latent_dim))
        input2 = layers.Input((self.latent_dim))
        input3 = layers.Input((self.time_sequence-1))
        input4 = layers.Input((self.time_sequence-1))
        input5 = layers.Input((self.time_sequence-1))
        input6 = layers.Input((self.time_sequence-1))
        #input3 = layers.Input((self.time_sequence))

        output = input2

        return tf.keras.Model([input1,input2,input3,input4,input5,input6],output,name='lstm_merge')

    def lstm_pooling(self):
        inputs = layers.Input((self.time_sequence,self.latent_dim))
        inputs1 = layers.Input((self.time_sequence,self.latent_dim))
        inputs2 = layers.Input((self.time_sequence,self.latent_dim))
        inputs3 = layers.Input((self.time_sequence,self.latent_dim))
        output = inputs[:,-1,:]

        return tf.keras.Model([inputs,inputs1,inputs2,inputs3],output,name="lstm_pooling")

    def lstm_ave_pooling(self):
        inputs = layers.Input((self.time_sequence,self.latent_dim))
        output_ = layers.AveragePooling1D(pool_size=3, strides=2, padding='valid')
        output = output_(inputs)

        return tf.keras.Model(inputs, output, name="lstm_ave_pooling")

    def lstm_conv_pooling(self):
        inputs = layers.Input((self.time_sequence, self.latent_dim))
        conv1d = layers.Conv1D(34,3,activation='relu')
        output = conv1d(inputs)

        return tf.keras.Model(inputs, output, name="conv1d_pooling")

    def scheduler(self,epoch,lr):
        if epoch < 25:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    def project_logit(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                layers.Dense(
                    1,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='sigmoid'
                )
            ],
            name="predictor",
        )
        return model


    def pre_train_supervised(self):
        self.lstm = self.lstm_encoder()
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                self.supervised_sample = np.zeros((x_batch_train.shape))
                self.check_y = y_batch_train
                for i in range(y_batch_train.shape[0]):
                    if y_batch_train[i] == 0:
                        index_neighbor = \
                            np.floor(
                                np.random.uniform(0, len(self.memory_bank_control), 1)).astype(
                                int)
                        self.supervised_sample[i,:,:] = self.memory_bank_control[index_neighbor,:,:]
                    if y_batch_train[i] == 1:
                        index_neighbor = \
                            np.floor(
                                np.random.uniform(0, len(self.memory_bank_cohort), 1)).astype(
                                int)
                        self.supervised_sample[i, :, :] = self.memory_bank_cohort[index_neighbor, :, :]

                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(x_batch_train)[:, self.time_sequence - 1, :], self.lstm(self.supervised_sample)[:,
                                                                                     self.time_sequence - 1, :]
                    loss, loss_prob = self.info_nce_loss(z1, z2)

                gradients = tape.gradient(loss, self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss_prob)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))
                    self.loss_track.append(loss)
                    self.loss_prob_track.append(loss_prob)

    def pre_train_infomax(self):
        #self.lstm = self.lstm_encoder()
        self.lstm = self.tcn_encoder_second_last_level()
        """
        self.lstm_sp = self.lstm_split()
        self.self_att = self.self_att_layer()
        self.global_extractor = self.lstm_merge()
        # self.lstm = self.tcn_encoder()

        self.projector = self.project_logit()
        # self.lstm_pool = self.lstm_pooling()
        # self.model = tf.keras.Sequential([self.lstm,self.lstm_pool,self.projector])
        # self.model = tf.keras.Sequential([self.lstm, self.lstm_sp,self.self_att, self.global_extractor,self.projector])

        inputs = layers.Input((self.time_sequence, 35))
        lstm = self.lstm(inputs)
        lstm_sp = self.lstm_sp(lstm)
        self_att = self.self_att(lstm_sp)

        self.att_lstm_model = tf.keras.Model(inputs,self_att,name="att_lstm_model")
        """

        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train, sofa_batch_train) in enumerate(self.train_dataset):
                #self.batch_cluster(sofa_batch_train)
                #noise1 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                #noise2 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                data_aug1 = x_batch_train# + noise1
                data_aug2 = x_batch_train# + noise2
                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(data_aug1)[:, -1, :], self.lstm(data_aug2)[:, 0:-1, :]
                    #z1, z2 = self.att_lstm_model(data_aug1)[1], self.att_lstm_model(data_aug2)[0]
                    loss, loss_prob = self.info_nce_loss_infomax(z1, z2)

                gradients = tape.gradient(loss, self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_variables))
                self.check_loss = loss
                self.check_loss_prob = loss_prob

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)
                    self.loss_prob_track.append(loss_prob)



    def batch_cluster(self,sofa_batch_train):
        kmeans = KMeans(n_clusters=4,random_state=0).fit(sofa_batch_train)
        self.batch_cluster_label = kmeans.labels_

    def pre_train_cluster_time(self):
        self.lstm = self.lstm_encoder()
        self.lstm_ave = self.lstm_ave_pooling()
        self.lstm_pre = tf.keras.Sequential([self.lstm, self.lstm_ave])
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train,sofa_batch_train) in enumerate(self.train_dataset):
                self.batch_cluster(sofa_batch_train)
                noise1 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                noise2 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                data_aug1 = x_batch_train + noise1
                data_aug2 = x_batch_train + noise2
                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(data_aug1)[:, -1, :], self.lstm(data_aug2)[:, -2, :]
                    loss, loss_prob = self.info_nce_loss_cluster(z1, z2,self.batch_cluster_label)

                gradients = tape.gradient(loss, self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_variables))
                self.check_loss = loss
                self.check_loss_prob = loss_prob

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)
                    self.loss_prob_track.append(loss_prob)


    def pre_train_cluster_self_time(self):
        self.lstm = self.lstm_encoder()
        self.lstm_ave = self.lstm_ave_pooling()
        self.lstm_pre = tf.keras.Sequential([self.lstm, self.lstm_ave])
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train, sofa_batch_train) in enumerate(self.train_dataset):
                self.batch_cluster(sofa_batch_train)
                noise1 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                noise2 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                data_aug1 = x_batch_train + noise1
                data_aug2 = x_batch_train + noise2
                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(data_aug1)[:, -1, :], self.lstm(data_aug2)[:, -2, :]
                    loss, loss_prob = self.info_nce_loss_cluster(z1, z2, self.batch_cluster_label)
                    loss_self, loss_prob_self = self.info_nce_loss(z1, z2)
                    total_loss = 0.8*loss_self+0.2*loss

                gradients = tape.gradient(total_loss, self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_variables))
                self.check_loss = loss
                self.check_loss_prob = loss_prob

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(total_loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(total_loss)
                    self.loss_prob_track.append(loss_prob)

    def pre_train_self_time_infomax(self):
        #self.lstm = self.lstm_encoder()
        self.lstm = self.tcn_encoder()
        # self.lstm_ave = self.lstm_ave_pooling()
        self.lstm_ave = self.lstm_conv_pooling()
        self.lstm_pre = tf.keras.Sequential([self.lstm, self.lstm_ave])
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train, sofa_batch_train) in enumerate(self.train_dataset):
                #noise1 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                #noise2 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                data_aug1 = x_batch_train #+ noise1
                data_aug2 = x_batch_train #+ noise2
                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(data_aug1), self.lstm(data_aug2)
                    loss, loss_prob = self.info_nce_loss_infomax(z1, z2)

                gradients = tape.gradient(loss, self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)
                    self.loss_prob_track.append(loss_prob)


    def pre_train_self_time(self):
        self.lstm = self.lstm_encoder()
        #self.lstm_ave = self.lstm_ave_pooling()
        self.lstm_ave = self.lstm_conv_pooling()
        self.lstm_pre = tf.keras.Sequential([self.lstm, self.lstm_ave])
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" %(epoch,))

            for step, (x_batch_train, y_batch_train,sofa_batch_train) in enumerate(self.train_dataset):
                #noise1 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                #noise2 = np.random.normal(self.gaussian_mu, self.gaussian_sigma, x_batch_train.shape)
                data_aug1 = x_batch_train# + noise1
                data_aug2 = x_batch_train# + noise2
                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(data_aug1)[:,-1,:], self.lstm(data_aug2)[:,-2,:]
                    #z3 = self.lstm(data_aug2)[:,-3,:]
                    loss,loss_prob = self.info_nce_loss(z1,z2)
                    #loss_,loss_prob_ = self.info_nce_loss(z1)

                    #loss_total = loss+loss_

                gradients = tape.gradient(loss,self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                    % (step, float(loss)))
                    print("seen so far: %s samples" % ((step+1)*self.batch_size))

                    self.loss_track.append(loss)
                    self.loss_prob_track.append(loss_prob)


    def pre_train_gaussian_noise(self):
        self.lstm = self.lstm_encoder()
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" %(epoch,))

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                noise1 = np.random.normal(self.gaussian_mu,self.gaussian_sigma,x_batch_train.shape)
                noise2 = np.random.normal(self.gaussian_mu,self.gaussian_sigma,x_batch_train.shape)
                data_aug1 = x_batch_train + noise1
                data_aug2 = x_batch_train + noise2
                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(data_aug1)[:,self.time_sequence-1,:], self.lstm(data_aug2)[:,self.time_sequence-1,:]
                    loss,loss_prob = self.info_nce_loss(z1,z2)

                gradients = tape.gradient(loss,self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_weights))

                self.loss_tracker.update_state(loss)

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                    % (step, float(self.loss_tracker.result())))
                    print("seen so far: %s samples" % ((step+1)*self.batch_size))


    def build_model(self):
        """
        buuild non att model
        """
        inputs = layers.Input((self.time_sequence, 35))
        inputs_mask = layers.Masking(mask_value=0, input_shape=(self.time_sequence, 35))(inputs)
        #self.lstm = self.lstm_encoder()
        self.lstm = self.tcn_encoder_second_last_level()
        self.lstm_pool = self.lstm_pooling()
        self.projector = self.project_logit()

        #mask = inputs_mask(inputs)
        lstm = self.lstm(inputs)
        lstm_pool = self.lstm_pool(lstm)
        projector = self.projector(lstm_pool)

        #self.model_mask = tf.keras.Model(inputs,mask,name="mask")

        self.model = tf.keras.Model(inputs, projector, name="lstm_model")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC()])
        #self.train_data = np.zeros((self.train_logit.shape[0], self.time_sequence, 35))

    def build_model_pre(self):
        inputs = layers.Input((self.time_sequence, 35))
        #self.lstm = self.lstm_encoder()
        #self.lstm = self.tcn_encoder_second_last_level()
        self.lstm_pool = self.lstm_pooling()
        self.projector = self.project_logit()

        lstm = self.lstm(inputs)
        lstm_pool = self.lstm_pool(lstm)
        projector = self.projector(lstm_pool)

        self.model = tf.keras.Model(inputs, projector, name="lstm_model")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC()])



    def build_model_att(self):
        """
        building temporal attention model
        """
        inputs = layers.Input((self.time_sequence,35))
        #self.lstm = self.lstm_encoder()
        self.lstm = self.tcn_encoder_second_last_level()
        """
        attention mechanism
        """
        self.lstm_sp = self.lstm_split_multi()
        self.self_att = self.self_att_multi_layer()
        self.global_extractor = self.lstm_merge()
        #self.lstm = self.tcn_encoder()

        self.projector = self.project_logit()
        #self.lstm_pool = self.lstm_pooling()
        #self.model = tf.keras.Sequential([self.lstm,self.lstm_pool,self.projector])
        #self.model = tf.keras.Sequential([self.lstm, self.lstm_sp,self.self_att, self.global_extractor,self.projector])


        lstm = self.lstm(inputs)
        lstm_sp = self.lstm_sp(lstm)
        self_att = self.self_att(lstm_sp)
        global_extractor = self.global_extractor(self_att)
        projector = self.projector(global_extractor)

        self.model_test = tf.keras.Model(inputs,self_att,name='att_test')
        self.model = tf.keras.Model(inputs, projector, name="att_lstm_model")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC()])

    def build_model_pre_att(self):
        #self.lstm.trainable = False
        inputs = layers.Input((self.time_sequence, 35))
        self.lstm_sp = self.lstm_split()
        self.self_att = self.self_att_layer()
        self.global_extractor = self.lstm_merge()
        # self.lstm = self.tcn_encoder()

        self.projector = self.project_logit()
        # self.lstm_pool = self.lstm_pooling()
        # self.model = tf.keras.Sequential([self.lstm,self.lstm_pool,self.projector])
        # self.model = tf.keras.Sequential([self.lstm, self.lstm_sp,self.self_att, self.global_extractor,self.projector])

        lstm = self.lstm(inputs)
        lstm_sp = self.lstm_sp(lstm)
        self_att = self.self_att(lstm_sp)
        global_extractor = self.global_extractor(self_att)
        projector = self.projector(global_extractor)

        #self.model = tf.keras.Sequential([self.lstm, self.lstm_pool,self.projector])
        self.model = tf.keras.Model(inputs, projector, name="att_lstm_model_infomax_pretrain")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC()])

    def train_classifier(self):
        #callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        history = self.model.fit(self.train_data, self.train_logit, batch_size=self.batch_size,epochs=self.epoch, validation_data=(self.val_data,self.val_logit))
        plt.plot(history.history["loss"][1:])
        plt.grid()
        plt.title("training loss(without pretrain)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.cla()

    def drat(self):
        axs[0].plot(seq.train_data[6,:,0])
        axs[1].plot(seq.train_data[6,:,1])
        axs[2].plot(seq.train_data[6,:,2])
        axs[3].plot(seq.train_data[6,:,3])
        axs[4].plot(seq.train_data[6,:,4])
        axs[5].plot(seq.train_data[6,:,5])
        axs[6].plot(seq.train_data[6,:,6])
        axs[7].plot(seq.train_data[6,:,7])
        axs[8].plot(seq.train_data[6,:,8])
        axs[9].plot(seq.train_data[6,:,9])
        axs[10].plot(seq.train_data[6,:,10])
        axs[11].plot(seq.train_data[6,:,11])



    def collect_stat(self):
        cohort = np.where(self.train_logit==1)[0]
        self.total_valid_cohort = []
        self.increase_relation_hr = []
        self.decrease_relation_hr = []
        self.increase_relation_po_in = []
        self.decrease_relation_po_in = []
        self.increase_relation_po_de = []
        self.decrease_relation_po_de = []
        self.ave_value_hr = 0
        self.ave_value_po = 0
        self.index_hr = 0
        self.index_po = 0


        for i in cohort:
            sample = np.expand_dims(self.train_data[i,:,:],0)
            att = self.model_test(sample)[5]
            att_location = np.where(att==np.max(att))[1][0]
            if att_location+1 < 3:
                continue
            hr = self.train_data[i,att_location+1-3:att_location+1,0]
            length = len(np.where(hr==0)[0])
            if length > 1:
                continue
            min_loc = np.where(hr==np.min(hr))[0]
            if len(min_loc)>1:
                min_loc=min_loc[0]
            max_loc = np.where(hr==np.max(hr))[0]
            if len(max_loc)>1:
                max_loc=max_loc[0]

            if min_loc<max_loc:
                self.increase_relation_hr.append(i)
            else:
                self.decrease_relation_hr.append(i)
            self.total_valid_cohort.append(i)
            ave = (min_loc+max_loc)/2
            self.ave_value_hr+=ave
            self.index_hr+=1

        for i in self.increase_relation_hr:
            sample = np.expand_dims(self.train_data[i, :, :], 0)
            att = self.model_test(sample)[5]
            att_location = np.where(att == np.max(att))[1][0]
            if att_location + 1 < 3:
                continue
            hr = self.train_data[i, att_location + 1 - 3:att_location + 1, 1]
            length = len(np.where(hr == 0)[0])
            if length > 1:
                continue
            min_loc = np.where(hr == np.min(hr))[0]
            if len(min_loc) > 1:
                min_loc = min_loc[0]
            max_loc = np.where(hr == np.max(hr))[0]
            if len(max_loc) > 1:
                max_loc = max_loc[0]

            if min_loc < max_loc:
                self.increase_relation_po_in.append(i)
            else:
                self.decrease_relation_po_in.append(i)

            ave = (min_loc+max_loc)/2
            self.ave_value_po +=ave
            self.index_po+=1

        for i in self.decrease_relation_hr:
            sample = np.expand_dims(self.train_data[i, :, :], 0)
            att = self.model_test(sample)[5]
            att_location = np.where(att == np.max(att))[1][0]
            if att_location + 1 < 3:
                continue
            hr = self.train_data[i, att_location + 1 - 3:att_location + 1, 1]
            length = len(np.where(hr == 0)[0])
            if length > 1:
                continue
            min_loc = np.where(hr == np.min(hr))[0]
            if len(min_loc) > 1:
                min_loc = min_loc[0]
            max_loc = np.where(hr == np.max(hr))[0]
            if len(max_loc) > 1:
                max_loc = max_loc[0]

            if min_loc < max_loc:
                self.increase_relation_po_de.append(i)
            else:
                self.decrease_relation_po_de.append(i)

            ave = (min_loc+max_loc)/2
            self.ave_value_po +=ave
            self.index_po+=1

    def collect_stat_relation_hr(self):
        cohort = np.where(self.train_logit==1)[0]
        self.total_relations = []

        for i in cohort:
            sample = np.expand_dims(self.train_data[i,:,:],0)
            self.att = np.array(self.model_test(sample)[5])[0]
            if len(np.where(self.att>0.05)[0]) == 0:
                continue
            att_location_first = np.where(self.att>0.05)[0][0]
            att_location_sec = np.where(self.att > 0.05)[0][-1]
            relation = [0,0]
            if att_location_first+1<3:
                hr = self.train_data_origin[i,att_location_first,0]
                if hr < self.heart_rate_range[0]:
                    relation[0] = 0
                elif hr > self.heart_rate_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_first+1-3:att_location_first+1, 0])
                if hr < self.heart_rate_range[0]:
                    relation[0] = 0
                elif hr > self.heart_rate_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1

            if att_location_sec+1<3:
                hr = self.train_data_origin[i,att_location_sec,0]
                if hr < self.heart_rate_range[0]:
                    relation[1] = 0
                elif hr > self.heart_rate_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_sec+1-3:att_location_sec+1, 0])
                if hr < self.heart_rate_range[0]:
                    relation[1] = 0
                elif hr > self.heart_rate_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1

            self.total_relations.append(relation)

    def collect_stat_relation_resp(self):
        cohort = np.where(self.train_logit==1)[0]
        self.total_relations = []

        for i in cohort:
            sample = np.expand_dims(self.train_data[i,:,:],0)
            self.att = np.array(self.model_test(sample)[5])[0]
            if len(np.where(self.att>0.05)[0]) == 0:
                continue
            att_location_first = np.where(self.att>0.05)[0][0]
            att_location_sec = np.where(self.att > 0.05)[0][-1]
            relation = [0,0]
            if att_location_first+1<3:
                hr = self.train_data_origin[i,att_location_first,6]
                if hr < self.resp_rate_range[0]:
                    relation[0] = 0
                elif hr > self.resp_rate_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_first+1-3:att_location_first+1, 0])
                if hr < self.resp_rate_range[0]:
                    relation[0] = 0
                elif hr > self.resp_rate_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1

            if att_location_sec+1<3:
                hr = self.train_data_origin[i,att_location_sec,6]
                if hr < self.resp_rate_range[0]:
                    relation[1] = 0
                elif hr > self.resp_rate_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_sec+1-3:att_location_sec+1, 0])
                if hr < self.resp_rate_range[0]:
                    relation[1] = 0
                elif hr > self.resp_rate_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1

            self.total_relations.append(relation)

    def collect_stat_relation_map(self):
        cohort = np.where(self.train_logit==1)[0]
        self.total_relations = []

        for i in cohort:
            sample = np.expand_dims(self.train_data[i,:,:],0)
            self.att = np.array(self.model_test(sample)[5])[0]
            if len(np.where(self.att>0.05)[0]) == 0:
                continue
            att_location_first = np.where(self.att>0.05)[0][0]
            att_location_sec = np.where(self.att > 0.05)[0][-1]
            relation = [0,0]
            if att_location_first+1<3:
                hr = self.train_data_origin[i,att_location_first,4]
                if hr < self.map_range[0]:
                    relation[0] = 0
                elif hr > self.map_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_first+1-3:att_location_first+1, 0])
                if hr < self.map_range[0]:
                    relation[0] = 0
                elif hr > self.map_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1

            if att_location_sec+1<3:
                hr = self.train_data_origin[i,att_location_sec,4]
                if hr < self.map_range[0]:
                    relation[1] = 0
                elif hr > self.map_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_sec+1-3:att_location_sec+1, 0])
                if hr < self.map_range[0]:
                    relation[1] = 0
                elif hr > self.map_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1

            self.total_relations.append(relation)

    def collect_stat_relation_sbp(self):
        cohort = np.where(self.train_logit==1)[0]
        self.total_relations = []

        for i in cohort:
            sample = np.expand_dims(self.train_data[i,:,:],0)
            self.att = np.array(self.model_test(sample)[5])[0]
            if len(np.where(self.att>0.05)[0]) == 0:
                continue
            att_location_first = np.where(self.att>0.05)[0][0]
            att_location_sec = np.where(self.att > 0.05)[0][-1]
            relation = [0,0]
            if att_location_first+1<3:
                hr = self.train_data_origin[i,att_location_first,3]
                if hr < self.sbp_range[0]:
                    relation[0] = 0
                elif hr > self.sbp_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_first+1-3:att_location_first+1, 0])
                if hr < self.sbp_range[0]:
                    relation[0] = 0
                elif hr > self.sbp_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1

            if att_location_sec+1<3:
                hr = self.train_data_origin[i,att_location_sec,3]
                if hr < self.sbp_range[0]:
                    relation[1] = 0
                elif hr > self.sbp_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_sec+1-3:att_location_sec+1, 0])
                if hr < self.sbp_range[0]:
                    relation[1] = 0
                elif hr > self.sbp_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1

            self.total_relations.append(relation)

    def collect_stat_relation_dbp(self):
        cohort = np.where(self.train_logit==1)[0]
        self.total_relations = []

        for i in cohort:
            sample = np.expand_dims(self.train_data[i,:,:],0)
            self.att = np.array(self.model_test(sample)[5])[0]
            if len(np.where(self.att>0.05)[0]) == 0:
                continue
            att_location_first = np.where(self.att>0.05)[0][0]
            att_location_sec = np.where(self.att > 0.05)[0][-1]
            relation = [0,0]
            if att_location_first+1<3:
                hr = self.train_data_origin[i,att_location_first,5]
                if hr < self.dbp_range[0]:
                    relation[0] = 0
                elif hr > self.dbp_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_first+1-3:att_location_first+1, 0])
                if hr < self.dbp_range[0]:
                    relation[0] = 0
                elif hr > self.dbp_range[1]:
                    relation[0] = 2
                else:
                    relation[0] = 1

            if att_location_sec+1<3:
                hr = self.train_data_origin[i,att_location_sec,5]
                if hr < self.dbp_range[0]:
                    relation[1] = 0
                elif hr > self.dbp_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1
            else:
                hr = np.median(self.train_data_origin[i, att_location_sec+1-3:att_location_sec+1, 0])
                if hr < self.dbp_range[0]:
                    relation[1] = 0
                elif hr > self.dbp_range[1]:
                    relation[1] = 2
                else:
                    relation[1] = 1

            self.total_relations.append(relation)

    def compute_relation_numbers(self):
        self.total_relation_num = np.zeros(9)
        for i in range(len(self.relations)):
            num = 0
            for j in self.total_relations:
                if j == self.relations[i]:
                    num+=1
            self.total_relation_num[i] = num







