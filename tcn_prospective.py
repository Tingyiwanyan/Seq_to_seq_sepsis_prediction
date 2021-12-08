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
            decay_rate=0.6)

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

    def create_memory_bank_prospect(self):
        # self.train_data, self.train_logit,self.train_sofa,self.train_sofa_score = self.aquire_data(0, self.train_data, self.length_train)
        # self.test_data, self.test_logit,self.test_sofa,self.test_sofa_score = self.aquire_data(0, self.test_data, self.length_test)
        # self.val_data, self.val_logit,self.val_sofa,self.val_sofa_score = self.aquire_data(0, self.validate_data, self.length_val)

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

        # index = np.array(self.read_d.missingness_95)
        # self.train_data = self.train_data[:,:,index]
        # self.val_data = self.val_data[:,:,index]

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_logit))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

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
        inputs = layers.Input((self.time_sequence,34))
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

    def lstm_split(self):
        inputs = layers.Input((self.time_sequence,self.latent_dim))
        output1 = inputs[:,-1,:]
        #output1 = tf.expand_dims(output1,1)
        output2 = inputs[:,0:-1,:]
        #output2 = inputs

        return tf.keras.Model(inputs,[output2,output1],name='lstm_split')

    def lstm_merge(self):
        input1 = layers.Input((self.time_sequence-1,self.latent_dim))
        #input1 = layers.Input((self.time_sequence, self.latent_dim))
        input2 = layers.Input((self.latent_dim))

        output = input2

        return tf.keras.Model([input1,input2],output,name='lstm_merge')


    def scheduler(self,epoch,lr):
        if epoch < 25:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    def project_logit(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.time_sequence,self.latent_dim)),
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


    def build_model(self):
        """
        buuild non att model
        """
        inputs = layers.Input((self.time_sequence, 34))
        #inputs_mask = layers.Masking(mask_value=0, input_shape=(self.time_sequence, 23))(inputs)
        self.lstm = self.lstm_encoder()
        self.lstm_s = self.lstm_split()
        self.lstm_pool = self.lstm_merge()
        #self.projector = self.project_logit()
        self.projector = layers.Dense(
                    1,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='sigmoid'
                )

        #mask = inputs_mask(inputs)
        lstm = self.lstm(inputs)
        lstm_s = self.lstm_s(lstm)
        lstm_pool = self.lstm_pool(lstm_s)
        projector = self.projector(lstm_pool)

        #self.model_mask = tf.keras.Model(inputs,mask,name="mask")

        self.model = tf.keras.Model(inputs, projector, name="lstm_model")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC()])
        self.mask_value = np.zeros((self.train_logit.shape[0],self.time_sequence,35))



    def train_classifier(self):
        #callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        history = self.model.fit(self.mask_value, self.train_logit, batch_size=self.batch_size,epochs=self.epoch, validation_data=(self.val_data,self.val_logit))
        plt.plot(history.history["loss"][1:])
        plt.grid()
        plt.title("training loss(without pretrain)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.cla()

