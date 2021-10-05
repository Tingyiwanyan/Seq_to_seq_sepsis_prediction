from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf

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
        self.pre_train_epoch = 5
        self.latent_dim = 100
        self.tau = 1
        self.time_sequence = self.read_d.time_sequence

        self.steps = self.epoch * (self.length_train//self.batch_size)
        self.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.03, decay_steps=self.steps)

    def aquire_data(self, starting_index, data_set,length):
        data = np.zeros((length,self.time_sequence,35))
        logit_dp = np.zeros((length,1))
        for i in range(length):
            name = data_set[starting_index+i]
            self.read_d.read_table(name)
            one_data = self.read_d.one_data_tensor
            #one_data = np.mean(one_data,0)
            data[i,:,:] = one_data
            logit_dp[i,0] = self.read_d.one_data_logit

        logit = logit_dp[:,0]
        return (data, logit)

    def create_memory_bank(self):
        #self.train_data, self.train_logit = self.aquire_data(0, self.train_data, self.length_train)
        #self.test_data, self.test_logit = self.aquire_data(0, self.test_data, self.length_test)
        #self.val_data, self.val_logit = self.aquire_data(0, self.validate_data, self.length_val)

        file_path = '/home/tingyi/physionet_data/'
        with open(file_path+'train.npy','rb') as f:
            self.train_data = np.load(f)
        with open(file_path+'train_logit.npy','rb') as f:
            self.train_logit = np.load(f)
        with open(file_path+'test.npy','rb') as f:
            self.test_data = np.load(f)
        with open(file_path+'test_logit.npy','rb') as f:
            self.test_logit = np.load(f)
        with open(file_path+'val.npy','rb') as f:
            self.val_data = np.load(f)
        with open(file_path+'val_logit.npy','rb') as f:
            self.val_logit = np.load(f)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_logit))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        cohort_index = np.where(self.train_logit == 1)[0]
        control_index = np.where(self.train_logit == 0)[0]
        self.memory_bank_cohort = self.train_data[cohort_index,:,:]
        self.memory_bank_control = self.train_data[control_index,:,:]


    def compute_positive_pair(self,z,p):
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)

        positive_dot_prod = tf.multiply(z,p)
        positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(positive_dot_prod,1)/self.tau)
        self.positive_dot_prod_check = positive_dot_prod_sum

        return positive_dot_prod_sum

    def compute_negative_paris(self,z,pos,neg,label):
        z = tf.math.l2_normalize(z, axis=1)
        similarity_matrix_pos = tf.matmul(z, tf.transpose(pos))
        similarity_matrix_neg = tf.matmul(z, tf.transpose(neg))
        index_first = np.expand_dims(np.array(range(label.shape[0])), -1)
        index_sec = tf.expand_dims(label,-1)
        index_whole = tf.cast(tf.concat([index_first,index_sec],-1),tf.int32)
        #mask = tf.linalg.diag(tf.zeros(z.shape[0]),padding_value=1)
        #negative_dot_prods = tf.multiply(similarity_matrix,mask)
        negative_dot_prods_sum_pos = tf.expand_dims(tf.reduce_sum(tf.math.exp(similarity_matrix_pos / self.tau),1),-1)
        negative_dot_prods_sum_neg = tf.expand_dims(tf.reduce_sum(tf.math.exp(similarity_matrix_neg / self.tau), 1),-1)

        self.negative_doc_prods_whole = tf.concat([negative_dot_prods_sum_pos,negative_dot_prods_sum_neg],-1)
        self.label_check = label
        negative_dot_prods_mask = tf.gather_nd(self.negative_doc_prods_whole,index_whole)
        #negative_dot_prods_sum = tf.reduce_sum(tf.math.exp(similarity_matrix / self.tau), 1)
        self.neg_dot_prod_check = negative_dot_prods_mask

        self.sim_pos_check = similarity_matrix_pos
        self.sim_neg_check = similarity_matrix_neg
        return negative_dot_prods_mask

    def info_nce_loss(self,z,p,pos,neg,label):
        positive_dot_prod_sum = self.compute_positive_pair(z,p)
        negative_dot_prod_sum = self.compute_negative_paris(z,pos,neg,label)

        denominator = tf.math.add(positive_dot_prod_sum,negative_dot_prod_sum)
        #denominator = negative_dot_prod_sum
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum,denominator))
        loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum,denominator),0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log,0))

        return loss, loss_prob


    def lstm_encoder(self):
        inputs = layers.Input((self.time_sequence,35))
        lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        lstm_2 = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        #lstm_3 = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        dense_stack = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            kernel_regularizer=tf.keras.regularizers.l1(0.01)
                                            #activity_regularizer=tf.keras.regularizers.l2(0.01)
                                            )
        whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
        whole_seq_output, final_memory_state, final_carry_state = lstm_2(whole_seq_output)
        #whole_seq_output, final_memory_state, final_carry_state = lstm_3(whole_seq_output)
        #whole_seq_output = dense_stack(whole_seq_output)

        return tf.keras.Model(inputs, whole_seq_output, name="lstm_encoder")

    def lstm_pooling(self):
        inputs = layers.Input((self.time_sequence,self.latent_dim))
        #output = inputs[:,self.time_sequence-1,:]
        output = tf.reduce_mean(inputs,1)

        return tf.keras.Model(inputs,output,name="lstm_pooling")

    def lstm_train_from_scratch(self):
        inputs = layers.Input((self.time_sequence,35))
        lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        lstm_2 = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        #lstm_3 = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        dense_stack = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            kernel_regularizer=tf.keras.regularizers.l1(0.01)
                                            #activity_regularizer=tf.keras.regularizers.l2(0.01)
                                            )
        whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
        whole_seq_output, final_memory_state, final_carry_state = lstm_2(whole_seq_output)
        #whole_seq_output, final_memory_state, final_carry_state = lstm_3(whole_seq_output)
        #whole_seq_output = dense_stack(whole_seq_output)

        return tf.keras.Model(inputs, whole_seq_output, name="lstm_encoder_from_scratch")

    def lstm_linear_evaluation(self):
        inputs = layers.Input((self.time_sequence, self.latent_dim))
        lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)

        whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)

        return tf.keras.Model(inputs, whole_seq_output, name="lstm_encoder_from_scratch")

   # def gaussain_noise_aug(self):

    def project_logit(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim,)),
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
        self.lstm_pre_train_pool = self.lstm_pooling()
        self.lstm_pre = tf.keras.Sequential([self.lstm, self.lstm_pre_train_pool])
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                #pos_batch = np.zeros((x_batch_train.shape))
                #neg_batch = np.zeros((x_batch_train.shape))
                pos_index = \
                    np.floor(np.random.uniform(0,len(self.memory_bank_cohort),x_batch_train.shape[0])).astype(int)
                neg_index = \
                    np.floor(np.random.uniform(0,len(self.memory_bank_control),x_batch_train.shape[0])).astype(int)
                pos_batch = self.memory_bank_cohort[pos_index,:]+\
                                 np.random.normal(self.gaussian_mu,self.gaussian_sigma,x_batch_train.shape)
                neg_batch = self.memory_bank_control[neg_index,:]+\
                                 np.random.normal(self.gaussian_mu, self.gaussian_sigma,
                                                                         x_batch_train.shape)
                pos = self.lstm_pre(pos_batch)
                neg = self.lstm_pre(neg_batch)
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
                    x_batch_train_gn = x_batch_train+\
                             np.random.normal(self.gaussian_mu,self.gaussian_sigma,x_batch_train.shape)
                    supervised_sample_gn = self.supervised_sample+\
                             np.random.normal(self.gaussian_mu,self.gaussian_sigma,x_batch_train.shape)
                    z1, z2 = self.lstm_pre(x_batch_train_gn), self.lstm_pre(supervised_sample_gn)
                    loss, loss_prob = self.info_nce_loss(z1, z2, pos, neg, y_batch_train)

                gradients = tape.gradient(loss, self.lstm_pre.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                optimizer.apply_gradients(zip(gradients, self.lstm_pre.trainable_variables))
                self.loss_tracker.update_state(loss)

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step,self.loss_tracker.result()))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

    def pre_train_self_time(self):
        self.lstm = self.lstm_encoder()
        self.loss_track = []
        self.loss_prob_track = []
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" %(epoch,))

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    z1, z2 = self.lstm(x_batch_train)[:,self.time_sequence-1,:], self.lstm(x_batch_train)[:,self.time_sequence-2,:]
                    loss,loss_prob = self.info_nce_loss(z1,z2)

                gradients = tape.gradient(loss,self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                    % (step, float(loss_prob)))
                    print("seen so far: %s samples" % ((step+1)*self.batch_size))

                    self.loss_track.append(loss)
                    self.loss_prob_track.append(loss_prob)


    def pre_train_gaussian_noise(self):
        self.lstm = self.lstm_encoder()
        self.lstm_pre_train_pool = self.lstm_pooling()
        self.lstm_pre = tf.keras.Sequential([self.lstm,self.lstm_pre_train_pool])
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
                    z1, z2 = self.lstm_pre(data_aug1), self.lstm_pre(data_aug2)
                    loss,loss_prob = self.info_nce_loss(z1,z2)

                gradients = tape.gradient(loss,self.lstm.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                #optimizer = tf.keras.optimizers.SGD(self.lr_decayed_fn, momentum=0.6)

                optimizer.apply_gradients(zip(gradients, self.lstm.trainable_weights))

                self.loss_tracker.update_state(loss)

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                    % (step, float(self.loss_tracker.result())))
                    print("seen so far: %s samples" % ((step+1)*self.batch_size))
                    #self.loss_track.append(loss)
                    #self.loss_prob_track.append(loss_prob)




    def build_model(self):
        self.lstm_train_scratch = self.lstm_train_from_scratch()
        self.projector = self.project_logit()
        self.lstm_pool = self.lstm_pooling()
        self.model = tf.keras.Sequential([self.lstm_train_scratch,self.lstm_pool,self.projector])
        self.model.compile(optimizer="adam",loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC()])

    def build_model_pre(self):
        #self.lstm.trainable = False
        #self.lstm_linear = self.lstm_linear_evaluation()
        self.projector = self.project_logit()
        #self.lstm_pool = self.lstm_pooling()
        self.model = tf.keras.Sequential([self.lstm_pre,self.projector])
        self.model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC()])

    def train_classifier(self):
        history = self.model.fit(self.train_data, self.train_logit, epochs=self.epoch, validation_data=(self.val_data,self.val_logit))
        """
        plt.plot(history.history["loss"][1:])
        plt.grid()
        plt.title("training loss(without pretrain)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.cla()
        """



