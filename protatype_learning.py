from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.cluster import KMeans
from tcn import TCN
from tensorflow import keras
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score


import matplotlib.pyplot as plt
import numpy as np

semantic_step_global = 4
unsupervised_cluster_num = 5
latent_dim_global = 100

class projection(keras.layers.Layer):
    def __init__(self, units=semantic_step_global, input_dim=latent_dim_global):
        super(projection, self).__init__()
        #w_init = tf.random_normal_initializer()
        w_init = tf.keras.initializers.Orthogonal()
        self.w = tf.Variable(
            initial_value=w_init(shape=(units,input_dim), dtype="float32"),
            trainable=True,
        )
        #b_init = tf.zeros_initializer()
        #self.b = tf.Variable(
            #initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        #)

    def call(self, inputs):
        return [tf.math.multiply(inputs[0], self.w), inputs[1]]

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
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 20
        self.pre_train_epoch = 20
        self.latent_dim = latent_dim_global
        self.tau = 1
        self.time_sequence = self.read_d.time_sequence
        self.tcn_filter_size = 3
        self.semantic_time_step = semantic_step_global
        self.unsupervised_cluster_num = unsupervised_cluster_num
        self.start_sampling_index = 4
        self.sampling_interval = 4
        self.max_value_projection = np.zeros((self.batch_size,self.semantic_time_step))
        self.basis_input = np.ones((self.unsupervised_cluster_num,self.latent_dim))

        """
        initialize orthogonal projection basis
        """
        self.initializer_basis = tf.keras.initializers.Orthogonal()
        self.projection_basis = tf.Variable(
                self.initializer_basis(shape=(self.semantic_time_step, self.latent_dim)))


        self.steps = self.length_train // self.batch_size
        self.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, decay_steps=self.steps)

        """
        initialize position encoding vectors
        """
        self.position_embedding = np.zeros((self.semantic_time_step,self.latent_dim))
        self.generate_orthogonal = ortho_group.rvs(self.latent_dim)
        for i in range(self.semantic_time_step):
            #self.position_embedding[i,:] = self.position_encoding(i)
            self.position_embedding[i, :] = self.generate_orthogonal[i]

        #self.batch_position_embedding = np.expand_dims(self.position_embedding,0)
        #self.batch_position_embedding = np.broadcast_to(self.batch_position_embedding,[self.batch_size,
                                                                                       #self.semantic_time_step,
                                                                                       #self.latent_dim])

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0003,
            decay_steps=self.steps,
            decay_rate=0.7)

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

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_logit))#,self.train_sofa_score))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        cohort_index = np.where(self.train_logit == 1)[0]
        control_index = np.where(self.train_logit == 0)[0]
        self.memory_bank_cohort = self.train_data[cohort_index,:,:]
        self.memory_bank_control = self.train_data[control_index,:,:]
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
        z = tf.math.l2_normalize(z, axis=1)
        p = tf.math.l2_normalize(p, axis=1)

        positive_dot_prod = tf.multiply(z, p)
        positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(positive_dot_prod, 1) / self.tau)

        return positive_dot_prod_sum

    def compute_negative_paris(self, z, negs):
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

    #def unsupervised_prototype_loss(self):


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

        return tf.keras.Model(inputs,
                              [self.outputs4, self.outputs3, self.outputs2,self.outputs1], name='tcn_encoder')

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

    def tcn_pull(self):
        inputs = layers.Input((self.time_sequence, self.latent_dim))
        inputs2 = layers.Input((self.time_sequence, self.latent_dim))
        inputs3 = layers.Input((self.time_sequence, self.latent_dim))
        inputs4 = layers.Input((self.time_sequence, self.latent_dim))
        output1 = inputs#[:, -1, :]

        # output2 = inputs

        return tf.keras.Model([inputs, inputs2, inputs3, inputs4],
                              output1, name='tcn_pull')

    def discrete_time_period_extract(self):
        inputs = layers.Input((self.time_sequence, self.latent_dim))
        inputs2 = layers.Input((self.time_sequence, self.latent_dim))
        inputs3 = layers.Input((self.time_sequence, self.latent_dim))
        inputs4 = layers.Input((self.time_sequence, self.latent_dim))
        sample_sequence = inputs4[:,self.start_sampling_index:self.time_sequence:self.sampling_interval,:]
        sample_sequence = sample_sequence[:,:self.semantic_time_step,:]
        sample_global = inputs[:,-1,:]

        return tf.keras.Model([inputs,inputs2,inputs3,inputs4],
                              [sample_sequence,sample_global],name='discrete_time_period_extractor')

    def position_encoding(self,pos):
        pos_embedding = np.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            if i%2 == 0:
                pos_embedding[i] = np.sin(pos/(np.power(10000,2*i/self.latent_dim)))
            else:
                pos_embedding[i] = np.cos(pos/(np.power(10000,2*i/self.latent_dim)))

        return tf.math.l2_normalize(pos_embedding)


    def E_step(self, batch_embedding,projection_basis):
        batch_embedding = tf.math.l2_normalize(batch_embedding, axis=1)
        basis = tf.math.l2_normalize(projection_basis, axis=1)

        batch_embedding = tf.expand_dims(batch_embedding,2)
        batch_embedding = tf.broadcast_to(batch_embedding,[batch_embedding.shape[0],
                                                           self.semantic_time_step,
                                                           self.semantic_time_step,
                                                           self.latent_dim])
        self.check_batch_embedding_E = batch_embedding

        basis = tf.expand_dims(basis,1)
        basis = tf.broadcast_to(basis,[projection_basis.shape[0],self.semantic_time_step,self.semantic_time_step,self.latent_dim])
        self.check_basis_E = basis

        projection = tf.multiply(batch_embedding, basis)
        projection = tf.reduce_sum(projection,3)

        self.check_projection_E = projection
        max_value_projection = np.argmax(projection, axis=2)
        self.check_max_value_projection = max_value_projection

        return max_value_projection

    def extract_semantic(self):

        order_input = layers.Input(self.semantic_time_step)
        #order_input = tf.cast(order_input,tf.int32)
        #order_input = layers.Input(self.latent_dim)
        projection_basis = layers.Input((self.semantic_time_step,self.latent_dim))
        #projection_basis = projection_basis[0,:,:]
        #batch_semantic_embedding = tf.gather(projection_basis,
                                             #order_input)

        self.total_sementic = []
        for i in range(self.semantic_time_step):
            check = order_input == i
            check = tf.cast(check,tf.float32)
            check = tf.expand_dims(check,2)
            batch_semantic_embedding_single = tf.math.multiply(projection_basis,
                                                        check)
            batch_semantic_embedding_single = tf.reduce_sum(batch_semantic_embedding_single,axis=1)
            batch_semantic_embedding_single = tf.expand_dims(batch_semantic_embedding_single,axis=1)
            self.total_sementic.append(batch_semantic_embedding_single)

        batch_semantic_embedding = tf.concat(self.total_sementic,axis=1)
        #batch_semantic_embedding = tf.reduce_sum(batch_semantic_embedding,axis=0)

        self.check_batch_semantic_embedding = batch_semantic_embedding

        batch_time_progression_semantic_embedding = batch_semantic_embedding + self.position_embedding

        self.check_progression = batch_time_progression_semantic_embedding

        batch_time_progression_semantic_embedding = tf.reduce_sum(batch_time_progression_semantic_embedding,
                                                                  axis=1)

        batch_time_progression_semantic_embedding = tf.math.l2_normalize(batch_time_progression_semantic_embedding,
                                                                         axis=1)

        self.check_projection_final = batch_time_progression_semantic_embedding

        #return tf.keras.Model(projection_basis,batch_time_progression_semantic_embedding,name='semantic_extractor')

        return tf.keras.Model([projection_basis,order_input],
                              [batch_time_progression_semantic_embedding,projection_basis], name='semantic_extractor')

    def train_semantic_time_progression(self):
        """
        define model
        """
        input = layers.Input((self.time_sequence,35))
        self.tcn = self.tcn_encoder_second_last_level()
        self.time_extractor = self.discrete_time_period_extract()
        self.tcn_pull = self.tcn_pull()


        tcn = self.tcn(input)
        time_extractor = self.time_extractor(tcn)
        #global_pull = self.tcn_pull(tcn)
        self.model_extractor = tf.keras.Model(input, time_extractor, name="time_extractor")
        #self.model_pull = tf.keras.Model(input, global_pull, name="tcn_pull")
        #for epoch in range(self.pre_train_epoch):
            #print("\nStart of epoch %d" % (epoch,))

        input_projection = layers.Input((self.semantic_time_step,self.latent_dim))
        input_projection_order = layers.Input(self.semantic_time_step)
        self.basis_model = self.projection_model()
        #update_projection_basis, fake_input_order = self.basis_model([input_projection_batch, input_order])
        self.semantic_model = self.extract_semantic()

        basis_model = self.basis_model([input_projection,input_projection_order])
        semantic_model_output = self.semantic_model(basis_model)

        self.semantic_model_extract = tf.keras.Model([input_projection,input_projection_order],semantic_model_output)

        index = 1
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):


            x_batch_train_cohort =
            input_projection_batch = np.ones((self.batch_size,self.semantic_time_step,self.latent_dim))
            input_order = np.ones((self.batch_size, self.semantic_time_step))


            self.check_batch = x_batch_train
            extract_time,global_pull = self.model_extractor(x_batch_train)

            fake_semantic_embedding,projection_basis = self.semantic_model_extract([input_projection_batch,input_order])
            #self.check_global = extract_global
            #semantic_embedding, update_projection_basis = self.semantic_model_extract()
            self.check_extract_time = extract_time
            self.check_fake_embedding = fake_semantic_embedding
            projection = self.E_step(extract_time,projection_basis)
            self.check_projection = projection


            #if index == 1:
                #break


            with tf.GradientTape() as tape:
                # z1, z2 = self.att_lstm_model(data_aug1)[1], self.att_lstm_model(data_aug2)[0]
                extract_time, global_pull = self.model_extractor(x_batch_train)
                semantic_embedding, projection_basis = self.semantic_model_extract([input_projection_batch, projection])
                self.check_projection_basis = projection_basis

                loss, loss_prob = self.info_nce_loss(semantic_embedding, global_pull)
                #mse = tf.keras.losses.MeanSquaredError()
                #loss = mse(global_pull1,global_pull2)

            trainable_variables = self.model_extractor.trainable_variables+self.semantic_model_extract.trainable_variables
            gradients = \
                tape.gradient(loss, trainable_variables)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            optimizer.apply_gradients(zip(gradients, trainable_variables))
            self.check_loss = loss
            self.check_loss_prob = loss_prob
        

            if step % 20 == 0:
                print("Training loss(for one batch) at step %d: %.4f"
                      % (step, float(loss)))
                print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                #self.loss_track.append(loss)
                #self.loss_prob_track.append(loss_prob)

    def pre_train_infomax(self):
        #self.lstm = self.lstm_encoder()
        #self.lstm = self.tcn_encoder_second_last_level()
        self.auc_all = []
        input = layers.Input((self.time_sequence, 35))
        self.tcn = self.tcn_encoder_second_last_level()
        self.time_extractor = self.discrete_time_period_extract()
        tcn_pull = self.tcn_pull()

        tcn = self.tcn(input)
        time_extractor = self.time_extractor(tcn)
        global_pull = tcn_pull(tcn)
        self.model_extractor = tf.keras.Model(input, time_extractor, name="time_extractor")

        self.basis_model = self.projection_model()
        self.projection_layer = self.project_logit()

        self.loss_track = []
        self.loss_prob_track = []

        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))
            extract_val, global_val = self.model_extractor(self.val_data)
            prediction_val = self.projection_layer(global_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit,prediction_val)
            self.auc_all.append(val_acc)
            print("auc")
            print(val_acc)

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                input_projection_batch = np.ones((x_batch_train.shape[0], self.semantic_time_step, self.latent_dim))
                input_order = np.ones((x_batch_train.shape[0], self.semantic_time_step))

                self.check_batch = x_batch_train
                extract_time, global_pull = self.model_extractor(x_batch_train)

                projection_basis, projection_order = self.basis_model(
                    [input_projection_batch, input_order])
                # self.check_global = extract_global
                # semantic_embedding, update_projection_basis = self.semantic_model_extract()
                self.check_projection_basis = projection_basis
                self.check_extract_time = extract_time
                order_input = self.E_step(extract_time, projection_basis)
                self.check_projection = order_input

                #if index == 1:
                    #break


                with tf.GradientTape() as tape:

                    extract_time, global_pull = self.model_extractor(x_batch_train)
                    prediction = self.projection_layer(global_pull)
                    self.check_global_pull = global_pull
                    projection_basis, projection_order = self.basis_model(
                        [input_projection_batch, order_input])
                    self.total_sementic = []
                    for i in range(self.semantic_time_step):
                        check = order_input == i
                        check = tf.cast(check, tf.float32)
                        check = tf.expand_dims(check, 2)
                        self.check_check = check
                        projection_single = tf.broadcast_to(tf.expand_dims(projection_basis[0,i,:],0)
                                                            ,projection_basis.shape)
                        self.check_projection_single = projection_single
                        batch_semantic_embedding_single = tf.math.multiply(projection_single,
                                                                           check)
                        self.check_batch_semantic_embedding_single = batch_semantic_embedding_single
                        position_embedding = tf.cast(tf.expand_dims(self.position_embedding,0),tf.float32)
                        batch_semantic_embedding_single_position = tf.math.multiply(position_embedding,check)
                        self.check_batch_semantic_embedding_single_position = batch_semantic_embedding_single_position
                        batch_semantic_embedding_single_final = \
                            batch_semantic_embedding_single+batch_semantic_embedding_single_position
                        self.check_batch_semantic_embedding_single_final = batch_semantic_embedding_single_final
                        #batch_semantic_embedding_single = tf.reduce_sum(batch_semantic_embedding_single, axis=1)
                        #batch_semantic_embedding_single = tf.expand_dims(batch_semantic_embedding_single, axis=1)
                        self.total_sementic.append(batch_semantic_embedding_single_final)

                    batch_semantic_embedding_whole = tf.stack(self.total_sementic)
                    batch_semantic_embedding_whole = tf.reduce_sum(batch_semantic_embedding_whole,axis=0)
                    batch_semantic_embedding_whole = tf.reduce_sum(batch_semantic_embedding_whole,axis=1)
                    self.check_batch_semantic_embedding_whole = batch_semantic_embedding_whole
                    semantic_time_progression_loss = self.info_nce_loss(batch_semantic_embedding_whole,global_pull)
                    bceloss = tf.keras.losses.BinaryCrossentropy()(y_batch_train,prediction)

                    loss = semantic_time_progression_loss + bceloss

                self.check_loss = loss
                #z1, z2 = self.att_lstm_model(data_aug1)[1], self.att_lstm_model(data_aug2)[0]

                trainable_weights = self.model_extractor.trainable_weights+\
                                    self.basis_model.trainable_weights+self.projection_layer.trainable_weights
                gradients = tape.gradient(loss, trainable_weights)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients, trainable_weights))
                self.check_loss = loss
                #self.check_loss_prob = loss_prob

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)

    def semantic_extraction(self):
        real_sequence = self.train_data_origin[:, self.start_sampling_index:self.time_sequence:self.sampling_interval, :]
        real_sequence = real_sequence[:, :self.semantic_time_step, :]
        input_projection_batch = np.ones((self.train_data.shape[0], self.semantic_time_step, self.latent_dim))
        input_order = np.ones((self.train_data.shape[0], self.semantic_time_step))

        extract_time, global_pull = self.model_extractor(self.train_data)
        projection_basis, projection_order = self.basis_model(
            [input_projection_batch, input_order])
        self.check_projection_basis = projection_basis
        self.check_extract_time = extract_time
        order_input = self.E_step(extract_time, projection_basis)
        self.check_order_input = order_input

        self.semantic_real_basis0 = []
        self.semantic_real_basis1 = []
        self.semantic_real_basis2 = []
        self.semantic_real_basis3 = []
        for i in range(self.train_data.shape[0]):
            for j in range(self.semantic_time_step):
                if order_input[i,j] == 0:
                    self.semantic_real_basis0.append(real_sequence[i,j,:])
                if order_input[i,j] == 1:
                    self.semantic_real_basis1.append(real_sequence[i, j, :])
                if order_input[i,j] == 2:
                    self.semantic_real_basis2.append(real_sequence[i, j, :])
                if order_input[i,j] == 3:
                    self.semantic_real_basis3.append(real_sequence[i, j, :])

        #for i in range(self.semantic_time_step):
        self.semantic0 = np.zeros(34)
        self.semantic1 = np.zeros(34)
        self.semantic2 = np.zeros(34)
        self.semantic3 = np.zeros(34)

        for i in range(34):
            k = np.array(self.semantic_real_basis0)[:,i]
            self.semantic0[i] = np.median(k[np.nonzero(k)])
            k = np.array(self.semantic_real_basis1)[:, i]
            self.semantic1[i] = np.median(k[np.nonzero(k)])
            k = np.array(self.semantic_real_basis2)[:, i]
            self.semantic2[i] = np.median(k[np.nonzero(k)])
            k = np.array(self.semantic_real_basis3)[:, i]
            self.semantic3[i] = np.median(k[np.nonzero(k)])




    def train_whole(self):
        input = layers.Input((self.time_sequence, 35))
        self.tcn = self.tcn_encoder_second_last_level()
        self.time_extractor = self.discrete_time_period_extract()
        self.tcn_pull = self.tcn_pull()
        self.auc_all = []
        tcn = self.tcn(input)
        time_extractor = self.time_extractor(tcn)
        self.loss_track = []

        self.model_extractor = tf.keras.Model(input, time_extractor, name="time_extractor")

        self.projection_layer = self.project_logit()
        #self.tcn_model = tf.keras.Sequential([self.model_extractor,self.projection_layer])
        self.bceloss = tf.keras.losses.BinaryCrossentropy()
        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            extract_val, global_val = self.model_extractor(self.val_data)
            prediction_val = self.projection_layer(global_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit, prediction_val)
            print("auc")
            print(val_acc)
            self.auc_all.append(val_acc)
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):

                with tf.GradientTape() as tape:
                    extract_time,global_pull = self.model_extractor(x_batch_train)
                    prediction = self.projection_layer(global_pull)
                    #self.check_global_pull = global_pull
                    #prediction = self.tcn_model(x_batch_train)
                    loss = self.bceloss(y_batch_train,prediction)

                #self.check_prediction = prediction
                #self.check_y = y_batch_train

                gradients = \
                    tape.gradient(loss,
                                  self.model_extractor.trainable_variables+self.projection_layer.trainable_weights)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients,
                                              self.model_extractor.trainable_variables+self.projection_layer.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)
                    #self.loss_prob_track.append(loss_prob)









    def project_logit(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                layers.Input((50)),
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
        inputs = layers.Input((self.time_sequence, 35))
        inputs_mask = layers.Masking(mask_value=0, input_shape=(self.time_sequence, 35))(inputs)
        #self.lstm = self.lstm_encoder()
        self.lstm = self.tcn_encoder_second_last_level()
        self.lstm_pool = self.tcn_pull()
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
