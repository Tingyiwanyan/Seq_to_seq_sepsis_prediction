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

semantic_step_global = 6
unsupervised_cluster_num = 30
latent_dim_global = 100
positive_sample_size = 5

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
        self.neg_size = self.batch_size
        self.pos_size = positive_sample_size
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 20
        self.feature_num = 34
        self.pre_train_epoch = 40
        self.latent_dim = latent_dim_global
        self.tau = 1
        self.time_sequence = self.read_d.time_sequence
        self.tcn_filter_size = 5
        self.semantic_time_step = semantic_step_global
        self.unsupervised_cluster_num = unsupervised_cluster_num
        self.start_sampling_index = 5
        self.sampling_interval = 5
        self.converge_threshold_E = 10
        self.max_value_projection = np.zeros((self.batch_size,self.semantic_time_step))
        self.basis_input = np.ones((self.unsupervised_cluster_num,self.latent_dim))

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
            decay_rate=0.3)

    def create_memory_bank(self):
        #self.train_data, self.train_logit,self.train_sofa,self.train_sofa_score = self.aquire_data(0, self.train_data, self.length_train)
        #self.test_data, self.test_logit,self.test_sofa,self.test_sofa_score = self.aquire_data(0, self.test_data, self.length_test)
        #self.val_data, self.val_logit,self.val_sofa,self.val_sofa_score = self.aquire_data(0, self.validate_data, self.length_val)

        file_path = '/home/tingyi/physionet_data/Interpolate_data/'
        with open(file_path + 'train.npy', 'rb') as f:
            self.train_data = np.load(f)
        with open(file_path + 'train_logit.npy', 'rb') as f:
            self.train_logit = np.load(f)
        with open(file_path + 'train_on_site_time.npy', 'rb') as f:
            self.train_on_site_time = np.load(f)

        #with open(file_path + 'test.npy', 'rb') as f:
            #self.test_data = np.load(f)
        #with open(file_path + 'test_logit.npy', 'rb') as f:
            #self.test_logit = np.load(f)
        with open(file_path + 'val.npy', 'rb') as f:
            self.val_data = np.load(f)
        with open(file_path + 'val_logit.npy', 'rb') as f:
            self.val_logit = np.load(f)
        with open(file_path + 'val_on_site_time.npy', 'rb') as f:
            self.val_on_site_time = np.load(f)
        #with open(file_path + 'train_sofa_score.npy', 'rb') as f:
            #self.train_sofa_score = np.load(f)
        #with open(file_path + 'train_sofa.npy', 'rb') as f:
            #self.train_sofa = np.load(f)

        with open(file_path + 'train_origin.npy', 'rb') as f:
            self.train_data_origin = np.load(f)


        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_logit,
                                                                 self.train_on_site_time))#,self.train_sofa_score))
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

    def compute_negative_paris(self, z, global_pull_cohort,global_pull_control,label):
        z = tf.math.l2_normalize(z, axis=1)

        global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=1)
        global_pull_control = tf.math.l2_normalize(global_pull_control, axis=1)

        similarity_matrix_cohort = tf.matmul(z, tf.transpose(global_pull_cohort))
        similarity_matrix_control = tf.matmul(z, tf.transpose(global_pull_control))

        neg_cohort_sum = tf.reduce_sum(tf.math.exp(similarity_matrix_cohort / self.tau), 1)
        self.check_neg_cohort_sum = neg_cohort_sum
        neg_control_sum = tf.reduce_sum(tf.math.exp(similarity_matrix_control / self.tau), 1)
        self.check_neg_control_sum = neg_control_sum
        label = tf.cast(label,tf.int32)
        self.check_label = label

        neg_sum_both = tf.stack((neg_cohort_sum,neg_control_sum),1)
        self.check_neg_sum_both = neg_sum_both
        negative_dot_prods_sum = [neg_sum_both[i,label[i]] for i in range(z.shape[0])]
        self.check_negative_dot_prods_sum = negative_dot_prods_sum

        return negative_dot_prods_sum

    def info_nce_loss(self, z, p, global_pull_cohort,global_pull_control,label):
        positive_dot_prod_sum = self.compute_positive_pair(z,p)
        negative_dot_prod_sum = self.compute_negative_paris(z,global_pull_cohort,global_pull_control,label)

        denominator = tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator))
        loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum, denominator), 0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss

    def compute_positive_pair_cl(self,z, global_pull_cohort, global_pull_control, label):
        z = tf.math.l2_normalize(z, axis=1)

        global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=1)
        global_pull_control = tf.math.l2_normalize(global_pull_control, axis=1)

        similarity_matrix_cohort = tf.matmul(z, tf.transpose(global_pull_cohort))
        similarity_matrix_control = tf.matmul(z, tf.transpose(global_pull_control))

        pos_cohort = tf.math.exp(similarity_matrix_cohort / self.tau)
        self.pos_cohort = pos_cohort
        pos_control = tf.math.exp(similarity_matrix_control / self.tau)
        self.pos_control = pos_control
        label = tf.cast(label, tf.int32)
        self.check_label = label

        pos_both = tf.stack((pos_control, pos_cohort), 2)
        self.check_pos_both = pos_both
        pos_dot_prods = [pos_both[i, :, label[i]] for i in range(z.shape[0])]
        self.check_pos_dot_prods = tf.stack(pos_dot_prods)

        return tf.stack(pos_dot_prods)

    def compute_negative_pair_cl(self, z, global_pull_cohort, global_pull_control, label):
        z = tf.math.l2_normalize(z, axis=1)

        global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=1)
        global_pull_control = tf.math.l2_normalize(global_pull_control, axis=1)

        similarity_matrix_cohort = tf.matmul(z, tf.transpose(global_pull_cohort))
        similarity_matrix_control = tf.matmul(z, tf.transpose(global_pull_control))

        neg_cohort_sum = tf.reduce_sum(tf.math.exp(similarity_matrix_cohort / self.tau), 1)
        #self.check_neg_cohort_sum = neg_cohort_sum
        neg_control_sum = tf.reduce_sum(tf.math.exp(similarity_matrix_control / self.tau), 1)
        #self.check_neg_control_sum = neg_control_sum
        label = tf.cast(label, tf.int32)
        #self.check_label = label

        neg_sum_both = tf.stack((neg_cohort_sum, neg_control_sum), 1)
        #self.check_neg_sum_both = neg_sum_both
        negative_dot_prods_sum = [neg_sum_both[i, label[i]] for i in range(z.shape[0])]
        #self.check_negative_dot_prods_sum = negative_dot_prods_sum

        return negative_dot_prods_sum


    def supervised_cl_loss(self, z, global_pull_cohort,global_pull_control,
                           global_pull_cohort_pos,global_pull_control_pos,label):
        positive_dot_prod = self.compute_positive_pair_cl(z, global_pull_cohort_pos, global_pull_control_pos, label)
        negative_dot_prod_sum = self.compute_negative_pair_cl(z, global_pull_cohort, global_pull_control, label)

        negative_dot_prod_sum = tf.expand_dims(negative_dot_prod_sum,1)
        negative_dot_prod_sum = tf.broadcast_to(negative_dot_prod_sum,shape = positive_dot_prod.shape)

        denominator = tf.math.add(positive_dot_prod, negative_dot_prod_sum)
        nomalized_prob_log = tf.reduce_sum(tf.math.log(tf.math.divide(positive_dot_prod,denominator)),1)
        self.check_divide = nomalized_prob_log
        #nomalized_prob_log = tf.math.log(divide)
        #loss_prob = tf.reduce_mean(tf.math.divide(positive_dot_prod_sum, denominator), 0)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss


    def compute_positive_pair_un(self, z, p):
        z = tf.math.l2_normalize(z, axis=-1)
        p = tf.math.l2_normalize(p, axis=-1)

        positive_dot_prod = tf.multiply(z, p)
        positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(positive_dot_prod, -1) / self.tau)

        return positive_dot_prod_sum


    def unsupervised_prototype_loss(self,extract_time, projection_basis,order_input):
        extract_time = tf.math.l2_normalize(extract_time, axis=1)
        #extract_time_order = tf.reshape(extract_time,
                                        #[extract_time.shape[0]*extract_time.shape[1],extract_time.shape[2]])
        projection_basis = tf.math.l2_normalize(projection_basis, axis=-1)
        projection_basis_expand = tf.expand_dims(projection_basis, axis=1)
        projection_basis_broad = tf.broadcast_to(projection_basis_expand,
                                                 [projection_basis.shape[0],extract_time.shape[1],
                                                  projection_basis.shape[1],projection_basis.shape[2]])

        extract_time_expand = tf.expand_dims(extract_time, axis=2)
        extract_time_broad = tf.broadcast_to(extract_time_expand,[extract_time.shape[0],extract_time.shape[1],
                                                                  projection_basis.shape[1],extract_time.shape[2]])

        denominator = tf.multiply(projection_basis_broad,extract_time_broad)

        negative_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(denominator, -1) / self.tau),2)

        negative_dot_prod_sum = tf.reshape(negative_dot_prod_sum,
                                           [negative_dot_prod_sum.shape[0]*negative_dot_prod_sum.shape[1]])

        self.total_sementic_un = []
        for i in range(self.unsupervised_cluster_num):
            check = order_input == i
            check = tf.cast(check, tf.float32)
            check = tf.expand_dims(check, 2)
            self.check_un = check
            # projection_single = tf.broadcast_to(tf.expand_dims(projection_basis[0,i,:],0)
            # ,projection_basis.shape)

            projection_basis_single = tf.expand_dims(projection_basis[:, i, :], 1)
            projection_single = tf.broadcast_to(projection_basis_single, shape=(projection_basis.shape[0],
                                                                                check.shape[1],
                                                                                projection_basis.shape[2]))

            self.check_projection_single_un = projection_single
            batch_semantic_embedding_single = tf.math.multiply(projection_single,
                                                               check)
            self.check_batch_semantic_embedding_single = batch_semantic_embedding_single
            # batch_semantic_embedding_single = tf.reduce_sum(batch_semantic_embedding_single, axis=1)
            # batch_semantic_embedding_single = tf.expand_dims(batch_semantic_embedding_single, axis=1)
            self.total_sementic_un.append(batch_semantic_embedding_single)

        batch_semantic_embedding_whole = tf.stack(self.total_sementic_un)
        batch_semantic_embedding_whole = tf.reduce_sum(batch_semantic_embedding_whole, axis=0)

        pos_prod_sum = self.compute_positive_pair_un(extract_time, batch_semantic_embedding_whole)

        pos_prod_sum = tf.reshape(pos_prod_sum,[pos_prod_sum.shape[0]*pos_prod_sum.shape[1]])

        nomalized_prob_log = tf.math.log(tf.math.divide(pos_prod_sum, negative_dot_prod_sum))

        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss

    def projection_regularize_loss(self,projection_basis):
        projection_basis = projection_basis[0,:,:]

        similarity_matrix = tf.matmul(projection_basis, tf.transpose(projection_basis))
        mask = tf.linalg.diag(tf.zeros(projection_basis.shape[0]), padding_value=1)

        negative_dot_prods = tf.math.abs(tf.multiply(similarity_matrix, mask))
        projection_regular_loss = tf.reduce_mean(tf.reduce_sum(negative_dot_prods , 1))

        return projection_regular_loss


    def first_lvl_resolution_deconv(self):
        inputs = layers.Input((1,self.latent_dim))

        tcn_deconv1 = tf.keras.layers.Conv1DTranspose(self.feature_num, self.tcn_filter_size)

        output = tcn_deconv1(inputs)

        return tf.keras.Model(inputs, output, name='tcn_deconv1')

    def one_h_resolution_deconv(self):
        inputs = layers.Input((1,self.latent_dim))

        tcn_deconv1 = tf.keras.layers.Conv1DTranspose(self.feature_num, 1)

        output = tcn_deconv1(inputs)

        return tf.keras.Model(inputs, output, name='tcn_deconv1')



    def tcn_encoder_second_last_level(self):
        """
        Implement tcn encoder
        """
        """
        define dilation for each layer(24 hours)
        """
        dilation1 = 1 #3 hours
        dilation2 = 2 #7hours
        dilation3 = 4 #15hours
        dilation4 = 8 # with filter size 3, 8x3=24, already covers the whole time sequence
        #dilation5 = 16

        """
        define the identical resolution
        """
        inputs = layers.Input((self.time_sequence, self.feature_num))
        #tcn_conv0 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
                                           #dilation_rate=dilation1, padding='valid')
        tcn_conv0 = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu', dilation_rate=1,padding='valid')
        layernorm1 = tf.keras.layers.BatchNormalization()
        #padding_1 = (self.tcn_filter_size - 1) * dilation1
        #inputs1 = tf.pad(inputs, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_1)
        self.outputs0 = tcn_conv0(inputs)
        #self.outputs1 = conv1_identity(self.outputs1)

        """
        define the first tcn layer, dilation=1
        """
        #inputs = layers.Input((self.time_sequence,self.feature_num))
        tcn_conv1 = tf.keras.layers.Conv1D(self.latent_dim,self.tcn_filter_size,activation='relu',dilation_rate=dilation1,padding='valid')
        conv1_identity = tf.keras.layers.Conv1D(self.latent_dim,1,activation='relu',dilation_rate=1)
        layernorm1 = tf.keras.layers.BatchNormalization()
        padding_1 = (self.tcn_filter_size-1) * dilation1
        #inputs1 = tf.pad(inputs, tf.constant([[0,0],[1,0],[0,0]]) * padding_1)

        inputs1 = tf.pad(self.outputs0, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_1)
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
                              [inputs, self.outputs4, self.outputs3, self.outputs2,self.outputs1,self.outputs0], name='tcn_encoder')

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

    def position_project_layer(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                #layers.Input((50)),
                layers.Dense(
                    self.latent_dim,
                    #use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                )
            ],
            name="position_projection",
        )
        return model

    def discrete_time_period_extract(self):
        original_inputs = layers.Input((self.time_sequence,self.feature_num))
        inputs = layers.Input((self.time_sequence, self.latent_dim))
        inputs2 = layers.Input((self.time_sequence, self.latent_dim))
        inputs3 = layers.Input((self.time_sequence, self.latent_dim))
        inputs4 = layers.Input((self.time_sequence, self.latent_dim))
        sample_sequence = inputs4[:,self.start_sampling_index:self.time_sequence:self.sampling_interval,:]
        sample_sequence = sample_sequence[:,:self.semantic_time_step,:]
        sample_global = inputs[:,-1,:]


        sample_original_time_list = []
        #group_index = np.zeros((self.semantic_time_step,self.tcn_filter_size))
        time_seq_range = np.array(range(self.time_sequence))
        time_index = time_seq_range[self.start_sampling_index:self.time_sequence:self.sampling_interval]
        time_index = time_index[:self.semantic_time_step]
        for i in range(self.semantic_time_step):
            group_index = time_seq_range[(time_index[i]-self.tcn_filter_size+1):time_index[i]+1]
            original_inputs_trans = tf.transpose(original_inputs,[1,0,2])
            sample_sequence_single = tf.gather(original_inputs_trans,group_index)
            sample_sequence_single = tf.transpose(sample_sequence_single,[1,0,2])
            sample_original_time_list.append(sample_sequence_single)

        self.check_sample_original_time_list = sample_original_time_list
        sample_original_time_sequence = tf.stack(sample_original_time_list,0)
        sample_original_time_sequence = tf.transpose(sample_original_time_sequence,[1,0,2,3])


        return tf.keras.Model([original_inputs,inputs,inputs2,inputs3,inputs4],
                              [sample_sequence,sample_global,sample_original_time_sequence],name='discrete_time_period_extractor')


    def position_encoding(self,pos):
        pos_embedding = np.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            if i%2 == 0:
                pos_embedding[i] = np.sin(pos/(np.power(2,2*i/self.latent_dim)))
            else:
                pos_embedding[i] = np.cos(pos/(np.power(2,2*i/self.latent_dim)))

        return tf.math.l2_normalize(pos_embedding)

    """
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
    """

    def E_step(self,batch_embedding,projection_basis):
        batch_embedding = tf.math.l2_normalize(batch_embedding, axis=1)

        projection_basis = projection_basis[0]
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
                                                            self.semantic_time_step,
                                                            self.unsupervised_cluster_num,
                                                            self.latent_dim])

        self.check_batch_embedding_E = batch_embedding

        check_converge = 100 * np.ones((batch_embedding.shape[0] * batch_embedding.shape[1]))

        self.check_check_converge = check_converge

        check_converge_num = 1000
        self.check_converge_num = check_converge_num

        max_value_projection = 0

        while(check_converge_num > self.converge_threshold_E):
            #print(check_converge_num)
            basis = tf.math.l2_normalize(projection_basis, axis=-1)
            self.check_basis = basis

            basis = tf.expand_dims(basis, 1)
            basis = tf.broadcast_to(basis, [projection_basis.shape[0], self.semantic_time_step,
                                            self.unsupervised_cluster_num,
                                            self.latent_dim])
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




        #print("converged")

        return max_value_projection, projection_basis
    
    
    def train_semantic_time_pregression(self):
        #self.lstm = self.lstm_encoder()
        #self.lstm = self.tcn_encoder_second_last_level()
        self.auc_all = []
        self.auc_iteration = []
        input = layers.Input((self.time_sequence, self.feature_num))
        self.tcn = self.tcn_encoder_second_last_level()
        self.tcn_1_lvl = self.first_lvl_resolution_deconv()
        self.time_extractor = self.discrete_time_period_extract()
        self.position_project = self.position_project_layer()
        tcn_pull = self.tcn_pull()

        tcn = self.tcn(input)
        time_extractor = self.time_extractor(tcn)
        #global_pull = tcn_pull(tcn)
        self.model_extractor = tf.keras.Model(input, time_extractor, name="time_extractor")

        self.model_tcn = tf.keras.Model(input, tcn,name="tcn_model")

        self.basis_model = self.projection_model()
        self.projection_layer = self.project_logit()

        self.loss_track = []
        self.loss_track_mse = []


        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))
            extract_val, global_val, sample_sequence_val = self.model_extractor(self.val_data)
            prediction_val = self.projection_layer(global_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit,prediction_val)
            self.auc_all.append(val_acc)
            print("auc")
            print(val_acc)

            extract_time_total, global_pull_total, sample_sequence_train_total = self.model_extractor(self.train_data)
            projection_basis = self.init_projection_basis
            projection_basis = tf.expand_dims(projection_basis, 0)

            order_input_total, projection_basis_total = self.E_step(extract_time_total, projection_basis)

            #projection_basis = projection_basis[0]
            #projection_basis = tf.expand_dims(projection_basis,0)

            self.check_order_input_total = order_input_total
            self.check_projection_basis_total = projection_basis_total

            self.converge_projection_basis = projection_basis_total[0]

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.train_data, self.train_logit,
                 order_input_total,projection_basis_total, sample_sequence_train_total))
            self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

            for step, (x_batch_train, y_batch_train,order_input,
                       projection_basis, sample_sequence_train) in enumerate(self.train_dataset):

                random_indices_cohort = np.random.choice(self.num_cohort,size=self.neg_size,replace=False)
                random_indices_control = np.random.choice(self.num_control,size=self.neg_size,replace=False)

                x_batch_train_cohort = self.memory_bank_cohort[random_indices_cohort,:,:]
                x_batch_train_control = self.memory_bank_control[random_indices_control,:,:]

                random_indices_cohort_pos = np.random.choice(self.num_cohort, size=self.pos_size, replace=False)
                random_indices_control_pos = np.random.choice(self.num_control, size=self.pos_size, replace=False)

                x_batch_train_cohort_pos = self.memory_bank_cohort[random_indices_cohort_pos, :, :]
                x_batch_train_control_pos = self.memory_bank_control[random_indices_control_pos, :, :]
                #input_projection_batch = np.ones((x_batch_train.shape[0], self.semantic_time_step, self.latent_dim))
                #input_order = np.ones((x_batch_train.shape[0], self.semantic_time_step))

                #if len(projection_basis.shape) !=3:
                    #projection_basis = tf.expand_dims(projection_basis,0)
                    #projection_basis = tf.broadcast_to(projection_basis,
                                                        #shape=(x_batch_train.shape[0],
                                                        #projection_basis.shape[1],projection_basis.shape[2]))
                sample_sequence_train = tf.reshape(sample_sequence_train,
                                                   [sample_sequence_train.shape[0] * sample_sequence_train.shape[1],
                                                    sample_sequence_train.shape[2],
                                                    sample_sequence_train.shape[3]])

                #sample_sequence_train = tf.transpose(sample_sequence_train,[1,0,2,3])

                self.check_sample_sequence_train = sample_sequence_train

                self.check_projection_basis = projection_basis

                self.check_batch = x_batch_train
                #extract_time, global_pull, sample_sequence = self.model_extractor(x_batch_train)

                #projection_basis, projection_order = self.basis_model(
                    #[input_projection_batch, input_order])
                # self.check_global = extract_global
                # semantic_embedding, update_projection_basis = self.semantic_model_extract()

                #self.check_extract_time = extract_time
                #order_input, projection_basis = self.E_step(extract_time, projection_basis)
                self.check_projection = order_input


                with tf.GradientTape() as tape:

                    extract_time, global_pull, sample_sequence = self.model_extractor(x_batch_train)
                    extract_time_cohort, global_pull_cohort, sample_sequence_cohort = \
                        self.model_extractor(x_batch_train_cohort)
                    extract_time_control, global_pull_control, sample_sequence_control = \
                        self.model_extractor(x_batch_train_control)

                    """
                    extract_time_cohort_pos, global_pull_cohort_pos, sample_sequence_cohort_pos = \
                        self.model_extractor(x_batch_train_cohort_pos)
                    extract_time_control_pos, global_pull_control_pos, sample_sequence_control_pos = \
                        self.model_extractor(x_batch_train_control_pos)
                    """


                    prediction = self.projection_layer(global_pull)
                    self.check_extract_time = extract_time
                    self.check_global_pull = global_pull
                    #projection_basis, projection_order = self.basis_model(
                        #[input_projection_batch, order_input])
                    self.total_sementic = []
                    for i in range(self.unsupervised_cluster_num):
                        check = order_input == i
                        check = tf.cast(check, tf.float32)
                        check = tf.expand_dims(check, 2)
                        self.check_check = check
                        #projection_single = tf.broadcast_to(tf.expand_dims(projection_basis[0,i,:],0)
                                                            #,projection_basis.shape)

                        projection_basis_single = tf.expand_dims(projection_basis[:, i, :], 1)
                        projection_single = tf.broadcast_to(projection_basis_single, shape=(projection_basis.shape[0],
                                                                                            check.shape[1],
                                                                                            projection_basis.shape[2]))

                        self.check_projection_single = projection_single
                        batch_semantic_embedding_single = tf.math.multiply(projection_single,
                                                                           check)
                        self.check_batch_semantic_embedding_single = batch_semantic_embedding_single
                        position_embedding = tf.cast(tf.expand_dims(self.position_embedding,0),tf.float32)
                        batch_semantic_embedding_single_position = tf.math.multiply(position_embedding,check)
                        self.check_batch_semantic_embedding_single_position = batch_semantic_embedding_single_position
                        batch_semantic_embedding_single_final = \
                            batch_semantic_embedding_single*batch_semantic_embedding_single_position
                        self.check_batch_semantic_embedding_single_final = batch_semantic_embedding_single_final
                        #batch_semantic_embedding_single = tf.reduce_sum(batch_semantic_embedding_single, axis=1)
                        #batch_semantic_embedding_single = tf.expand_dims(batch_semantic_embedding_single, axis=1)
                        self.total_sementic.append(batch_semantic_embedding_single_final)

                    batch_semantic_embedding_whole_ = tf.stack(self.total_sementic)
                    self.check_batch_semantic_embedding_whole_ = batch_semantic_embedding_whole_

                    batch_semantic_embedding_whole = self.position_project(batch_semantic_embedding_whole_)
                    #batch_semantic_embedding_whole = tf.math.sigmoid(batch_semantic_embedding_whole_)
                    batch_semantic_embedding_whole = tf.reduce_sum(batch_semantic_embedding_whole,axis=0)
                    batch_semantic_embedding_whole = tf.reduce_sum(batch_semantic_embedding_whole,axis=1)
                    self.check_batch_semantic_embedding_whole = batch_semantic_embedding_whole
                    semantic_time_progression_loss = self.info_nce_loss(batch_semantic_embedding_whole,global_pull,
                                                                        global_pull_cohort,global_pull_control,
                                                                        y_batch_train)

                    #semantic_time_progression_loss = self.info_nce_loss(batch_semantic_embedding_whole, global_pull)

                    unsupervised_loss = self.unsupervised_prototype_loss(extract_time, projection_basis, order_input)
                    #supervised_loss = self.supervised_cl_loss(global_pull,
                                                              #global_pull_cohort,global_pull_control,
                                                              #global_pull_cohort_pos,global_pull_control_pos,
                                                              #y_batch_train)
                    bceloss = tf.keras.losses.BinaryCrossentropy()(y_batch_train,prediction)



                    #projection_regular_loss = self.projection_regularize_loss(projection_basis)

                    loss = bceloss + 0.6*semantic_time_progression_loss + 0.2*unsupervised_loss#+0.4*supervised_loss
                           #+ 0.2*projection_regular_loss


                    #loss = bceloss

                self.check_loss = loss
                #z1, z2 = self.att_lstm_model(data_aug1)[1], self.att_lstm_model(data_aug2)[0]

                trainable_weights = self.model_extractor.trainable_weights+self.projection_layer.trainable_weights# + \
                                    #self.position_project.trainable_weights
                                    #+self.tcn_1_lvl.trainable_weights
                                    #self.basis_model.trainable_weights
                gradients = tape.gradient(loss, trainable_weights)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients, trainable_weights))
                self.check_loss = loss
                #self.check_loss_prob = loss_prob

                if step % 10 == 0:
                    print("Training loss(for one batch, semantic_progress+bce) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)

                    extract_val_iter, global_val_iter, sample_sequence_val_iter = self.model_extractor(self.val_data)
                    prediction_val_iter = self.projection_layer(global_val_iter)
                    self.check_prediction_val_iter = prediction_val_iter
                    val_acc_iter = roc_auc_score(self.val_logit, prediction_val_iter)
                    self.auc_iteration.append(val_acc_iter)
                    print("auc")
                    print(val_acc_iter)


                with tf.GradientTape() as tape:
                    extract_time_reshape = tf.reshape(extract_time, [extract_time.shape[0] * extract_time.shape[1],
                                                                     extract_time.shape[2]])

                    extract_time_reshape = tf.expand_dims(extract_time_reshape, 1)

                    reconstruct_1_lvl = self.tcn_1_lvl(extract_time_reshape)

                    self.check_extract_time_reshape = extract_time_reshape
                    self.check_reconstruct_1_lvl = reconstruct_1_lvl

                    mseloss = tf.keras.losses.MeanSquaredError()(sample_sequence_train, reconstruct_1_lvl)

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
        self.reconstruct_semantic_origin = np.zeros((self.unsupervised_cluster_num, self.tcn_filter_size, self.feature_num))
        projection_basis = tf.expand_dims(self.check_projection_basis[0],1)
        self.reconstruct_semantic_norm = self.tcn_1_lvl(projection_basis)

        for k in range(self.unsupervised_cluster_num):
            for i in range(self.feature_num):
                if self.read_d.std_all[i] == 0:
                    self.reconstruct_semantic_origin[:,:,i] = self.read_d.ave_all[i]
                else:
                    for j in range(self.tcn_filter_size):
                        self.reconstruct_semantic_origin[k, j, i] = \
                            (self.reconstruct_semantic_norm[k, j, i] * self.read_d.std_all[i]) + self.read_d.ave_all[i]

    def semantic_extraction(self):
        real_sequence = self.train_data_origin[:, self.start_sampling_index:self.time_sequence:self.sampling_interval, :]
        real_sequence = real_sequence[:, :self.semantic_time_step, :]
        input_projection_batch = np.ones((self.train_data.shape[0], self.semantic_time_step, self.latent_dim))
        input_order = np.ones((self.train_data.shape[0], self.semantic_time_step))

        extract_time, global_pull,sample_sequence = self.model_extractor(self.train_data)
        #projection_basis, projection_order = self.basis_model(
            #[input_projection_batch, input_order])

        projection_basis = self.check_projection_basis_total
        self.check_projection_basis = projection_basis
        self.check_extract_time = extract_time
        order_input,projection_basis_ = self.E_step(extract_time, projection_basis)
        self.check_order_input = order_input
        """
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
        """
        sepsis_label = np.where(self.train_logit==1)[0]
        non_sepsis_label = np.where(self.train_logit==0)[0]

        self.sepsis_order = order_input[sepsis_label,:]
        self.non_sepsis_order = order_input[non_sepsis_label,:]

        self.row_sepsis = npi.mode(self.sepsis_order)
        self.row_non_sepsis = npi.mode(self.non_sepsis_order)

        self.remove_sepsis_order = []
        self.remove_non_sepsis_order = []

        for i in range(self.sepsis_order.shape[0]):
            l = self.sepsis_order[i,:] == self.row_sepsis
            if False in l:
                self.remove_sepsis_order.append(self.sepsis_order[i,:])

        for i in range(self.non_sepsis_order.shape[0]):
            l = self.non_sepsis_order[i,:] == self.row_non_sepsis
            if False in l:
                self.remove_non_sepsis_order.append(self.non_sepsis_order[i,:])

        self.remove_sepsis_order = np.stack(self.remove_sepsis_order,0)
        self.remove_non_sepsis_order = np.stack(self.remove_non_sepsis_order,0)

        self.row_sepsis_remove = npi.mode(self.remove_sepsis_order)
        self.row_non_sepsis_remove = npi.mode(self.remove_non_sepsis_order)

        self.remove_sepsis_order_sec = []
        self.remove_non_sepsis_order_sec = []

        for i in range(self.remove_sepsis_order.shape[0]):
            l = self.remove_sepsis_order[i, :] == self.row_sepsis_remove
            if False in l:
                self.remove_sepsis_order_sec.append(self.remove_sepsis_order[i, :])

        for i in range(self.remove_non_sepsis_order.shape[0]):
            l = self.remove_non_sepsis_order[i, :] == self.row_non_sepsis_remove
            if False in l:
                self.remove_non_sepsis_order_sec.append(self.remove_non_sepsis_order[i, :])



    def train_whole(self):
        input = layers.Input((self.time_sequence, self.feature_num))
        self.tcn = self.tcn_encoder_second_last_level()
        self.time_extractor = self.discrete_time_period_extract()
        #self.tcn_pull = self.tcn_pull()
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

            extract_val, global_val,k = self.model_extractor(self.val_data)
            prediction_val = self.projection_layer(global_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit, prediction_val)
            print("auc")
            print(val_acc)
            self.auc_all.append(val_acc)
            for step, (x_batch_train, y_batch_train,on_site_time) in enumerate(self.train_dataset):

                with tf.GradientTape() as tape:
                    extract_time,global_pull,k = self.model_extractor(x_batch_train)
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


    def train_standard(self):
        #input = layers.Input((self.time_sequence, self.feature_num))
        self.tcn = self.tcn_encoder_second_last_level()
        #tcn = self.tcn(input)
        self.auc_all = []
        self.loss_track = []
        #self.model_extractor = tf.keras.Model(input, tcn, name="time_extractor")
        self.projection_layer = self.project_logit()
        self.bceloss = tf.keras.losses.BinaryCrossentropy()

        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            #extract_val, global_val,k = self.model_extractor(self.val_data)
            tcn_temporal_output_val = self.tcn(self.val_data)
            last_layer_output_val = tcn_temporal_output_val[1]
            on_site_extract_val = [last_layer_output_val[i,int(self.val_on_site_time[i]-1),:] for i in range(self.val_on_site_time.shape[0])]
            on_site_extract_array_val = tf.stack(on_site_extract_val)
            prediction_val = self.projection_layer(on_site_extract_array_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit, prediction_val)
            print("auc")
            print(val_acc)
            self.auc_all.append(val_acc)
            for step, (x_batch_train, y_batch_train, on_site_time) in enumerate(self.train_dataset):
                self.check_x_batch = x_batch_train
                self.check_on_site_time = on_site_time
                self.check_label = y_batch_train
                with tf.GradientTape() as tape:
                    tcn_temporal_output = self.tcn(x_batch_train)
                    self.check_output = tcn_temporal_output
                    last_layer_output = tcn_temporal_output[1]
                    on_site_extract = [last_layer_output[i,int(on_site_time[i]-1),:] for i in range(on_site_time.shape[0])]
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
                #layers.Input((50)),
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
        inputs = layers.Input((self.time_sequence, self.feature_num))
        inputs_mask = layers.Masking(mask_value=0, input_shape=(self.time_sequence, self.feature_num))(inputs)
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
