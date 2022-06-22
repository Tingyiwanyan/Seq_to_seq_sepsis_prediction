from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

semantic_step_global = 6
semantic_positive_sample = 4
unsupervised_cluster_num = 10
latent_dim_global = 100
positive_sample_size = 10
batch_size = 128
unsupervised_neg_size = 5


class projection(keras.layers.Layer):
    def __init__(self, units=unsupervised_cluster_num, input_dim=latent_dim_global):
        super(projection, self).__init__()
        w_init = tf.random_normal_initializer()
        #w_init = tf.keras.initializers.Orthogonal()
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

        self.ave_all = [8.38230435e+01, 9.75000000e+01, 3.69060000e+01, 1.18333333e+02,
                        7.71140148e+01, 5.90000000e+01, 1.81162791e+01, 0.00000000e+00,
                        -2.50000000e-01, 2.43333333e+01, 5.04195804e-01, 7.38666667e+00,
                        4.00504808e+01, 9.60000000e+01, 4.20000000e+01, 1.65000000e+01,
                        7.70000000e+01, 8.35000000e+00, 1.06000000e+02, 9.00000000e-01,
                        1.16250000e+00, 1.25333333e+02, 1.65000000e+00, 2.00000000e+00,
                        3.36666667e+00, 4.08000000e+00, 7.00000000e-01, 3.85000000e+00,
                        3.09000000e+01, 1.05000000e+01, 3.11000000e+01, 1.08333333e+01,
                        2.55875000e+02, 1.93708333e+02]

        self.std_all = [1.40828962e+01, 2.16625304e+00, 5.53108392e-01, 1.66121889e+01,
                        1.08476132e+01, 9.94962122e+00, 3.59186362e+00, 0.00000000e+00,
                        3.89407506e+00, 3.91858658e+00, 2.04595954e-01, 5.93467422e-02,
                        7.72257867e+00, 8.87388075e+00, 5.77276895e+02, 1.79879091e+01,
                        1.36508822e+02, 6.95188900e-01, 5.09788015e+00, 1.43347221e+00,
                        3.75415153e+00, 4.03968485e+01, 1.71418146e+00, 3.15505742e-01,
                        1.17084555e+00, 4.77914796e-01, 3.62933460e+00, 9.91058703e+00,
                        4.60374699e+00, 1.64019340e+00, 1.68795640e+01, 6.23941196e+00,
                        1.75014175e+02, 1.03316340e+02]

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
        self.pre_train_epoch = 7
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

        self.max_train_data = np.max(np.reshape(self.train_data,(self.train_data.shape[0]*self.train_data.shape[1],
                                                                 self.train_data.shape[2])),0)
        self.min_train_data = np.min(np.reshape(self.train_data,(self.train_data.shape[0]*self.train_data.shape[1],
                                                                 self.train_data.shape[2])),0)

        self.index_train = np.array(range(self.train_data.shape[0]))

        self.train_data_range = self.max_train_data - self.min_train_data
        for i in range(self.train_data_range.shape[0]):
            if self.train_data_range[i] == 0:
                self.train_data_range[i] = 1

        self.train_data_norm = (np.reshape(self.train_data,
                                           (self.train_data.shape[0]*self.train_data.shape[1],
                                                                 self.train_data.shape[2])) - self.min_train_data)\
                               /self.train_data_range

        self.train_data_norm = np.reshape(self.train_data_norm,(self.train_data.shape[0],self.train_data.shape[1],
                                                                self.train_data.shape[2]))

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_data_norm, self.train_logit, self.train_on_site_time, self.train_data_norm,self.index_train))  # ,self.train_sofa_score))
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


    def compute_positive_pair(self, z,  global_pull_cohort, global_pull_control, label):
        z = tf.math.l2_normalize(z, axis=-1)
        #z = tf.cast(z,tf.float32)
        global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=-1)
        global_pull_control = tf.math.l2_normalize(global_pull_control, axis=-1)
        #self.check_global_pull_cohort = global_pull_cohort

        random_indices_cohort = np.random.choice(global_pull_cohort.shape[0], size=self.pos_size, replace=False)
        #self.check_random_indices_cohort = random_indices_cohort
        random_indices_control = np.random.choice(global_pull_control.shape[0], size=self.pos_size, replace=False)

        pos_train_cohort = tf.convert_to_tensor([global_pull_cohort[i, :] for i in random_indices_cohort])
        pos_train_control = tf.convert_to_tensor([global_pull_control[i,:] for i in random_indices_control])

        similarity_matrix_cohort = tf.matmul(z, tf.transpose(pos_train_cohort))
        similarity_matrix_control = tf.matmul(z, tf.transpose(pos_train_control))

        pos_cohort_sum = tf.math.exp(similarity_matrix_cohort / self.tau)
        #self.check_pos_cohort_sum = pos_cohort_sum
        pos_control_sum = tf.math.exp(similarity_matrix_control / self.tau)
        #self.check_pos_control_sum = pos_control_sum
        label = tf.cast(label, tf.int64)
        #self.check_label = label

        pos_sum_both = tf.stack((pos_cohort_sum, pos_control_sum), 1)
        #self.check_pos_sum_both = pos_sum_both
        pos_dot_prods_sum = [pos_sum_both[i, 1-label[i]] for i in range(z.shape[0])]
        #self.check_pos_dot_prods_sum = pos_dot_prods_sum

        return tf.stack(pos_dot_prods_sum)


    def compute_negative_paris(self, z, global_pull_cohort, global_pull_control, label):
        z = tf.math.l2_normalize(z, axis=1)

        global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=-1)
        global_pull_control = tf.math.l2_normalize(global_pull_control, axis=-1)

       # global_pull_cohort = tf.reshape(global_pull_cohort, (global_pull_cohort.shape[0] * global_pull_cohort.shape[1],
                                                             #global_pull_cohort.shape[2]))

        #global_pull_control = tf.reshape(global_pull_control,
                                         #(global_pull_control.shape[0] * global_pull_control.shape[1],
                                         # global_pull_control.shape[2]))

        #random_indices_cohort = np.random.choice(global_pull_cohort.shape[0], size=self.neg_size, replace=False)
        # self.check_random_indices_cohort = random_indices_cohort
        #random_indices_control = np.random.choice(global_pull_control.shape[0], size=self.neg_size, replace=False)

        #neg_train_cohort = np.array([global_pull_cohort[i, :] for i in random_indices_cohort])
        #neg_train_control = np.array([global_pull_control[i, :] for i in random_indices_control])
        #z = tf.cast(z,tf.float32)

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

        return tf.stack(negative_dot_prods_sum)

    def info_nce_loss(self, z, global_pull_cohort, global_pull_control, label):
        positive_dot_prod_sum = self.compute_positive_pair(z, global_pull_cohort, global_pull_control, label)
        negative_dot_prod_sum = self.compute_negative_paris(z, global_pull_cohort, global_pull_control, label)

        #self.check_pos_dot_prods_sum = positive_dot_prod_sum
        negative_dot_prod_sum = tf.expand_dims(negative_dot_prod_sum,1)
        #self.check_negative_dot_prods_sum = negative_dot_prod_sum

        denominator = tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator))
        nomalized_prob_log = tf.reduce_sum(nomalized_prob_log,1)
        loss_batch = tf.math.negative(nomalized_prob_log)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss,loss_batch

    def compute_positive_pair_progression(self, z, on_site_extract):
        z = tf.math.l2_normalize(z, axis=-1)
        #global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=-1)
        #global_pull_control = tf.math.l2_normalize(global_pull_control, axis=-1)

        on_site_extract = tf.math.l2_normalize(on_site_extract,axis=-1)
        # self.check_global_pull_cohort = global_pull_cohort
        z = tf.cast(z,tf.float32)
        similarity_matrix = tf.reduce_sum(tf.multiply(z,on_site_extract),1)

        pos_sum = tf.math.exp(similarity_matrix / self.tau)
        self.check_pos_sum = pos_sum

        return pos_sum


    def info_nce_loss_progression(self, semantic_temporal, on_site_temporal, global_pull_cohort, global_pull_control, label):
        positive_dot_prod_sum = self.compute_positive_pair_progression(semantic_temporal, on_site_temporal)
        #positive_dot_prod_sum = self.compute_positive_pair_progression(semantic_temporal, global_pull_cohort, global_pull_control, label)
        negative_dot_prod_sum = self.compute_negative_paris(semantic_temporal, global_pull_cohort, global_pull_control, label)

        # self.check_pos_dot_prods_sum = positive_dot_prod_sum
        negative_dot_prod_sum = tf.expand_dims(negative_dot_prod_sum, 1)
        # self.check_negative_dot_prods_sum = negative_dot_prod_sum

        denominator = tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator))
        nomalized_prob_log = tf.reduce_sum(nomalized_prob_log, 1)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss



    def compute_positive_pairs_prot(self,batch_embedding_whole, projection_basis_whole, semantic_group):
        z = tf.math.l2_normalize(batch_embedding_whole, axis=-1)
        #global_pull_cohort = tf.math.l2_normalize(global_pull_cohort, axis=-1)
        #global_pull_control = tf.math.l2_normalize(global_pull_control, axis=-1)
        # self.check_global_pull_cohort = global_pull_cohort
        z = tf.expand_dims(z,1)
        z = tf.broadcast_to(z,shape=(z.shape[0],self.pos_size,z.shape[2]))

        pos_train_group = []
        for k in range(batch_embedding_whole.shape[0]):
            index = int(projection_basis_whole[k])
            sample_sequence = np.zeros(self.pos_size)
            for j in range(self.pos_size):
                sample_sequence[j] = int(random.randint(0, int(semantic_group[index].shape[0])-1))
            self.check_sample_sequence = sample_sequence
            post_train_group_single = tf.convert_to_tensor([semantic_group[index][int(i),:] for i in sample_sequence])
            pos_train_group.append(post_train_group_single)
        pos_train_group = tf.stack(pos_train_group)
        pos_train_group = tf.math.l2_normalize(pos_train_group, axis=-1)
        self.check_pos_train_group = pos_train_group

        similarity_matrix = tf.multiply(z, pos_train_group)

        pos_sum = tf.reduce_sum(tf.math.exp(similarity_matrix / self.tau),-1)
        self.check_pos_sum = pos_sum

        return pos_sum

    def compute_negative_pairs_prot(self, batch_embedding_whole, semantic_group):
        z = tf.math.l2_normalize(batch_embedding_whole, axis=-1)
        #z = tf.expand_dims(z, 1)

        neg_train_group = []
        for i in range(self.unsupervised_cluster_num):
            if semantic_group[i] != []:
                neg_train_group.append(semantic_group[i])

        neg_train_group = tf.concat(neg_train_group,0)
        self.check_neg_train_group = neg_train_group


        similarity_matrix_neg = tf.matmul(z, tf.transpose(neg_train_group))

        neg_sum = tf.reduce_sum(tf.math.exp(similarity_matrix_neg / self.tau), 1)
        self.check_neg_sum = neg_sum

        return neg_sum


    def unsupervised_prot_loss(self, batch_embedding_whole, projection_basis_whole, semantic_group):
        positive_dot_prod_sum = self.compute_positive_pairs_prot(batch_embedding_whole,
                                                                 projection_basis_whole, semantic_group)
        negative_dot_prod_sum = self.compute_negative_pairs_prot(batch_embedding_whole, semantic_group)
        # self.check_pos_dot_prods_sum = positive_dot_prod_sum
        negative_dot_prod_sum = tf.expand_dims(negative_dot_prod_sum, 1)
        # self.check_negative_dot_prods_sum = negative_dot_prod_sum

        denominator = tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator))
        nomalized_prob_log = tf.reduce_sum(nomalized_prob_log, 1)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss

    def extract_reconstruction_resolution(self,on_site_time,x_batch_origin,resolution):
        self.check_batch_origin = x_batch_origin
        sample_sequence_origin = np.zeros((x_batch_origin.shape[0], resolution,
                                           x_batch_origin.shape[-1]))

        for k in range(x_batch_origin.shape[0]):
            single_on_site = on_site_time[k]
            self.check_single_on_site = single_on_site
            if single_on_site > resolution:
                sample_sequence_origin[k,:,:] = x_batch_origin[k,-resolution:,:]
            if single_on_site == resolution:
                sample_sequence_origin[k,:,:] = x_batch_origin[k,0:resolution,:]
            if single_on_site < resolution:
                sample_sequence_origin[k,int(resolution-single_on_site):,:] = x_batch_origin[k,0:int(single_on_site),:]
                sample_sequence_origin[k,0:int(resolution-single_on_site),:] = x_batch_origin[k,0,:]

        return sample_sequence_origin

    def extract_temporal_semantic(self,x_batch_feature,on_site_time,x_batch_origin,index_batch,indicator):
        #temporal_semantic = \
            #np.zeros((x_batch_feature.shape[0], self.semantic_positive_sample, self.latent_dim))

        #temporal_semantic_origin = \
            #np.zeros((x_batch_feature.shape[0], self.semantic_positive_sample, x_batch_origin.shape[2]))

        temporal_semantic = []
        temporal_semantic_origin = []

        if indicator == 1:
            sample_sequence_batch = np.zeros((x_batch_feature.shape[0], self.semantic_positive_sample+1))
        else:
            sample_sequence_batch = np.zeros((x_batch_feature.shape[0], self.semantic_positive_sample))
        for k in range(x_batch_feature.shape[0]):
            single_on_site = on_site_time[k]
            #sample_sequence_feature = np.zeros((self.semantic_positive_sample, self.latent_dim))
            #sample_sequence_origin = np.zeros((self.semantic_positive_sample, x_batch_origin.shape[2]))
            if single_on_site == 0:
                sample_sequence = np.zeros(self.semantic_positive_sample)
            elif single_on_site < self.semantic_positive_sample:
                sample_sequence = np.zeros(self.semantic_positive_sample)
                for j in range(self.semantic_positive_sample):
                    sample_sequence[j] = random.randint(0, int(single_on_site))
                sample_sequence = np.sort(sample_sequence)
            else:
                if single_on_site > 46:
                    single_on_site = 46
                sample_sequence = np.sort(random.sample(range(0, int(single_on_site)), self.semantic_positive_sample))
            if indicator == 1:
                sample_sequence = np.append(sample_sequence,index_batch[k])
            sample_sequence_batch[k, :] = sample_sequence
            sample_sequence = tf.cast(sample_sequence,tf.int32)
            #self.check_sample_sequence_ = sample_sequence
            if indicator == 1:
                origin_num = self.semantic_positive_sample+1
            else:
                origin_num = self.semantic_positive_sample
            sample_sequence_origin = np.zeros((origin_num,self.tcn_filter_size,
                                               x_batch_origin.shape[-1]))
            for j in range(origin_num):
                #sample_sequence_feature[j, :] = x_batch_feature[k, int(sample_sequence[j]), :]
                if int(sample_sequence[j])==0:
                    sample_sequence_origin[j,:,:] = x_batch_origin[k,0,:]
                elif int(sample_sequence[j])<self.tcn_filter_size:
                    compensate = self.tcn_filter_size - int(sample_sequence[j]) - 1
                    for jj in range(self.tcn_filter_size):
                        if jj < compensate:
                            sample_sequence_origin[j,jj,:] = x_batch_origin[k,0,:]
                        else:
                            sample_sequence_origin[j,jj,:] = x_batch_origin[k,jj-compensate,:]
                else:
                    compensate = int(sample_sequence[j]) - self.tcn_filter_size + 1
                    for jj in range(self.tcn_filter_size):
                        sample_sequence_origin[j,jj,:] = x_batch_origin[k,jj+compensate,:]

                #sample_sequence_origin[j, :,:] = x_batch_origin[k, int(sample_sequence[j]), :]

            sample_sequence_feature = tf.gather(x_batch_feature[k,:,:],indices=sample_sequence)
            #sample_sequence_origin = tf.gather(x_batch_origin[k,:,:],indices=sample_sequence)


            temporal_semantic.append(sample_sequence_feature)
            temporal_semantic_origin.append(sample_sequence_origin)

        self.check_temporal_semantic_previous = temporal_semantic
        temporal_semantic = tf.cast(tf.stack(temporal_semantic),tf.float64)
        self.check_temporal_semantic_after = temporal_semantic
        temporal_semantic_origin = tf.cast(tf.stack(temporal_semantic_origin),tf.float64)
            #temporal_semantic[k, :, :] = sample_sequence_feature
            #temporal_semantic_origin[k, :, :] = sample_sequence_origin
            #self.check_temporal_semantic = temporal_semantic

        return temporal_semantic,sample_sequence_batch, temporal_semantic_origin


    def E_step_initial(self,batch_embedding,projection_basis,batch_origin):
        batch_embedding = tf.math.l2_normalize(batch_embedding, axis=1)

        #projection_basis = projection_basis[0]
        semantic_cluster = []
        projection_basis = tf.expand_dims(projection_basis, 0)
        projection_basis = tf.broadcast_to(projection_basis,
                                           shape=(batch_embedding.shape[0],
                                                  projection_basis.shape[1], projection_basis.shape[2]))

        self.first_check_projection = projection_basis

        batch_embedding_whole = tf.reshape(batch_embedding,(batch_embedding.shape[0]*batch_embedding.shape[1],
                                                            batch_embedding.shape[2]))
        self.check_batch_embedding_whole = batch_embedding_whole

        batch_origin_whole = tf.reshape(batch_origin, (batch_origin.shape[0] * batch_origin.shape[1],
                                                             batch_origin.shape[2]))

        self.check_batch_origin_whole = batch_origin_whole

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
            #semantic_group = []

            for i in range(self.unsupervised_cluster_num):
                semantic_index = np.where(self.projection_basis_whole == i)[0]
                semantic = tf.gather(batch_embedding_whole, semantic_index)
                #semantic_group.append(semantic)
                semantic = tf.reduce_mean(semantic, 0)
                semantic_cluster.append(semantic)

            #semantic_group = tf.stack(semantic_group,0)
            semantic_cluster = tf.stack(semantic_cluster, 0)

            self.check_semantic_cluster = semantic_cluster

            projection_basis = semantic_cluster
            projection_basis_ = semantic_cluster

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

        semantic_group = []
        #semantic_cluster = []

        for i in range(self.unsupervised_cluster_num):
            semantic_index = np.where(self.projection_basis_whole == i)[0]
            semantic = tf.gather(batch_origin_whole, semantic_index)
            semantic_group.append(semantic)
        #self.check_semantic_group = semantic_group
        #semantic_group = tf.stack(semantic_group, 0)

        return max_value_projection, semantic_cluster

    def projection_regularize_loss(self, projection_basis):
        projection_basis = projection_basis[0, :, :]

        similarity_matrix = tf.matmul(projection_basis, tf.transpose(projection_basis))
        mask = tf.linalg.diag(tf.zeros(projection_basis.shape[0]), padding_value=1)

        negative_dot_prods = tf.math.abs(tf.multiply(similarity_matrix, mask))
        projection_regular_loss = tf.reduce_mean(tf.reduce_sum(negative_dot_prods, 1))

        return projection_regular_loss

    def first_lvl_resolution_deconv(self):
        #inputs = layers.Input((1, self.latent_dim))

        #tcn_deconv1 = tf.keras.layers.Conv1DTranspose(self.feature_num, self.tcn_filter_size)

        #output = tcn_deconv1(inputs)

        #return tf.keras.Model(inputs, output, name='tcn_deconv1')

        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((1, self.latent_dim)),
                layers.Dense(
                    self.latent_dim,
                    # use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                ),
                layers.Conv1DTranspose(self.feature_num, self.tcn_filter_size)
                # layers.Input((50)),
            ],
            name="deconv_layer",
        )
        return model

    def one_h_resolution_deconv(self):
        inputs = layers.Input((1, self.latent_dim))

        tcn_deconv1 = tf.keras.layers.Conv1DTranspose(2*self.feature_num, 1)

        tcn_deconv1_ = tf.keras.layers.Conv1DTranspose(self.feature_num, 1)

        output = tcn_deconv1(inputs)
        output = tcn_deconv1_(output)

        return tf.keras.Model(inputs, output, name='tcn_deconv1')

    def whole_signal_deconv(self):
        dilation1 = 1  # 3 hours
        dilation2 = 2  # 7hours
        dilation3 = 4  # 15hours
        dilation4 = 8
        dilation5 = 16

        inputs = layers.Input((1, self.latent_dim))

        kernel_size1 = dilation1*(self.tcn_filter_size-1)+inputs.shape[1]
        kernel_size1 = kernel_size1 - inputs.shape[1] + 1

        tcn_deconv1 = tf.keras.layers.Conv1DTranspose(self.latent_dim, kernel_size1,activation='relu',
                                           dilation_rate=dilation1)
        output_deconv1 = tcn_deconv1(inputs)

        kernal_size2 = dilation2*(self.tcn_filter_size-1)+output_deconv1.shape[1]
        kernal_size2 = kernal_size2 - output_deconv1.shape[1] + 1

        tcn_deconv2 = tf.keras.layers.Conv1DTranspose(self.latent_dim,kernal_size2,activation='relu',
                                           dilation_rate=dilation1)

        #conv2_identity = tf.keras.layers.Conv1D(self.feature_num, 1, activation='sigmoid',
                                                #dilation_rate=1)
        output_deconv2 = tcn_deconv2(output_deconv1)
        #output_deconv2 = conv2_identity(output_deconv2)


        kernal_size3 = dilation3*(self.tcn_filter_size-1)+output_deconv2.shape[1]
        kernal_size3 = kernal_size3 - output_deconv2.shape[1] + 1

        tcn_deconv3 = tf.keras.layers.Conv1DTranspose(self.latent_dim, kernal_size3, activation='relu',
                                                      dilation_rate=dilation1)

        output_deconv3 = tcn_deconv3(output_deconv2)

        kernal_size4 = dilation4*(self.tcn_filter_size-1)+output_deconv3.shape[1]
        kernal_size4 = kernal_size4 - output_deconv3.shape[1] + 1

        tcn_deconv4 = tf.keras.layers.Conv1DTranspose(self.feature_num, kernal_size4, activation='relu',
                                                      dilation_rate=dilation1)
        conv4_identity = tf.keras.layers.Conv1D(self.feature_num, 1, activation='sigmoid',
                                                dilation_rate=1)

        output_deconv4 = tcn_deconv4(output_deconv3)
        output_deconv4 = conv4_identity(output_deconv4)


        return tf.keras.Model(inputs,
                              output_deconv4,
                              name='tcn_deconv')


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
        dilation5 = 16

        """
        define the first tcn layer, dilation=1
        """
        inputs = layers.Input((self.time_sequence,self.feature_num))
        tcn_conv1 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
                                           dilation_rate=dilation1, padding='valid')
        conv1_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu', dilation_rate=1)
        layernorm1 = tf.keras.layers.BatchNormalization()
        padding_1 = (self.tcn_filter_size - 1) * dilation1
        # inputs1 = tf.pad(inputs, tf.constant([[0,0],[1,0],[0,0]]) * padding_1)

        inputs1 = tf.pad(inputs, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_1)
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
                              [inputs, self.outputs4, self.outputs3, self.outputs2, self.outputs1],
                              name='tcn_encoder')


    def tcn_first_level(self):
        """
        Implement tcn encoder
        """
        """
        define dilation for each layer(24 hours)
        """
        dilation1 = 1  #3 hours
        dilation2 = 2  #7 hours

        """
        define the first tcn layer, dilation=1
        """
        inputs = layers.Input((self.time_sequence,self.feature_num))
        tcn_conv1 = tf.keras.layers.Conv1D(self.latent_dim, self.tcn_filter_size, activation='relu',
                                           dilation_rate=dilation1, padding='valid')
        conv1_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu', dilation_rate=1)
        layernorm1 = tf.keras.layers.BatchNormalization()
        padding_1 = (self.tcn_filter_size - 1) * dilation1
        # inputs1 = tf.pad(inputs, tf.constant([[0,0],[1,0],[0,0]]) * padding_1)

        inputs1 = tf.pad(inputs, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_1)
        self.outputs1_first_lvl = tcn_conv1(inputs1)
        self.outputs1_first_lvl = conv1_identity(self.outputs1_first_lvl)
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
        inputs2 = tf.pad(self.outputs1_first_lvl, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_2)
        self.outputs2 = tcn_conv2(inputs2)
        self.outputs2 = conv2_identity(self.outputs2)

        return tf.keras.Model(inputs,
                              [inputs, self.outputs2],
                              name='tcn_encoder_first_lvl')

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

    def translation_layer(self):
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
            name="translation_layer",
        )
        return model

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

    def transition_project_layer(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                # layers.Input((50)),
                layers.Dense(
                    self.latent_dim,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='sigmoid'
                )
            ],
            name="transition_projection",
        )
        return model


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

    def train_cl_first_lvl(self):
        self.tcn_first = self.tcn_first_level()
        # tcn = self.tcn(input)
        self.auc_all = []
        self.loss_track = []
        self.projection_layer = self.project_logit()
        self.transition_layer = self.transition_project_layer()
        self.translation = self.translation_layer()
        self.mseloss = tf.keras.losses.MeanSquaredError()
        self.deconv = self.first_lvl_resolution_deconv()
        # self.model_extractor = tf.keras.Model(input, tcn, name="time_extractor")

        extract_importance_temporal = np.zeros(self.train_data.shape[0])

        extract_compare = np.ones(self.train_data.shape[0])
        for epoch in range(self.pre_train_epoch):
            #input_translation = np.ones(self.latent_dim)
            #if epoch > 0:
                #self.save_embedding_first_lvl(str(epoch))
            print("\nStart of epoch %d" % (epoch,))
            compare = extract_compare- extract_importance_temporal
            self.check_extract_compare = extract_compare
            self.extract_importance_temporal = extract_importance_temporal
            self.num_compare = np.where(compare!=0)[0].shape
            print("num different is")
            print(self.num_compare)
            for i in range(self.train_data.shape[0]):
                extract_compare[i] = extract_importance_temporal[i]
            for step, (x_batch_train, y_batch_train, on_site_time, semantic_origin, index_train) \
                    in enumerate(self.train_dataset):
                #if step > 20:
                    #continue
                self.check_x_batch = x_batch_train
                self.check_on_site_time = on_site_time
                self.check_label = y_batch_train
                self.check_index_train = index_train
                index_compare = [extract_importance_temporal[i] for i in index_train]
                self.check_index_compare = index_compare

                random_indices_cohort = np.random.choice(self.num_cohort, size=x_batch_train.shape[0], replace=False)
                random_indices_control = np.random.choice(self.num_control, size=x_batch_train.shape[0], replace=False)

                x_batch_train_cohort = self.memory_bank_cohort[random_indices_cohort, :, :]
                x_batch_train_control = self.memory_bank_control[random_indices_control, :, :]
                on_site_time_cohort = self.memory_bank_cohort_on_site[random_indices_cohort]
                on_site_time_control = self.memory_bank_control_on_site[random_indices_control]

                with tf.GradientTape() as tape:
                    tcn_temporal_output_first = self.tcn_first(x_batch_train)[1]
                    tcn_temporal_output_first_cohort = self.tcn_first(x_batch_train_cohort)[1]
                    tcn_temporal_output_first_control = self.tcn_first(x_batch_train_control)[1]
                    #translation_vector =

                    self.check_temporal_output_first = tcn_temporal_output_first
                    temporal_semantic, sample_sequence_batch, temporal_semantic_origin = \
                        self.extract_temporal_semantic(tcn_temporal_output_first, on_site_time,
                                                       x_batch_train,index_compare,1)

                    self.check_sample_sequence_batch = sample_sequence_batch

                    temporal_semantic_cohort, sample_sequence_batch_cohort, temporal_semantic_origin_cohort = \
                        self.extract_temporal_semantic(tcn_temporal_output_first_cohort,
                                                       on_site_time_cohort, x_batch_train_cohort,
                                                       index_compare,0)


                    temporal_semantic_control, sample_sequence_batch_control, temporal_semantic_origin_control = \
                        self.extract_temporal_semantic(tcn_temporal_output_first_control,
                                                       on_site_time_control, x_batch_train_control,
                                                       index_compare,0)

                    self.check_temporal_semantic = temporal_semantic
                    self.check_temporal_semantic_cohort = temporal_semantic_cohort
                    self.check_temporal_semantic_origin = temporal_semantic_origin

                    y_batch_train = tf.expand_dims(y_batch_train,axis=1)
                    y_batch_train = tf.broadcast_to(y_batch_train,
                                               shape=(temporal_semantic.shape[0],
                                                      temporal_semantic.shape[1]))
                    y_batch_train = tf.reshape(y_batch_train,(y_batch_train.shape[0]*y_batch_train.shape[1]))
                    #temporal_semantic = tf.squeeze(temporal_semantic)
                    temporal_semantic = tf.reshape(temporal_semantic,
                                                   (temporal_semantic.shape[0]*temporal_semantic.shape[1],
                                                    temporal_semantic.shape[2]))
                    #temporal_semantic_cohort = tf.squeeze(temporal_semantic_cohort)
                    temporal_semantic_cohort = tf.reshape(temporal_semantic_cohort,
                                                   (temporal_semantic_cohort.shape[0] * temporal_semantic_cohort.shape[1],
                                                    temporal_semantic_cohort.shape[2]))
                    #temporal_semantic_control = tf.squeeze(temporal_semantic_control)
                    temporal_semantic_control = tf.reshape(temporal_semantic_control,
                                                   (temporal_semantic_control.shape[0] * temporal_semantic_control.shape[1],
                                                    temporal_semantic_control.shape[2]))

                    temporal_semantic_transit = self.transition_layer(temporal_semantic)
                    temporal_semantic_cohort_transit = self.transition_layer(temporal_semantic_cohort)
                    temporal_semantic_control_transit = self.transition_layer(temporal_semantic_control)

                    temporal_semantic = tf.cast(tf.math.l2_normalize(temporal_semantic_transit,axis=-1),tf.float64)
                    temporal_semantic_cohort = tf.cast(tf.math.l2_normalize(temporal_semantic_cohort_transit,axis=-1),tf.float64)
                    temporal_semantic_control = tf.cast(tf.math.l2_normalize(temporal_semantic_control_transit,axis=-1),tf.float64)

                    temporal_semantic_ = tf.expand_dims(temporal_semantic,1)
                    temporal_semantic_reconstruct = self.deconv(temporal_semantic_)
                    self.check_temporal_semantic_origin_ = temporal_semantic_origin
                    #temporal_semantic_origin = tf.squeeze(temporal_semantic_origin)
                    temporal_semantic_origin = tf.reshape(temporal_semantic_origin,
                                                          (temporal_semantic_origin.shape[0]*
                                                           temporal_semantic_origin.shape[1],
                                                           temporal_semantic_origin.shape[2],
                                                           temporal_semantic_origin.shape[3]))



                    #temporal_semantic_reconstruct = tf.cast(temporal_semantic_reconstruct,tf.float64)
                    #temporal_semantic_origin = tf.cast(temporal_semantic_origin,tf.float64)
                    #temporal_semantic = tf.cast(temporal_semantic,tf.float32)

                    #mse_loss = self.mseloss(temporal_semantic_reconstruct,temporal_semantic_origin)


                    cl_loss_temporal,cl_loss_batch = self.info_nce_loss(temporal_semantic,temporal_semantic_cohort,
                                                          temporal_semantic_control, y_batch_train)



                    self.check_cl_loss_temporal = cl_loss_temporal
                    self.check_cl_loss_batch = cl_loss_batch

                    select_cl_loss = tf.reshape(cl_loss_batch,(x_batch_train.shape[0],self.semantic_positive_sample+1))
                    select_cl_loss_offset = tf.concat([tf.zeros((x_batch_train.shape[0],self.semantic_positive_sample)),
                                                       tf.expand_dims(0.07*tf.ones(x_batch_train.shape[0]),1)],1)
                    self.check_select_cl_loss_offset = select_cl_loss_offset
                    self.check_select_cl_loss = select_cl_loss

                    select_cl_loss_ = tf.math.subtract(select_cl_loss,tf.cast(select_cl_loss_offset,tf.float64))
                    self.check_select_cl_loss_minus = select_cl_loss_
                    if epoch>0:
                        select_index_max = tf.math.argmin(select_cl_loss_,1)
                    else:
                        select_index_max = tf.math.argmin(select_cl_loss, 1)

                    update_inportant_temporal = sample_sequence_batch

                    range_batch = tf.convert_to_tensor(range(select_cl_loss.shape[0]))
                    select_index_max = tf.stack((tf.cast(range_batch,tf.int64),select_index_max),axis=1)

                    self.check_select_index_max = select_index_max

                    cl_loss_temporal_final = tf.gather_nd(select_cl_loss,select_index_max)
                    self.check_cl_loss_temporal_final = cl_loss_temporal_final

                    update_important_temporal = tf.gather_nd(sample_sequence_batch,select_index_max)
                    self.check_update_important_temporal = update_important_temporal
                    for j in range(index_train.shape[0]):
                        index = index_train[j]
                        extract_importance_temporal[index] = update_important_temporal[j]

                    cl_loss_temporal_mean = tf.reduce_mean(cl_loss_temporal_final)
                    #if epoch < 2:
                    #if epoch == 0 or epoch % 2 == 0:
                    loss = cl_loss_temporal_mean

                    #if epoch % 2 == 1:
                        #loss =progression_loss

                #if epoch == 0 or epoch % 2 == 0:
                gradients = \
                    tape.gradient(loss,
                                  self.tcn_first.trainable_variables+self.transition_layer.trainable_variables)
                                  #+self.deconv.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients,
                                              self.tcn_first.trainable_variables
                                              +self.transition_layer.trainable_variables))
                #if epoch % 2 == 1:
                 #   gradients = \
                  #      tape.gradient(loss, self.transition_layer.trainable_variables)
                   # optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                    #optimizer.apply_gradients(zip(gradients, self.transition_layer.trainable_variables))

                if step % 20 == 0:
                    #if epoch == 0 or epoch % 2 == 0:
                    print("Training cl_loss_temporal(for one batch) at step %d: %.4f"
                          % (step, float(cl_loss_temporal)))
                    #print("Training mse_loss(for one batch) at step %d: %.4f"
                     #     % (step, float(mse_loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))
                    #compare = extract_compare - extract_importance_temporal
                    #num_compare = np.where(compare != 0)[0].shape
                    #print("num different is")
                    #print(num_compare)

                    self.loss_track.append(loss)


    def train_cl(self):
        # input = layers.Input((self.time_sequence, self.feature_num))
        self.tcn = self.tcn_encoder_second_last_level()
        self.tcn_first = self.tcn_first_level()
        # tcn = self.tcn(input)
        self.auc_all = []
        self.loss_track = []
        self.projection_layer = self.project_logit()
        self.transition_layer = self.transition_project_layer()
        self.translation = self.translation_layer()
        self.mseloss = tf.keras.losses.MeanSquaredError()
        self.deconv = self.first_lvl_resolution_deconv()
        self.deconv_whole = self.whole_signal_deconv()
        # self.model_extractor = tf.keras.Model(input, tcn, name="time_extractor")

        for epoch in range(self.pre_train_epoch):
            input_translation = np.ones(self.latent_dim)
            if epoch > 0:
                self.save_embedding(str(epoch))
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

            """
            tcn_temporal_output_whole = self.tcn(self.train_data)
            output_1h_resolution_whole = tcn_temporal_output_whole[5]
            # on_site_extract_whole = [last_layer_output[i, np.abs(int(self.train_on_site_time[i] - 1)), :] for i in
            # range(self.train_on_site_time.shape[0])]

            temporal_semantic_whole, sample_sequence_batch_whole, temporal_semantic_origin_whole = \
                self.extract_temporal_semantic(output_1h_resolution_whole, self.train_on_site_time, self.train_data)

            self.check_temporal_semantic_whole = temporal_semantic_whole
            self.check_sample_sequence_batch_whole = sample_sequence_batch_whole
            self.check_temporal_semantic_origin_whole = temporal_semantic_origin_whole

            order_input_total_init, projection_cluster = \
                self.E_step_initial(temporal_semantic_whole, self.init_projection_basis, temporal_semantic_origin_whole)

            self.check_order_input_total_init = order_input_total_init
            self.check_projection_group_total = projection_cluster
            """

            for step, (x_batch_train, y_batch_train, on_site_time, semantic_origin) in enumerate(self.train_dataset):
                self.check_x_batch = x_batch_train
                self.check_on_site_time = on_site_time
                self.check_label = y_batch_train

                self.check_semantic_origin = semantic_origin
                identity_input_translation = np.zeros((x_batch_train.shape[0],self.latent_dim))

                random_indices_cohort = np.random.choice(self.num_cohort, size=x_batch_train.shape[0], replace=False)
                random_indices_control = np.random.choice(self.num_control, size=x_batch_train.shape[0], replace=False)

                x_batch_train_cohort = self.memory_bank_cohort[random_indices_cohort, :, :]
                x_batch_train_control = self.memory_bank_control[random_indices_control, :, :]
                on_site_time_cohort = self.memory_bank_cohort_on_site[random_indices_cohort]
                on_site_time_control = self.memory_bank_control_on_site[random_indices_control]

                batch_resolution_reconstruct = self.extract_reconstruction_resolution(on_site_time,semantic_origin,31)
                self.check_batch_resolution_reconstruct = batch_resolution_reconstruct


                with tf.GradientTape() as tape:
                    tcn_temporal_output = self.tcn(x_batch_train)
                    tcn_temporal_output_cohort = self.tcn(x_batch_train_cohort)
                    tcn_temporal_output_control = self.tcn(x_batch_train_control)

                    self.check_output = tcn_temporal_output
                    last_layer_output = tcn_temporal_output[1]
                    last_layer_output_cohort = tcn_temporal_output_cohort[1]
                    last_layer_output_control = tcn_temporal_output_control[1]
                    on_site_extract = [last_layer_output[i, np.abs(int(on_site_time[i] - 1)), :] for i in
                                       range(on_site_time.shape[0])]
                    on_site_extract_array = tf.stack(on_site_extract)

                    on_site_extract_cohort = [last_layer_output_cohort[i, np.abs(int(on_site_time_cohort[i]-1)),:] for i
                                              in range(on_site_time_cohort.shape[0])]
                    on_site_extract_array_cohort = tf.stack(on_site_extract_cohort)

                    on_site_extract_control = [last_layer_output_control[i, np.abs(int(on_site_time_control[i]-1)),:] for i
                                               in range(on_site_time_control.shape[0])]
                    on_site_extract_array_control = tf.stack(on_site_extract_control)

                    temporal_semantic_ = tf.expand_dims(on_site_extract_array, 1)
                    temporal_semantic_reconstruct = self.deconv_whole(temporal_semantic_)
                    self.check_temporal_semantic_reconstruct = temporal_semantic_reconstruct


                    temporal_semantic_reconstruct = tf.cast(temporal_semantic_reconstruct,tf.float64)
                    batch_resolution_reconstruct = tf.cast(batch_resolution_reconstruct,tf.float64)

                    mse_loss = tf.cast(self.mseloss(temporal_semantic_reconstruct,batch_resolution_reconstruct),
                                       tf.float64)

                    cl_loss = tf.cast(self.info_nce_loss(on_site_extract_array,on_site_extract_array_cohort,
                                              on_site_extract_array_control, y_batch_train),tf.float64)

                    #cl_loss_temporal = self.info_nce_loss(temporal_semantic_transit,temporal_semantic_cohort_transit,
                                                          #temporal_semantic_control_transit, y_batch_train)

                    #progression_loss = self.info_nce_loss_progression(temporal_semantic_transit,on_site_extract_array,
                                                                      #on_site_extract_array_cohort,
                                                                      #on_site_extract_array_control, y_batch_train)
                    #if epoch < 2:
                    #if epoch == 0 or epoch % 2 == 0:
                    loss = cl_loss+mse_loss

                    #if epoch % 2 == 1:
                        #loss =progression_loss

                #if epoch == 0 or epoch % 2 == 0:
                gradients = \
                    tape.gradient(loss,
                                  self.tcn.trainable_variables+self.deconv_whole.trainable_variables)
                                  #+self.deconv.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients,
                                              self.tcn.trainable_variables+self.deconv_whole.trainable_variables))
                #if epoch % 2 == 1:
                 #   gradients = \
                  #      tape.gradient(loss, self.transition_layer.trainable_variables)
                   # optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                    #optimizer.apply_gradients(zip(gradients, self.transition_layer.trainable_variables))

                if step % 20 == 0:
                    #if epoch == 0 or epoch % 2 == 0:
                    print("Training cl_loss(for one batch) at step %d: %.4f"
                          % (step, float(cl_loss)))
                    #print("Training cl_loss_temporal(for one batch) at step %d: %.4f"
                     #     % (step, float(cl_loss_temporal)))
                    #print("Training progression_loss(for one batch) at step %d: %.4f"
                      #    % (step, float(progression_loss)))
                    print("Training mse_loss(for one batch) at step %d: %.4f"
                          % (step, float(mse_loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)


    def reconstruct_signal_first_lvl(self):
        temporal_cohort_1_lvl_resolution = self.tcn_first(self.memory_bank_cohort)[1]
        temporal_control_1_lvl_resolution = self.tcn_first(self.memory_bank_control)[1]
        on_site_time_cohort = self.memory_bank_cohort_on_site
        on_site_time_control = self.memory_bank_control_on_site

        self.temporal_semantic_cohort, sample_sequence_batch_cohort, temporal_semantic_origin_cohort = \
            self.extract_temporal_semantic(temporal_cohort_1_lvl_resolution,
                                           on_site_time_cohort, self.memory_bank_cohort)

        self.temporal_semantic_control, sample_sequence_batch_control, temporal_semantic_origin_control = \
            self.extract_temporal_semantic(temporal_control_1_lvl_resolution,
                                           on_site_time_control, self.memory_bank_control)

        self.temporal_semantic_cohort = tf.math.l2_normalize(tf.squeeze(self.temporal_semantic_cohort),axis=-1)
        self.temporal_semantic_control = tf.math.l2_normalize(tf.squeeze(self.temporal_semantic_control),axis=-1)

        with open('temporal_semantic_embedding_cohort.npy','wb') as f:
            np.save(f,self.temporal_semantic_cohort)
        with open('temporal_semantic_embedding_control.npy','wb') as f:
            np.save(f,self.temporal_semantic_control)


    def reconstruct_signal(self):
        temporal_cohort_on_site_ = self.tcn(self.memory_bank_cohort)[1]
        temporal_cohort_1_lvl_resolution = self.tcn(self.memory_bank_cohort)[4]
        temporal_control_on_site_ = self.tcn(self.memory_bank_control)[1]
        temporal_control_1_lvl_resolution = self.tcn(self.memory_bank_control)[4]
        on_site_time_cohort = self.memory_bank_cohort_on_site
        on_site_time_control = self.memory_bank_control_on_site
        temporal_cohort_on_site = [temporal_cohort_on_site_[i, np.abs(int(on_site_time_cohort[i] - 1)), :] for i
         in range(on_site_time_cohort.shape[0])]
        self.temporal_cohort_on_site = tf.math.l2_normalize(tf.stack(temporal_cohort_on_site),axis=-1)

        temporal_control_on_site = [temporal_control_on_site_[i, np.abs(int(on_site_time_control[i] - 1)), :] for i
                                   in range(on_site_time_control.shape[0])]
        self.temporal_control_on_site = tf.math.l2_normalize(tf.stack(temporal_control_on_site),axis=-1)

        with open('temporal_semantic_embedding_cohort.npy','wb') as f:
            np.save(f,self.temporal_cohort_on_site)

        with open('temporal_semantic_embedding_control.npy','wb') as f:
            np.save(f,self.temporal_control_on_site)

        self.center_temporal_cohort_on_site = tf.reduce_mean(self.temporal_cohort_on_site,0)
        self.center_temporal_control_on_site = tf.reduce_mean(self.temporal_control_on_site,0)

        self.reconstruct_cohort = self.deconv_whole(tf.expand_dims(tf.expand_dims(self.center_temporal_cohort_on_site,0),0))[0]
        self.reconstruct_control = self.deconv_whole(tf.expand_dims(tf.expand_dims(self.center_temporal_control_on_site,0),0))[0]

        self.reconstruct_cohort = self.reconstruct_cohort*self.train_data_range+self.min_train_data
        self.reconstruct_cohort = self.reconstruct_cohort*self.std_all+self.ave_all

        self.reconstruct_control = self.reconstruct_control*self.train_data_range+self.min_train_data
        self.reconstruct_control = self.reconstruct_control*self.std_all+self.ave_all

        self.df_cohort = pd.DataFrame(np.transpose(np.array(self.reconstruct_cohort)))
        self.df_control = pd.DataFrame(np.transpose(np.array(self.reconstruct_control)))
        self.df_cohort.to_csv('df_cohort.csv',seq='\t')
        self.df_control.to_csv('df_control.csv',seq='\t')

        self.temporal_semantic_cohort, sample_sequence_batch_cohort, temporal_semantic_origin_cohort = \
            self.extract_temporal_semantic(temporal_cohort_1_lvl_resolution,
                                           on_site_time_cohort, self.memory_bank_cohort)

        self.temporal_semantic_control, sample_sequence_batch_control, temporal_semantic_origin_control = \
            self.extract_temporal_semantic(temporal_control_1_lvl_resolution,
                                           on_site_time_control, self.memory_bank_control)

        self.temporal_semantic_cohort = tf.math.l2_normalize(tf.squeeze(self.temporal_semantic_cohort),axis=-1)
        self.temporal_semantic_control = tf.math.l2_normalize(tf.squeeze(self.temporal_semantic_control),axis=-1)



    def vis_embedding(self):
        plt.cla()
        train = self.tcn(self.train_data)[1]
        train_embedding = [train[i,np.abs(int(self.train_on_site_time[i] - 1)), :] for i in
                                       range(self.train_on_site_time.shape[0])]
        self.train_embedding = train_embedding

        CL_k = TSNE(n_components=2).fit_transform(np.array(train_embedding)[0:5000,:])
        for i in range(5000):
            if self.train_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.train_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        plt.show()


    def vis_embedding_tsl(self):
        plt.cla()
        output_1h_resolution_whole = self.tcn(self.train_data)[4]
        temporal_semantic_whole, sample_sequence_batch_whole, temporal_semantic_origin_whole = \
            self.extract_temporal_semantic(output_1h_resolution_whole, self.train_on_site_time, self.train_data)

        temporal_semantic_whole = tf.squeeze(temporal_semantic_whole)
        self.check_vis_temporal = tf.squeeze(temporal_semantic_whole)


        CL_k = TSNE(n_components=2).fit_transform(temporal_semantic_whole[0:5000,:])

        for i in range(5000):
            if self.train_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.train_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        plt.show()

    def save_embedding_first_lvl(self,id):
        output_1h_resolution_whole = self.tcn_first(self.train_data)[1]
        #train_embedding = [train[i, np.abs(int(self.train_on_site_time[i] - 1)), :] for i in
                           #range(self.train_on_site_time.shape[0])]

        #output_1h_resolution_whole = self.tcn(self.train_data)[4]
        temporal_semantic_whole, sample_sequence_batch_whole, temporal_semantic_origin_whole = \
            self.extract_temporal_semantic(output_1h_resolution_whole, self.train_on_site_time, self.train_data,
                                           self.index_train,0)

        temporal_semantic_whole = tf.squeeze(temporal_semantic_whole)
        #temporal_semantic_whole_transit = self.transition_layer(temporal_semantic_whole)
        self.check_vis_temporal = tf.squeeze(temporal_semantic_whole)

        #train_embedding = np.array(train_embedding)
        #self.train_embedding = train_embedding

       # with open('on_site_embedding'+id+'.npy', 'wb') as f:
            #np.save(f, train_embedding)

        with open('on_site_logit'+id+'.npy','wb') as f:
            np.save(f,self.train_logit)

        with open('temporal_semantic_embedding'+id+'.npy','wb') as f:
            np.save(f,temporal_semantic_whole)

    def save_embedding(self,id):
        train = self.tcn(self.train_data)[1]
        train_embedding = [train[i, np.abs(int(self.train_on_site_time[i] - 1)), :] for i in
                           range(self.train_on_site_time.shape[0])]

        output_1h_resolution_whole = self.tcn(self.train_data)[4]
        temporal_semantic_whole, sample_sequence_batch_whole, temporal_semantic_origin_whole = \
            self.extract_temporal_semantic(output_1h_resolution_whole, self.train_on_site_time, self.train_data)

        temporal_semantic_whole = tf.squeeze(temporal_semantic_whole)
        temporal_semantic_whole_transit = self.transition_layer(temporal_semantic_whole)
        self.check_vis_temporal = tf.squeeze(temporal_semantic_whole)

        train_embedding = np.array(train_embedding)
        self.train_embedding = train_embedding

        with open('on_site_embedding'+id+'.npy', 'wb') as f:
            np.save(f, train_embedding)

        with open('on_site_logit'+id+'.npy','wb') as f:
            np.save(f,self.train_logit)


        #with open('temporal_semantic_embedding'+id+'.npy','wb') as f:
         #   np.save(f,temporal_semantic_whole_transit)

    def load_embedding(self):
        with open('on_site_embedding.npy', 'rb') as f:
            self.on_site_embedding = np.load(f)
        with open('on_site_logit.npy','rb') as f:
            self.on_site_logit = np.load(f)
        with open('temporal_semantic_embedding.npy','rb') as f:
            self.temporal_semantic_embedding = np.load(f)


    def vis_embedding_load(self):

        CL_k = TSNE(n_components=2).fit_transform(np.array(self.on_site_embedding)[0:5000,:])
        for i in range(5000):
            if self.on_site_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.on_site_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        plt.show()


    def vis_embedding_tsl_load(self):

        CL_k = TSNE(n_components=2).fit_transform(self.temporal_semantic_embedding[0:5000,:])

        for i in range(5000):
            if self.train_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.train_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        plt.show()



    def semantic_combination_recommandation(self):

        train_cohort = self.tcn(self.memory_bank_cohort)[1]
        train_embedding_cohort = [train_cohort[i, np.abs(int(self.memory_bank_cohort_on_site[i] - 1)), :] for i in
                           range(self.memory_bank_cohort_on_site.shape[0])]

        train_embedding_cohort = tf.math.l2_normalize(tf.stack(train_embedding_cohort))
        self.check_train_embedding_cohort = train_embedding_cohort

        train_control = self.tcn(self.memory_bank_control)[1]
        train_embedding_control = [train_control[i, np.abs(int(self.memory_bank_control_on_site[i] - 1)), :] for i in
                                  range(self.memory_bank_control_on_site.shape[0])]

        train_embedding_control = tf.stack(train_embedding_control)



        position_encode_whole = np.zeros((self.semantic_positive_sample, self.latent_dim))
        for i in range(self.semantic_positive_sample):
            position_encode_whole[i, :] = self.position_encoding(i)

        position_encode_whole = tf.expand_dims(position_encode_whole, 0)

        tcn_temporal_output_cohort = self.tcn(self.memory_bank_cohort)[4]
        tcn_temporal_output_control = self.tcn(self.memory_bank_control)[4]

        temporal_semantic_cohort, sample_sequence_batch_cohort, temporal_semantic_origin_cohort = \
            self.extract_temporal_semantic(tcn_temporal_output_cohort,
                                           self.memory_bank_cohort_on_site, self.memory_bank_cohort)

        temporal_semantic_control, sample_sequence_batch_control, temporal_semantic_origin_control = \
            self.extract_temporal_semantic(tcn_temporal_output_control,
                                           self.memory_bank_control_on_site, self.memory_bank_control)

        self.check_sample_sequence_cohort = sample_sequence_batch_cohort

        self.check_sample_sequence_control = sample_sequence_batch_control


        temporal_semantic_position_cohort = \
            tf.math.add(temporal_semantic_cohort, position_encode_whole)

        batch_semantic_temporal_feature_seperate_cohort = self.position_project(temporal_semantic_position_cohort)

        self.check_batch_semantic_cohort_separate = batch_semantic_temporal_feature_seperate_cohort

        batch_semantic_temporal_feature_cohort = \
            tf.math.l2_normalize(tf.reduce_sum(batch_semantic_temporal_feature_seperate_cohort, 1))

        self.check_batch_semantic_temporal_feature_cohort = batch_semantic_temporal_feature_cohort

        temporal_semantic_position_control = \
            tf.math.add(temporal_semantic_control, position_encode_whole)

        batch_semantic_temporal_feature_seperate_control = self.position_project(temporal_semantic_position_control)

        batch_semantic_temporal_feature_control = \
            tf.math.l2_normalize(tf.reduce_sum(batch_semantic_temporal_feature_seperate_control, 1))

        self.check_batch_semantic_temporal_feature_control = batch_semantic_temporal_feature_control




        self.check_on_site_extract_cohort = tf.math.l2_normalize(tf.stack(train_embedding_cohort),axis=-1)
        self.check_on_site_extract_control = tf.math.l2_normalize(tf.stack(train_embedding_control),axis=-1)

        self.score_cohort = tf.matmul(batch_semantic_temporal_feature_cohort,
                                      tf.transpose(self.check_on_site_extract_cohort))
        self.score_control = tf.matmul(batch_semantic_temporal_feature_cohort,
                                       tf.transpose(self.check_on_site_extract_control))

        self.score_cohort = tf.reduce_mean(tf.reduce_mean(self.score_cohort,1))
        self.score_control = tf.reduce_mean(tf.reduce_mean(self.score_control,1))



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
