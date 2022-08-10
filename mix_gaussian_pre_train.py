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
from tensorflow.keras import regularizers
from sklearn.linear_model import LogisticRegression
import tensorflow_addons as tfa
import umap
from sklearn.utils import shuffle
import seaborn as sns

semantic_step_global = 6
semantic_positive_sample = 3
unsupervised_cluster_num = 4
latent_dim_global = 100
positive_sample_size = 10
batch_size = 128
unsupervised_neg_size = 3
reconstruct_resolution = 7
feature_num = 34

class protatype_ehr():
    def __init__(self):
        #self.read_d = read_d
        #self.train_data = read_d.train_data
        #self.test_data = read_d.test_data
        #self.validate_data = read_d.val_data
        #self.length_train = len(self.train_data)
        #self.length_test = len(self.test_data)
        #self.length_val = len(self.validate_data)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.ave_all = [8.38230435e+01, 9.75000000e+01, 3.69060000e+01, 1.18333333e+02,
                        7.71140148e+01, 5.90000000e+01, 1.81162791e+01, 0.00000000e+00,
                        2.50000000e-01, 2.43333333e+01, 5.04195804e-01, 7.38666667e+00,
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
        self.reconstruct_resolution = reconstruct_resolution
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 20
        self.feature_num = 34
        self.pre_train_epoch = 6
        self.latent_dim = latent_dim_global
        self.tau = 1
        self.time_sequence = 48#self.read_d.time_sequence
        self.tcn_filter_size = 3
        self.semantic_time_step = semantic_step_global
        self.unsupervised_cluster_num = unsupervised_cluster_num
        self.unsupervised_neg_size = unsupervised_neg_size
        self.start_sampling_index = 5
        self.sampling_interval = 5
        self.converge_threshold_E = 20
        self.semantic_positive_sample = semantic_positive_sample
        self.max_value_projection = np.zeros((self.batch_size, self.semantic_time_step))
        self.basis_input = np.ones((self.unsupervised_cluster_num, self.latent_dim))

        self.num_cluster_cohort = 30
        self.num_cluster_control = 30

        self.create_memory_bank()
        self.length_train = len(self.train_data)

        """
        initialize orthogonal projection basis
        """
        self.initializer_basis = tf.keras.initializers.Orthogonal(seed=4)
        self.offset_mu = self.initializer_basis(shape=(1,self.latent_dim))
        self.offset_mu = np.array(self.offset_mu)[0]*0.1

        self.initializer_center1 = tf.keras.initializers.Orthogonal(seed=6)
        self.center_1 = self.initializer_center1(shape=(1,self.latent_dim))
        self.center_1 = np.array(self.center_1)[0]

        self.center_2 = self.center_1 + self.offset_mu

        self.init_projection_basis = tf.cast(tf.Variable(
            self.initializer_basis(shape=(self.unsupervised_cluster_num, self.latent_dim))),tf.float64)

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
        #file_path = '/prj0129/tiw4003/Interpolate_data/'
        #file_path = '/Users/tingyi/Downloads/Interpolate_data/'
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

        #self.train_data = np.expand_dims(self.train_data,axis=3)
        #self.val_data = np.expand_dims(self.val_data,axis=3)
        self.ave_train = np.array(tf.reduce_mean(self.train_data,axis=1))
        self.ave_val = np.array(tf.reduce_mean(self.val_data,axis=1))
        """
        ave_train = np.zeros((self.train_data.shape[0],self.train_data.shape[-1]))
        for i in range(self.train_data.shape[0]):
            index = int(self.train_on_site_time[i])
            ave_train_ = self.train_data[i,index-1,:]
            ave_train[i,:] = ave_train_
        self.ave_train = ave_train
        ave_val = np.zeros((self.val_data.shape[0], self.val_data.shape[-1]))
        for i in range(self.val_data.shape[0]):
            index = int(self.val_on_site_time[i])
            ave_val_ = self.val_data[i,index-1,:]
            ave_val[i,:] = ave_val_
        self.ave_val = ave_val
        """
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_data, self.train_logit, self.train_on_site_time))  # ,self.train_sofa_score))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024,seed=4).batch(self.batch_size)
        cohort_index = np.where(self.train_logit == 1)[0]
        control_index = np.where(self.train_logit == 0)[0]
        self.memory_bank_cohort = self.train_data[cohort_index, :]
        self.memory_bank_control = self.train_data[control_index, :]
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

        return loss#loss_batch

    def compute_positive_pair_local(self, z, cluster_basis, value_projection):
        z = tf.math.l2_normalize(z, axis=-1)
        cluster_basis = tf.math.l2_normalize(cluster_basis, axis=-1)
        #z = tf.cast(z,tf.float32)
        positive_projection = []
        [positive_projection.append(cluster_basis[int(i)]) for i in value_projection]
        positive_projection = tf.stack(positive_projection)
        self.check_positive_projection = positive_projection

        similarity_score = tf.math.exp(tf.reduce_sum(tf.multiply(tf.cast(z,tf.float64),
                                                     tf.cast(positive_projection,tf.float64))/self.tau,-1))

        #similarity_score = tf.math.exp(tf.norm(tf.math.subtract(tf.cast(z,tf.float64),
                                                                #tf.cast(positive_projection,tf.float64)),
                                              # ord='euclidean',axis=-1)/self.tau)


        return similarity_score

    def compute_negative_paris_local(self, z, cluster_basis):
        z = tf.math.l2_normalize(z, axis=-1)

        #z = tf.expand_dims(z,1)
        """
        z = tf.expand_dims(z, 1)
        z = tf.broadcast_to(z, [z.shape[0],self.unsupervised_cluster_num,z.shape[-1]])
        cluster_basis = tf.expand_dims(cluster_basis, 0)
        cluster_basis = tf.broadcast_to(cluster_basis,
                                           shape=(z.shape[0],
                                                  cluster_basis.shape[1], cluster_basis.shape[2]))
        """
        cluster_basis = tf.math.l2_normalize(cluster_basis, axis=-1)

        similarity_matrix = tf.matmul(tf.cast(z,tf.float64), tf.cast(tf.transpose(cluster_basis),tf.float64))

        #similarity_matrix = tf.math.exp(tf.norm(tf.math.subtract(tf.cast(z,tf.float64),
                                                                 #tf.cast(cluster_basis,tf.float64)),
                                                #ord='euclidean', axis=-1)/self.tau)

        similarity_score = tf.reduce_sum(tf.math.exp(similarity_matrix), -1)

        return similarity_score

    def info_nce_loss_local(self, z, cluster_basis, value_projection):
        positive_dot_prod_sum = self.compute_positive_pair_local(z, cluster_basis, value_projection)
        negative_dot_prod_sum = self.compute_negative_paris_local(z, cluster_basis)
        self.check_positive_dot_prod_sum = positive_dot_prod_sum
        self.check_negative_dot_prod_sum = negative_dot_prod_sum

        #self.check_pos_dot_prods_sum = positive_dot_prod_sum
        #negative_dot_prod_sum = tf.expand_dims(negative_dot_prod_sum,1)
        #self.check_negative_dot_prods_sum = negative_dot_prod_sum

        denominator = negative_dot_prod_sum#tf.math.add(positive_dot_prod_sum, negative_dot_prod_sum)
        nomalized_prob_log = tf.math.log(tf.math.divide(positive_dot_prod_sum, denominator))
        #nomalized_prob_log = tf.reduce_sum(nomalized_prob_log,1)
        loss_batch = tf.math.negative(nomalized_prob_log)
        loss = tf.math.negative(tf.reduce_mean(nomalized_prob_log, 0))

        return loss#loss_batch


    def E_step_initial(self,batch_embedding):
        batch_embedding = tf.cast(tf.math.l2_normalize(batch_embedding, axis=1),tf.float64)

        self.check_batch_embedding_init = batch_embedding
        #projection_basis = projection_basis[0]
        semantic_cluster = []
        random_indices_basis = np.random.choice(batch_embedding.shape[0],
                                                size=self.unsupervised_cluster_num, replace=False)

        self.check_random_indices_basis = random_indices_basis


        #x_batch_train_cohort = self.memory_bank_cohort[random_indices_cohort, :, :]
        projection_basis = [batch_embedding[i] for i in random_indices_basis]
        projection_basis = tf.stack(projection_basis)

        #projection_basis = batch_embedding[random_indices_basis,:]
        self.check_projection_basis = projection_basis

        projection_basis = tf.expand_dims(projection_basis, 0)
        projection_basis = tf.broadcast_to(projection_basis,
                                           shape=(batch_embedding.shape[0],
                                                  projection_basis.shape[1], projection_basis.shape[2]))

        self.first_check_projection = projection_basis

        batch_embedding_whole = batch_embedding#tf.reshape(batch_embedding,(batch_embedding.shape[0]*batch_embedding.shape[1],
                                                            #batch_embedding.shape[2]))

        self.check_batch_embedding_whole = batch_embedding_whole


        batch_embedding = tf.expand_dims(batch_embedding, 1)
        batch_embedding = tf.broadcast_to(batch_embedding, [batch_embedding.shape[0],
                                                            self.unsupervised_cluster_num,
                                                            batch_embedding.shape[-1]])

        self.check_batch_embedding_E = batch_embedding

        check_converge = 100 * np.ones(batch_embedding.shape[0])

        self.check_check_converge = check_converge

        check_converge_num = 1000
        self.check_converge_num = check_converge_num

        max_value_projection = 0

        while(check_converge_num > self.converge_threshold_E):
            print(check_converge_num)
            #basis = tf.math.l2_normalize(projection_basis, axis=-1)
            #self.check_basis = basis

            basis = tf.cast(projection_basis,tf.float64)

            #projection = tf.multiply(batch_embedding, basis)
            projection = tf.math.subtract(batch_embedding,basis)
            projection = tf.norm(projection,ord='euclidean',axis=-1)
            #projection = tf.reduce_sum(projection, 2)

            self.check_projection_E = projection
            max_value_projection = np.argmin(projection, axis=1)
            self.check_max_value_projection = max_value_projection

            projection_basis_whole = max_value_projection #tf.reshape(max_value_projection,
                                                #(max_value_projection.shape[0]*max_value_projection.shape[1]))

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
            """
            normalize the basis
            """
            semantic_cluster = tf.math.l2_normalize(semantic_cluster,axis=-1)

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
        #semantic_cluster = []

        return max_value_projection, semantic_cluster

    def E_find_cluster(self,batch_embedding,projection_basis_cohort,projection_basis_control,batch_label):

        index_cohort = np.where(batch_label == 1)[0]

        batch_embedding_cohort = tf.gather(batch_embedding, index_cohort)

        batch_embedding_cohort_return = batch_embedding_cohort

        #self.check_batch_embedding_cohort = batch_embedding_cohort

        # batch_label_control = tf.cast(tf.expand_dims(np.where(batch_label>0.5,0,1)),tf.float64)
        # batch_embedding_control = tf.multiply(batch_embedding,batch_label_control)
        index_control = np.where(batch_label == 0)[0]

        batch_embedding_control = tf.gather(batch_embedding, index_control)
        batch_embedding_control_return = batch_embedding_control

        #self.check_batch_embedding_control = batch_embedding_control

        projection_basis_cohort = tf.expand_dims(projection_basis_cohort, 0)
        projection_basis_cohort = tf.broadcast_to(projection_basis_cohort,
                                           shape=(batch_embedding_cohort.shape[0],
                                                  projection_basis_cohort.shape[1],
                                                  projection_basis_cohort.shape[2]))
        #batch_label_cohort = tf.cast(tf.expand_dims(batch_label,1),tf.float64)
        #batch_embedding_cohort = tf.multiply(batch_embedding,batch_label_cohort)



        projection_basis_control = tf.expand_dims(projection_basis_control, 0)
        projection_basis_control = tf.broadcast_to(projection_basis_control,
                                                  shape=(batch_embedding_control.shape[0],
                                                         projection_basis_control.shape[1],
                                                         projection_basis_control.shape[2]))

        #batch_label_control = tf.expand_dims(batch_label,1)

        batch_embedding_cohort = tf.expand_dims(batch_embedding_cohort, 1)
        batch_embedding_cohort = tf.broadcast_to(batch_embedding_cohort, [batch_embedding_cohort.shape[0],
                                                            projection_basis_cohort.shape[1],
                                                            self.latent_dim])

        batch_embedding_control = tf.expand_dims(batch_embedding_control, 1)
        batch_embedding_control = tf.broadcast_to(batch_embedding_control, [batch_embedding_control.shape[0],
                                                                          projection_basis_control.shape[1],
                                                                          self.latent_dim])

        #projection_cohort = tf.multiply(tf.cast(batch_embedding_cohort,tf.float64), tf.cast(projection_basis_cohort,tf.float64))
        #projection_cohort = tf.reduce_sum(projection_cohort, 2)

        projection_cohort = tf.math.subtract(tf.cast(batch_embedding_cohort,tf.float64),
                                             tf.cast(projection_basis_cohort,tf.float64))
        projection_cohort = tf.norm(projection_cohort, ord='euclidean', axis=-1)

        max_value_projection_cohort = np.argmin(projection_cohort, axis=1)

        #projection_control = tf.multiply(tf.cast(batch_embedding_control,tf.float64), tf.cast(projection_basis_control,tf.float64))
        #projection_control = tf.reduce_sum(projection_control, 2)

        projection_control = tf.math.subtract(tf.cast(batch_embedding_control, tf.float64),
                                             tf.cast(projection_basis_control, tf.float64))
        projection_control = tf.norm(projection_control, ord='euclidean', axis=-1)

        max_value_projection_control = np.argmin(projection_control, axis=1)

        return max_value_projection_cohort, max_value_projection_control, \
               batch_embedding_cohort_return, batch_embedding_control_return


    def simple_forward_layer(self):

        input = layers.Input(self.feature_num)

        forward_layer = tf.keras.layers.Dense(units=100,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=4),
                                                   activation=tf.nn.relu)
        forward_layer2 = tf.keras.layers.Dense(units=100,
                                              kernel_initializer=tf.keras.initializers.he_normal(seed=4),
                                              activation=tf.nn.relu)

        forward_layer3 = tf.keras.layers.Dense(units=100,
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=4),
                                               activation=tf.nn.relu)

        embedding_output = forward_layer(input)
        embedding_output = forward_layer2(embedding_output)
        embedding_output = forward_layer3(embedding_output)
        return tf.keras.Model(input,
                              [tf.math.l2_normalize(embedding_output, axis=-1)],
                              name='forward_layer')


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
                                           dilation_rate=dilation1, padding='valid',
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=4))
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
                                           padding='valid',
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=4))
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
                                           padding='valid',
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=4))
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
                                           padding='valid',
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=4))
        conv4_identity = tf.keras.layers.Conv1D(self.latent_dim, 1, activation='relu',
                                                dilation_rate=1)
        layernorm4 = tf.keras.layers.BatchNormalization()
        padding_4 = (self.tcn_filter_size - 1) * dilation4
        inputs4 = tf.pad(self.outputs3, tf.constant([[0, 0], [1, 0], [0, 0]]) * padding_4)
        self.outputs4 = tcn_conv4(inputs4)
        self.outputs4 = conv4_identity(self.outputs4)
        # self.outputs4 = layernorm4(self.outputs4)


        return tf.keras.Model(inputs,
                              tf.math.l2_normalize(self.outputs4,axis=-1))#, self.outputs3, self.outputs2, self.outputs1],
                              #name='tcn_encoder')


    def project_logit(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                layers.Dense(
                    50,
                    # use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                ),
                layers.Dense(
                    1,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='sigmoid'
                )
            ],
            name="projection_logit",
        )
        return model

    def train_mix_gaussian(self):
        self.tcn = self.tcn_encoder_second_last_level()
        # tcn = self.tcn(input)
        self.auc_all = []
        self.loss_track = []

    def train_regularizer(self):
        self.tcn = self.tcn_encoder_second_last_level()
        self.auc_all = []
        self.loss_track = []
        # self.model_extractor = tf.keras.Model(input, tcn, name="time_extractor")
        self.projection_layer = self.project_logit()
        self.bceloss = tf.keras.losses.BinaryCrossentropy()
        # self.model_extractor = tf.keras.Model(input, tcn, name="time_extractor")
        mse = tf.keras.losses.MeanSquaredError()

        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            tcn_cohort_whole = self.tcn(self.memory_bank_cohort)
            tcn_control_whole = self.tcn(self.memory_bank_control)

            on_site_extract_cohort_whole = [tcn_cohort_whole[i, np.abs(int(self.memory_bank_cohort_on_site[i] - 1)), :]
                                            for i
                                            in range(self.memory_bank_cohort_on_site.shape[0])]
            on_site_extract_array_cohort_whole = tf.stack(on_site_extract_cohort_whole)

            on_site_extract_control_whole = [
                tcn_control_whole[i, np.abs(int(self.memory_bank_control_on_site[i] - 1)), :] for i
                in range(self.memory_bank_control_on_site.shape[0])]

            on_site_extract_array_control_whole = tf.stack(on_site_extract_control_whole)

            self.max_value_projection_cohort, self.semantic_cluster_cohort = \
                self.E_step_initial(on_site_extract_array_cohort_whole)

            self.max_value_projection_control, self.semantic_cluster_control = \
                self.E_step_initial(on_site_extract_array_control_whole)

            self.cluster_cohort = []
            self.cluster_control = []
            self.cluster_cohort_index = []
            self.cluster_control_index = []
            for i in range(self.unsupervised_cluster_num):
                semantic_index_cohort = np.where(self.max_value_projection_cohort == i)[0]
                semantic_cohort = tf.gather(self.memory_bank_cohort, semantic_index_cohort)
                #semantic_group.append(semantic)
                self.cluster_cohort.append(semantic_cohort)
                self.cluster_cohort_index.append(semantic_index_cohort)

                semantic_index_control = np.where(self.max_value_projection_control == i)[0]
                semantic_control = tf.gather(self.memory_bank_control, semantic_index_control)
                self.cluster_control.append(semantic_control)
                self.cluster_control_index.append(semantic_index_control)


            y_label_cohort = np.ones(on_site_extract_array_cohort_whole.shape[0])
            y_label_control = np.zeros(on_site_extract_array_control_whole.shape[0])
            y_label = tf.concat([y_label_cohort,y_label_control], 0)

            train_data_whole = tf.concat([self.memory_bank_cohort,self.memory_bank_control],0)

            value_projection_whole = tf.concat([self.max_value_projection_cohort,self.max_value_projection_control],0)

            on_site_time_whole = tf.concat([self.memory_bank_cohort_on_site,self.memory_bank_control_on_site],0)

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data_whole, y_label, on_site_time_whole, value_projection_whole))  # ,self.train_sofa_score))
            self.train_dataset = self.train_dataset.shuffle(buffer_size=train_data_whole.shape[0],
                                                            seed=4).batch(self.batch_size)#,reshuffle_each_iteration=True).repeat(1).batch(self.batch_size)


            for step, (x_batch_train, y_batch_train, on_site_time, value_projection) in enumerate(self.train_dataset):
                self.check_x_batch = x_batch_train
                self.check_on_site_time = on_site_time
                self.check_label = y_batch_train

                #self.check_semantic_origin = semantic_origin
                prior_centers = np.zeros((x_batch_train.shape[0],self.latent_dim))

                index_positive = np.where(y_batch_train == 1)[0]
                index_negative = np.where(y_batch_train == 0)[0]

                for i in index_positive:
                    prior_centers[i,:] = self.center_1

                for i in index_negative:
                    prior_centers[i,:] = self.center_2

                self.check_prior_centers = prior_centers

                batch_cluster_cohort = []
                batch_cluster_cohort_index = []
                for i in range(self.unsupervised_cluster_num):
                    random_indices = np.random.choice(self.cluster_cohort[i].shape[0],
                                                      size=self.num_cluster_cohort,replace=False)
                    batch_cluster_cohort_single = tf.gather(self.cluster_cohort[i],random_indices)
                    batch_cluster_cohort_index_single = tf.gather(self.cluster_cohort_index[i],random_indices)
                    batch_cluster_cohort.append(batch_cluster_cohort_single)
                    batch_cluster_cohort_index.append(batch_cluster_cohort_index_single)
                batch_cluster_cohort = tf.stack(batch_cluster_cohort)
                self.check_batch_cluster_cohort = batch_cluster_cohort
                batch_cluster_cohort_index = tf.stack(batch_cluster_cohort_index)
                self.check_batch_cluster_cohort_index = batch_cluster_cohort_index

                batch_cluster_control = []
                batch_cluster_control_index = []
                for i in range(self.unsupervised_cluster_num):
                    random_indices = np.random.choice(self.cluster_control[i].shape[0],
                                                      size=self.num_cluster_control,replace=False)
                    batch_cluster_control_single = tf.gather(self.cluster_control[i],random_indices)
                    batch_cluster_control_index_single = tf.gather(self.cluster_control_index[i], random_indices)
                    batch_cluster_control.append(batch_cluster_control_single)
                    batch_cluster_control_index.append(batch_cluster_control_index_single)
                batch_cluster_control = tf.stack(batch_cluster_control)
                self.check_batch_cluster_control = batch_cluster_control
                batch_cluster_control_index = tf.stack(batch_cluster_control_index)
                self.check_batch_cluster_control_index = batch_cluster_control_index


                with tf.GradientTape() as tape:
                    tcn_temporal_output = self.tcn(x_batch_train)
                    tcn_temporal_output_cohort = self.tcn(tf.reshape(batch_cluster_cohort,
                                                                     [batch_cluster_cohort.shape[0]*
                                                                      batch_cluster_cohort.shape[1],
                                                                      batch_cluster_cohort.shape[2],
                                                                      batch_cluster_cohort.shape[3]]))
                    tcn_output_cohort_index = tf.reshape(batch_cluster_cohort_index,[batch_cluster_cohort_index.shape[0]*
                                                                                      batch_cluster_cohort_index.shape[1]])


                    self.check_tcn_temporal_output_cohort = tcn_temporal_output_cohort
                    self.check_tcn_output_cohort_index = tcn_output_cohort_index
                    tcn_temporal_output_control = self.tcn(tf.reshape(batch_cluster_control,
                                                                     [batch_cluster_control.shape[0] *
                                                                      batch_cluster_control.shape[1],
                                                                      batch_cluster_control.shape[2],
                                                                      batch_cluster_control.shape[3]]))

                    tcn_output_control_index = tf.reshape(batch_cluster_control_index,
                                                          [batch_cluster_control_index.shape[0] *
                                                           batch_cluster_control_index.shape[1]])



                    on_site_extract = [tcn_temporal_output[i, np.abs(int(on_site_time[i] - 1)), :] for i in
                                       range(on_site_time.shape[0])]
                    on_site_extract_array = tf.stack(on_site_extract)

                    self.check_on_site_extract = on_site_extract_array

                    index_cohort = np.where(y_batch_train == 1)[0]

                    batch_embedding_cohort = tf.gather(on_site_extract_array, index_cohort)

                    batch_embedding_cohort_project = tf.gather(value_projection, index_cohort)

                    index_control = np.where(y_batch_train == 0)[0]

                    batch_embedding_control = tf.gather(on_site_extract_array, index_control)

                    batch_embedding_control_project = tf.gather(value_projection, index_control)



                    on_site_extract_cohort_cluster = [tcn_temporal_output_cohort[i,
                                                      np.abs(int(self.memory_bank_cohort_on_site[int(tcn_output_cohort_index[i])] - 1)), :] for i in
                                       range(tcn_output_cohort_index.shape[0])]
                    on_site_extract_cohort_cluster_array = tf.stack(on_site_extract_cohort_cluster)
                    self.check_on_site_extract_cohort_cluster_array = on_site_extract_cohort_cluster_array

                    semantic_cluster_cohort_whole_ = tf.reduce_mean(on_site_extract_cohort_cluster_array,axis=0)
                    self.check_semantic_cluster_cohort_whole_ = semantic_cluster_cohort_whole_

                    on_site_extract_cohort_cluster = tf.reshape(on_site_extract_cohort_cluster_array,
                                                            [batch_cluster_cohort.shape[0],
                                                             batch_cluster_cohort.shape[1],
                                                             on_site_extract_cohort_cluster_array.shape[-1]])

                    semantic_cluster_cohort = tf.reduce_mean(on_site_extract_cohort_cluster,axis=1)
                    self.check_semantic_cluster_cohort = semantic_cluster_cohort

                    self.check_on_site_extract_cohort_cluster = on_site_extract_cohort_cluster

                    on_site_extract_control_cluster = [
                        tcn_temporal_output_control[i, np.abs(int(self.memory_bank_control_on_site[int(tcn_output_control_index[i])] - 1)), :] for i in
                        range(tcn_output_control_index.shape[0])]
                    on_site_extract_control_array = tf.stack(on_site_extract_control_cluster)

                    self.check_on_site_extract_control_array = on_site_extract_control_array

                    semantic_cluster_control_whole_ = tf.reduce_mean(on_site_extract_control_array,axis=0)
                    self.check_semantic_cluster_control_whole_ = semantic_cluster_control_whole_

                    semantic_cluster_whole_ = tf.stack([semantic_cluster_control_whole_,semantic_cluster_cohort_whole_])
                    self.check_semantic_cluster_whole_ = semantic_cluster_whole_

                    on_site_extract_control_cluster = tf.reshape(on_site_extract_control_array,
                                                             [batch_cluster_control.shape[0],
                                                              batch_cluster_control.shape[1],
                                                              on_site_extract_control_array.shape[-1]])

                    semantic_cluster_control = tf.reduce_mean(on_site_extract_control_cluster,axis=1)
                    self.check_semantic_cluster_control = semantic_cluster_control

                    cl_loss = self.info_nce_loss_local(on_site_extract_array,
                                                       semantic_cluster_whole_,
                                                       y_batch_train)
                    mse_loss = tf.cast(mse(prior_centers,on_site_extract_array),tf.float64)

                    cl_loss_local_cohort = self.info_nce_loss_local(batch_embedding_cohort,
                                                                    semantic_cluster_cohort,
                                                                    batch_embedding_cohort_project)

                    cl_loss_local_control = self.info_nce_loss_local(batch_embedding_control,
                                                                     semantic_cluster_control,
                                                                     batch_embedding_control_project)

                    #loss = tf.cast(cl_loss_local_control,tf.float64)# + 0.4*tf.cast(mse_loss,tf.float64)
                    loss = 0.4*cl_loss_local_control + 0.4*cl_loss_local_cohort + cl_loss
                    #loss = cl_loss
                    #if epoch % 2 == 1:
                        #loss =progression_loss

                #if epoch == 0 or epoch % 2 == 0:
                gradients = \
                    tape.gradient(loss,
                                  self.tcn.trainable_variables)#+ self.projection_layer.trainable_weights)
                                  #+self.deconv.trainable_variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients,
                                              self.tcn.trainable_variables))#+ self.projection_layer.trainable_weights))

                if step % 20 == 0:
                    #if epoch == 0 or epoch % 2 == 0:
                    print("Training cl_loss cohort(for one batch) at step %d: %.4f"
                          % (step, float(cl_loss)))
                    print("Training cl_local_cohort(for one batch) at step %d: %.4f"
                          % (step, float(cl_loss_local_cohort)))
                    print("Training cl_local_control(for one batch) at step %d: %.4f"
                          % (step, float(cl_loss_local_control)))

                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))
                    #print(self.check_semantic_cluster_cohort)

                    self.loss_track.append(loss)


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
            for step, (x_batch_train, y_batch_train, on_site_time) in enumerate(self.train_dataset):
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


    def vis_distribution(self,number_vis,min_dist,c_num,train_num,scale):
        tcn_cohort_whole = self.tcn(self.memory_bank_cohort)
        tcn_control_whole = self.tcn(self.memory_bank_control)

        on_site_extract_cohort_whole = [tcn_cohort_whole[i, np.abs(int(self.memory_bank_cohort_on_site[i] - 1)), :] for
                                        i
                                        in range(self.memory_bank_cohort_on_site.shape[0])]
        on_site_extract_array_cohort_whole = tf.stack(on_site_extract_cohort_whole)

        on_site_extract_control_whole = [tcn_control_whole[i, np.abs(int(self.memory_bank_control_on_site[i] - 1)), :]
                                         for i
                                         in range(self.memory_bank_control_on_site.shape[0])]

        on_site_extract_array_control_whole = tf.stack(on_site_extract_control_whole)

        cohort_vis = on_site_extract_array_cohort_whole[0:number_vis]
        label_cohort_vis = self.max_value_projection_cohort[0:number_vis]
        self.check_cohort_vis = cohort_vis
        control_vis = on_site_extract_array_control_whole[0:number_vis]
        label_control_vis = self.max_value_projection_control[0:number_vis]+self.unsupervised_cluster_num
        self.check_control_vis = control_vis

        y_label = np.zeros(2*number_vis)
        y_label[0:number_vis] = 1
        vis_total = tf.concat([cohort_vis,control_vis],axis=0)
        y_label_cluster = tf.concat([label_cohort_vis,label_control_vis],axis=0)
        self.check_vis_total = vis_total

        CL_k = np.squeeze(umap.UMAP(min_dist=min_dist,random_state=42,n_components=1).fit_transform(vis_total))/scale
        #CL_k = (CL_k - CL_k.min())/56
        #CL_k = np.array(tf.math.l2_normalize(CL_k, axis=-1))
        self.check_CL_k = CL_k
        CL_k_fit = np.expand_dims(CL_k,1)

        dataframe = np.transpose(np.stack([y_label_cluster,CL_k]))
        #dataframe = np.transpose(np.stack([y_label, CL_k]))
        self.check_dataframe = dataframe

        df = pd.DataFrame(dataframe,columns=['label','Embedding'])


        self.check_df = df

        sns.displot(df,x='Embedding',hue='label',kind='kde',palette=['b','b','b','r','r','r'])

        #sns.displot(df, x='Embedding', hue='label', kind='kde', palette=['b','r'])

        train_lr_total,train_lr_label = shuffle(CL_k_fit,y_label, random_state=4)
        self.c_total = []
        for i in range(c_num):
            lr = LogisticRegression()
            lr.fit(train_lr_total[i*train_num:(i+1)*train_num], train_lr_label[i*train_num:(i+1)*train_num])
            #lr.fit(CL_k_fit, y_label)

            b = lr.intercept_[0]
            w = lr.coef_[0][0]
            # Calculate the intercept and gradient of the decision boundary.
            c = -b / w
            self.c_total.append(c)

            # ax = plt.gca()
            # ax.autoscale(True)
            x_vals = np.array([c,c])
            #x_vals = np.array([CL_k[:,0].min()-x_scale, CL_k[:,0].max()]+x_scale)
            #y_vals_1 = m * x_vals + c
            y_vals = np.array([0,1])
            plt.plot(x_vals, y_vals, '--', c="black", linewidth=1.5)

        #plt.ylim(0,1)
        #plt.xlim(0,1)
        plt.show()




    def vis_embedding(self,number_vis,min_dist,c_num,train_num):

        tcn_cohort_whole = self.tcn(self.memory_bank_cohort)
        tcn_control_whole = self.tcn(self.memory_bank_control)

        on_site_extract_cohort_whole = [tcn_cohort_whole[i, np.abs(int(self.memory_bank_cohort_on_site[i] - 1)), :] for
                                        i
                                        in range(self.memory_bank_cohort_on_site.shape[0])]
        on_site_extract_array_cohort_whole = tf.stack(on_site_extract_cohort_whole)

        on_site_extract_control_whole = [tcn_control_whole[i, np.abs(int(self.memory_bank_control_on_site[i] - 1)), :]
                                         for i
                                         in range(self.memory_bank_control_on_site.shape[0])]

        on_site_extract_array_control_whole = tf.stack(on_site_extract_control_whole)

        cohort_vis = on_site_extract_array_cohort_whole[0:number_vis]
        self.check_cohort_vis = cohort_vis
        control_vis = on_site_extract_array_control_whole[0:number_vis]
        self.check_control_vis = control_vis

        y_label = np.zeros(2*number_vis)
        y_label[0:number_vis] = 1
        vis_total = tf.concat([cohort_vis,control_vis],axis=0)
        self.check_vis_total = vis_total


        CL_k = umap.UMAP(min_dist=min_dist,random_state=42,n_components=2).fit_transform(vis_total)
        #CL_k = np.array(tf.math.l2_normalize(CL_k, axis=-1))
        self.check_CL_k = CL_k

        """
        tcn_whole_1 = self.tcn(self.train_data[0:3000, :, :])
        train_label_1 = self.train_logit[0:3000]
        train_on_site_whole_1 = self.train_on_site_time[0:3000]
        on_site_extract_whole_1 = [tcn_whole_1[i, np.abs(int(train_on_site_whole_1[i] - 1)), :] for
                                   i
                                   in range(train_on_site_whole_1.shape[0])]
        
        CL_k_1 = umap.UMAP(min_dist=min_dist, random_state=42, n_components=2).fit_transform(on_site_extract_whole_1)
        """

        train_lr_total,train_lr_label = shuffle(CL_k,y_label, random_state=4)

        self.check_train_lr_total = train_lr_total
        self.check_train_lr_label = train_lr_label
        x_scale = (CL_k[:, 0].max() - CL_k[:, 0].min()) / 10

        for i in range(c_num):
            lr = LogisticRegression()
            lr.fit(train_lr_total[i*train_num:(i+1)*train_num], train_lr_label[i*train_num:(i+1)*train_num])

            b = lr.intercept_[0]
            w1, w2 = lr.coef_.T
            # Calculate the intercept and gradient of the decision boundary.
            c = -b / w2
            m = -w1 / w2

            # ax = plt.gca()
            # ax.autoscale(True)
            x_vals = np.array([CL_k[:,0].min()-x_scale, CL_k[:,0].max()]+x_scale)
            y_vals_1 = m * x_vals + c
            plt.plot(x_vals, y_vals_1, '--', c="black", linewidth=1.5)

        #plt.scatter(CL_k[:,0],CL_k[:,1])

        for i in range(number_vis):
            plt.plot(CL_k[i][0], CL_k[i][1], '.', fillstyle='none',color='red', markersize=2)
        for i in range(number_vis):
            plt.plot(CL_k[i+number_vis][0], CL_k[i+number_vis][1], '.', fillstyle='none',
                      color='blue', markersize=2)


        # plt.plot(CL_k[-2][0], CL_k[-2][1], 'o', color='yellow', markersize=9)
        # plt.plot(CL_k[-1][0], CL_k[-1][1], 'o', color='green', markersize=9)


        plt.xlim([CL_k[:,0].min()-x_scale,CL_k[:,0].max()+x_scale])
        y_scale = (CL_k[:,1].max()-CL_k[:,1].min())/8
        plt.ylim([CL_k[:, 1].min()-y_scale , CL_k[:, 1].max()+y_scale])
        plt.show()
