import tensorflow as tf
#from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import numpy as np

class load_embeddings:
    def __init__(self):
        self.unsupervised_cluster_num_cohort = 5
        self.unsupervised_cluster_num_control = 5
        self.latent_dim = 100
        self.converge_threshold_E = 50
        self.load_embedding()
        self.initializer_basis = tf.keras.initializers.Orthogonal()
        self.init_projection_basis_cohort = tf.Variable(
            self.initializer_basis(shape=(self.unsupervised_cluster_num_cohort, self.latent_dim)))
        self.init_projection_basis_control = tf.Variable(
            self.initializer_basis(shape=(self.unsupervised_cluster_num_control, self.latent_dim)))

    def load_embedding(self):
        with open('on_site_embedding4.npy', 'rb') as f:
            self.on_site_embedding = np.load(f)
        with open('on_site_logit4.npy', 'rb') as f:
            self.on_site_logit = np.load(f)
        #with open('temporal_semantic_embedding5.npy', 'rb') as f:
         #   self.temporal_semantic_embedding = np.load(f)
        #with open('temporal_semantic_embedding_cohort.npy','rb') as f:
         #   self.temporal_semantic_embedding_cohort = np.load(f)
        #with open('temporal_semantic_embedding_control.npy','rb') as f:
         #   self.temporal_semantic_embedding_control = np.load(f)

        """
        self.on_site_logit = tf.expand_dims(self.on_site_logit,1)
        self.on_site_logit = tf.broadcast_to(self.on_site_logit,(self.on_site_logit.shape[0],
                                                                 self.temporal_semantic_embedding.shape[1]))
        self.on_site_logit = tf.reshape(self.on_site_logit,(self.on_site_logit.shape[0]*self.on_site_logit.shape[1]))
        self.temporal_semantic_embedding = tf.reshape(self.temporal_semantic_embedding,
                                                      (self.temporal_semantic_embedding.shape[0]
                                                       *self.temporal_semantic_embedding.shape[1],
                                                       self.temporal_semantic_embedding.shape[2]))
        self.temporal_semantic_embedding_cohort = tf.reshape(self.temporal_semantic_embedding_cohort,
                                                      (self.temporal_semantic_embedding_cohort.shape[0]
                                                       * self.temporal_semantic_embedding_cohort.shape[1],
                                                       self.temporal_semantic_embedding_cohort.shape[2]))
        self.temporal_semantic_embedding_control = tf.reshape(self.temporal_semantic_embedding_control,
                                                      (self.temporal_semantic_embedding_control.shape[0]
                                                       * self.temporal_semantic_embedding_control.shape[1],
                                                       self.temporal_semantic_embedding_control.shape[2]))
        """

    def vis_embedding_load(self):
        #CL_k = TSNE(n_components=2).fit_transform(np.array(self.on_site_embedding)[0:5000, :])

        CL_k = umap.UMAP().fit_transform(np.array(self.on_site_embedding)[0:5000, :])
        for i in range(5000):
            if self.on_site_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.on_site_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        plt.show()



    def unsupervised_prototype_detection(self,batch_embedding,projection_basis):

        semantic_cluster = []
        projection_basis_transform = tf.expand_dims(projection_basis, 0)
        projection_basis_transform = tf.broadcast_to(projection_basis_transform,
                                           shape=(batch_embedding.shape[0],
                                                  projection_basis.shape[0], projection_basis.shape[1]))

        self.first_check_projection = projection_basis_transform

        batch_embedding_whole = batch_embedding
        self.check_batch_embedding_whole = batch_embedding_whole

        batch_embedding = tf.expand_dims(batch_embedding, 1)
        batch_embedding = tf.broadcast_to(batch_embedding, [batch_embedding.shape[0],
                                                            projection_basis.shape[0],
                                                            self.latent_dim])

        self.check_batch_embedding_E = batch_embedding

        check_converge = 100 * np.ones((batch_embedding.shape[0]))

        self.check_check_converge = check_converge

        check_converge_num = 1000
        self.check_converge_num = check_converge_num

        max_value_projection = 0

        while (check_converge_num > self.converge_threshold_E):
            print(check_converge_num)
            basis = tf.math.l2_normalize(projection_basis_transform, axis=-1)
            self.check_basis = basis

            #basis = tf.expand_dims(basis, 0)
            #basis = tf.broadcast_to(basis, [batch_embedding.shape[0],
             #                               projection_basis.shape[0],
              #                              self.latent_dim])
            basis = tf.cast(basis, tf.float64)
            self.check_basis_E = basis

            projection = tf.multiply(batch_embedding, basis)
            projection = tf.reduce_sum(projection, 2)

            self.check_projection_E = projection
            max_value_projection = np.argmax(projection, axis=1)
            self.check_max_value_projection = max_value_projection

            projection_basis_whole = max_value_projection

            self.projection_basis_whole = projection_basis_whole

            semantic_cluster = []

            for i in range(projection_basis.shape[0]):
                semantic_index = np.where(self.projection_basis_whole == i)[0]
                semantic = tf.gather(batch_embedding_whole, semantic_index)
                semantic = tf.reduce_mean(semantic, 0)
                semantic_cluster.append(semantic)

            semantic_cluster = tf.stack(semantic_cluster, 0)

            self.check_semantic_cluster = semantic_cluster

            projection_basis_transform = semantic_cluster

            projection_basis_transform = tf.expand_dims(projection_basis_transform, 0)
            projection_basis_transform = tf.broadcast_to(projection_basis_transform,
                                               shape=(batch_embedding.shape[0],
                                                      projection_basis.shape[0], projection_basis.shape[1]))

            cluster_diff = projection_basis_whole - check_converge
            check_converge = projection_basis_whole
            self.check_converge = check_converge

            self.check_cluster_diff = cluster_diff

            check_converge_num = len(np.where(cluster_diff != 0)[0])

        return max_value_projection, semantic_cluster

    def vis_embedding_tsl_load(self):
        #center_cohort = np.expand_dims(np.mean(self.temporal_semantic_embedding_cohort,0),0)
        #center_control = np.expand_dims(np.mean(self.temporal_semantic_embedding_control,0),0)
        max_value_projection_cohort,semantic_cluster_cohort = \
            self.unsupervised_prototype_detection(self.temporal_semantic_embedding_cohort,
                                                 self.init_projection_basis_cohort)
        max_value_projection_control, semantic_cluster_control = \
            self.unsupervised_prototype_detection(self.temporal_semantic_embedding_control,
                                                 self.init_projection_basis_control)
        self.test_temporal_semantic_embedding = self.temporal_semantic_embedding[0:5000,:]
        self.test_temporal_semantic_embedding = np.concatenate((self.test_temporal_semantic_embedding,
                                                                semantic_cluster_cohort),
                                                               axis=0)
        self.test_temporal_semantic_embedding = np.concatenate((self.test_temporal_semantic_embedding,
                                                                semantic_cluster_control),
                                                               axis=0)
        CL_k = umap.UMAP().fit_transform(self.test_temporal_semantic_embedding)



        for i in range(5000):
            if self.on_site_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.on_site_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        for i in range(self.unsupervised_cluster_num_cohort):
            plt.plot(CL_k[-i-self.unsupervised_cluster_num_control][0],
                     CL_k[-i-self.unsupervised_cluster_num_control][1],'o',color='green',markersize=9)
        for i in range(self.unsupervised_cluster_num_control):
            plt.plot(CL_k[-i][0],CL_k[-i][1], 'o', color='yellow', markersize=9)
        self.check_CL = CL_k


        plt.show()

    def vis_embedding_all_load(self):
        CL_k = TSNE(n_components=2).fit_transform(self.temporal_semantic_embedding[0:3000, :])
        CL_k_on_site = TSNE(n_components=2).fit_transform(np.array(self.on_site_embedding)[0:3000, :])

        for i in range(3000):
            if self.on_site_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
                plt.plot(CL_k_on_site[i][0], CL_k_on_site[i][1], 'o', color='yellow', markersize=1)
            if self.on_site_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)
                plt.plot(CL_k_on_site[i][0], CL_k_on_site[i][1], 'o', color='green', markersize=5)


if __name__ == "__main__":
    l = load_embeddings()