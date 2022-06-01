from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

class load_embeddings():
    def __int__(self):
        self.load_embedding()

    def load_embedding(self):
        with open('on_site_embedding.npy', 'rb') as f:
            self.on_site_embedding = np.load(f)
        with open('on_site_logit.npy', 'rb') as f:
            self.on_site_logit = np.load(f)
        with open('temporal_semantic_embedding.npy', 'rb') as f:
            self.temporal_semantic_embedding = np.load(f)


    def vis_embedding_load(self):
        CL_k = TSNE(n_components=2).fit_transform(np.array(self.on_site_embedding)[0:5000, :])
        for i in range(5000):
            if self.on_site_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.on_site_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        plt.show()


    def vis_embedding_tsl_load(self):
        CL_k = TSNE(n_components=2).fit_transform(self.temporal_semantic_embedding[0:5000, :])

        for i in range(5000):
            if self.on_site_logit[i] == 0:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='blue', markersize=1)
            if self.on_site_logit[i] == 1:
                plt.plot(CL_k[i][0], CL_k[i][1], 'o', color='red', markersize=5)

        plt.show()

if __name__ == "__main__":
    l = load_embeddings()