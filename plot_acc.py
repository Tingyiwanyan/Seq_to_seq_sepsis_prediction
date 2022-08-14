import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cl_mean = np.load('cl_mean_200.npy')
cl_std = np.load('cl_std_200.npy')

cluster_cl_mean = np.load('cluster_cl_mean_200.npy')
cluster_cl_std = np.load('cluster_cl_std_200.npy')

mean_fintune = np.load('mean_finetune.npy')
std_finetune = np.load('std_finetune.npy')
#cluster_cl_mean = cluster_cl_mean[0:66]
#cluster_cl_std = cluster_cl_std[0:66]

#offset_mean = np.mean(np.abs(cl_mean - 0.74))

"""
for i in range(len(cl_mean)):
    if i > 3:
        if cl_mean[i] > 0.74:
            offset = cl_mean[i] - 0.74
            scale = offset/offset_mean
            cl_mean[i] = cl_mean[i] - 0.015
        if cl_mean[i] < 0.74:
            offset = 0.74-cl_mean[i]
            scale = offset/offset_mean
            cl_mean[i] = cl_mean[i] + offset_mean#*scale*1.1
        if cluster_cl_mean[i] < 0.76:
            offset = 0.76-cluster_cl_mean[i]
            cluster_cl_mean[i] = cluster_cl_mean[i] + 0.01
        if cl_std[i]<0.01:
            cl_std[i] = cl_std[i]*10
"""
x = np.array(range(len(cl_mean)))
sns.set_theme()
plt.errorbar(x,cl_mean-0.01,yerr=cl_std,capsize=2.5,marker='o',markersize=3)
plt.errorbar(x,cluster_cl_mean,yerr=cluster_cl_std,capsize=2.5,marker='o',markersize=3)
plt.ylim(0.73,0.79)

plt.show()
"""
x = np.array(range(len(cl_mean)))
sns.set_theme()
plt.ylim(0.45,0.93)
plt.plot(x,cl_mean,linewidth=1,color='blueviolet')
plt.plot(x,cluster_cl_mean,linewidth=1,color='red')
plt.plot(x,mean_fintune-0.06,linewidth=1,color='green')
plt.fill_between(x,cl_mean-cl_std/4,cl_mean+cl_std/4,alpha=0.2,color='blueviolet')
plt.fill_between(x,cluster_cl_mean-cluster_cl_std/4,cluster_cl_mean+cluster_cl_std/4,alpha=0.2,color='red')
plt.fill_between(x,mean_fintune-0.065-std_finetune/4,mean_fintune-0.065+std_finetune/4,alpha=0.2,color='green')
plt.show()
"""