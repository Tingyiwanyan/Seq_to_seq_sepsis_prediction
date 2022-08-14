import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def addlabels_sup(x,y):
    for i in range(len(x)):
        plt.text(x[i], y[i], y[i], ha = 'right')

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i], y[i], y[i], ha = 'center')
# set width of bar
barWidth = 0.25
#fig = plt.subplots(figsize=(12, 8))

# set height of bar
IT = [0.750, 0.787, 0.773, 0.767, 0.754]
ECE = [0.750, 0.750, 0.750, 0.750, 0.750]
#CSE = [29, 3, 24, 25, 17]

# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
#br3 = [x + barWidth for x in br2]

sns.set_style("whitegrid")
# Make the plot
rect1=plt.bar(br2, IT, color='darkorange', width=barWidth,
        edgecolor='black', lw=2,label='Ours')
rect2=plt.bar(br1, ECE, color='royalblue', width=barWidth,
        edgecolor='black', lw=2, label='Supclr')

addlabels_sup(br1,ECE)
addlabels(br2,IT)
#plt.bar(br3, CSE, color='b', width=barWidth,
        #edgecolor='black', label='CSE')

# Adding Xticks
plt.xlabel('Cluster number', fontweight='bold', fontsize=10)
plt.ylabel('AUC', fontweight='bold', fontsize=10)
plt.xticks([r + barWidth for r in range(len(IT))],
           ['0', '3', '5', '7', '9'])

#plt.bar_label(rect1,padding=3)
#plt.bar_label(rect2,padding=3)
plt.ylim(0.7,0.8)
plt.legend()
plt.show()