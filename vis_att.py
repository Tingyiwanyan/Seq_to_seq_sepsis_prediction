import pandas as pd
import plotly.graph_objects as go
import numpy as np

with open('temporal_progression_att_274', 'rb') as f:
    temporal_progression_att = np.load(f)
with open('final_embedding_att_274', 'rb') as f:
    final_embedding_att = np.load(f)

label_single = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']
source_index_single = list(range(34))
label_whole = []
#source_ = []
for i in range(38):
    label_whole = label_whole + label_single
    #source_ = source_ + list(np.add(source_index_single,34*i))

label_whole = label_whole + ['sepsis_on_site']
#source_ = source_ + [len(source_)]

final_embedding_att = final_embedding_att[38]
relation_on_site = np.where(final_embedding_att == final_embedding_att.max())[0][0]

"""
label_list = ['cat', 'dog', 'domesticated', 'female', 'male', 'wild']
# cat: 0, dog: 1, domesticated: 2, female: 3, male: 4, wild: 5
source = [0, 0, 1, 3, 4, 4]
target = [3, 4, 4, 2, 2, 5]
count = [21, 6, 22, 21, 6, 22]

fig = go.Figure(data=[go.Sankey(
    node = {"label": label_list},
    link = {"source": source, "target": target, "value": count}
    )])
fig.show()
"""

source = []
target = []
value = []
previous_index = []
previous_index_next = []
max_info_index = relation_on_site
value_single = []
#for i in range(3):
check_att_values = []
check_feature_index =[]
for i in range(4):
    print(target)
    if i == 0:
        top_values = np.sort(final_embedding_att)[::-1][0:4]
        for k in top_values:
            index_preserve = np.where(final_embedding_att==k)[0][0]
            source = source + [index_preserve+(37-i)*34]
            target = target + [len(label_whole)-1]
            value_single = value_single + [k]
            previous_index_next = previous_index_next + [index_preserve]
        [previous_index.append(l) for l in previous_index_next]
        #print(previous_index)
        #value_single = [i/np.sum(value_single) for i in value_single]
        value = value + value_single
        value_single = []
        previous_index_next = []
        #value = value + list(final_embedding_att)
    else:
        print(previous_index)
        for j in previous_index:
            top_values = np.sort(temporal_progression_att[37-i][j])[::-1][0:4]
            check_att_values.append(temporal_progression_att[37-i][j])
            check_feature_index.append([37-i,j])
            for k in top_values:
                index_preserve = np.where(temporal_progression_att[37-i][j] == k)[0][0]
                source = source + [index_preserve+(37-i)*34]
                target = target + [j+(38-i)*34]
                value = value + [k]
                previous_index_next = previous_index_next + [index_preserve]
        previous_index = []
        [previous_index.append(l) for l in previous_index_next]
        #value_single = [i / np.sum(value_single) for i in value_single]
        value = value + value_single
        value_single = []
        previous_index_next = []


# data
"""
label = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
source = [0, 0, 1, 1, 0]
target = [2, 3, 4, 5, 4]
value = [8, 2, 2, 8, 4]
# data to dict, dict to sankey
"""
#source = source[3:]
#target = target[3:]
#value = value[3:]
link = dict(source = source, target = target, value = value)
node = dict(label = label_whole, pad=100, thickness=10)
data = go.Sankey(link = link, node=node)
# plot
fig = go.Figure(data)
fig.show()
