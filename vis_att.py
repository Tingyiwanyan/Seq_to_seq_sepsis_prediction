import pandas as pd
import plotly.graph_objects as go
import numpy as np

with open('temporal_progression_att.npy', 'rb') as f:
    temporal_progression_att = np.load(f)
with open('final_embedding_att.npy', 'rb') as f:
    final_embedding_att = np.load(f)

label_single = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']

label_whole = []
for i in range(8):
    label_whole = label_whole + label_single

label_whole = label_whole + ['sepsis_on_site']

final_embedding_att = final_embedding_att[8]
relation_on_site = np.where(final_embedding_att == final_embedding_att.max())[0][0]

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