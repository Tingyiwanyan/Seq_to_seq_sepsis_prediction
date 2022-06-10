import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir
#from seqehr_origin import seq_seq_ehr
from cvae import protatype_ehr, projection, translation
#from tcn_prospective import seq_seq_ehr


class read_data():
    """
    Loading data
    """
    def __init__(self):
        self.file_path = '/home/tingyi/physionet_data/training_setA/training/'
        self.file_path_val = '/home/tingyi/physionet_data/training_setB/training/'
        self.file_names = listdir(self.file_path)
        self.train_prop = 0.7
        self.test_prop = 0.3
        self.total_size = 7000
        self.total_logit = np.zeros(7000)
        self.total_data = self.file_names[0:self.total_size]
        self.train_data = self.file_names[0:14235]
        self.test_data = self.file_names[14235:18302]
        self.val_data = self.file_names[18302:20336]
        self.sepsis_group = []
        self.non_sepsis_group = []
        self.total_data = []
        self.total_data_label = []
        self.total_gender_label = []
        self.female_group = []
        self.male_group = []
        self.median_vital_signal = np.zeros(35)
        self.std_vital_signal = np.zeros(35)
        self.median_vital_signal_female = np.zeros(35)
        self.std_vital_signal_female = np.zeros(35)
        self.median_vital_signal_male = np.zeros(35)
        self.std_vital_signal_male = np.zeros(35)
        self.dic_item = {}
        self.dic_item_female = {}
        self.dic_item_male = {}
        self.dic_item_sepsis = {}
        self.dic_item_non_sepsis = {}
        self.time_sequence = 48

        self.ave_all = [ 8.38230435e+01,  9.75000000e+01,  3.69060000e+01,  1.18333333e+02,
        7.71140148e+01,  5.90000000e+01,  1.81162791e+01,  0.00000000e+00,
       -2.50000000e-01,  2.43333333e+01,  5.04195804e-01,  7.38666667e+00,
        4.00504808e+01,  9.60000000e+01,  4.20000000e+01,  1.65000000e+01,
        7.70000000e+01,  8.35000000e+00,  1.06000000e+02,  9.00000000e-01,
        1.16250000e+00,  1.25333333e+02,  1.65000000e+00,  2.00000000e+00,
        3.36666667e+00,  4.08000000e+00,  7.00000000e-01,  3.85000000e+00,
        3.09000000e+01,  1.05000000e+01,  3.11000000e+01,  1.08333333e+01,
        2.55875000e+02,  1.93708333e+02,  6.47200000e+01]

        self.std_all = [1.40828962e+01, 2.16625304e+00, 5.53108392e-01, 1.66121889e+01,
       1.08476132e+01, 9.94962122e+00, 3.59186362e+00, 0.00000000e+00,
       3.89407506e+00, 3.91858658e+00, 2.04595954e-01, 5.93467422e-02,
       7.72257867e+00, 8.87388075e+00, 5.77276895e+02, 1.79879091e+01,
       1.36508822e+02, 6.95188900e-01, 5.09788015e+00, 1.43347221e+00,
       3.75415153e+00, 4.03968485e+01, 1.71418146e+00, 3.15505742e-01,
       1.17084555e+00, 4.77914796e-01, 3.62933460e+00, 9.91058703e+00,
       4.60374699e+00, 1.64019340e+00, 1.68795640e+01, 6.23941196e+00,
       1.75014175e+02, 1.03316340e+02, 1.62930171e+01]

        self.standard = [96.0,112.4,62.4,75.1,21.6,61.2,232.1,1.9,2.9,33.2,7.33,41.9,3.2]
        self.standard_name = ['HR','SBP','DBP','MBP','Resp','FiO2','Platelets','Creatinine','Lactate','BUN','PH',
                              'PaCO2']

        self.names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
       'HospAdmTime', 'ICULOS', 'SepsisLabel']
        #self.index = [0,3,5,6,10,33,19,22,15,11,12]

        self.missingness_all = [0.08236603, 0.12678666, 0.65877425, 0.15378036, 0.1088652 ,
       0.50510896, 0.104005  , 1.        , 0.89657094, 0.91816108,
       0.86952604, 0.88554305, 0.9132697 , 0.9497429 , 0.98520725,
       0.91700666, 0.98564741, 0.95106905, 0.91522709, 0.93263183,
       0.99864895, 0.87366508, 0.96673776, 0.9227391 , 0.95044262,
       0.8898439 , 0.98793365, 0.99876108, 0.87908121, 0.90976203,
       0.95157457, 0.9233135 , 0.99264736, 0.93361607]

        self.missingness_9 = [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 11, 21, 25, 28]
        self.missingness_95 = [ 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 15, 18, 19, 21,
        23, 25, 28, 29, 31, 33]

        self.index = [4,26,34,19]

        self.cardiovas_index = 4
        self.liver_index = 26
        self.coagulation_index = 34
        self.kidney_index = 19

        self.cardiovas_standard = 0.1
        self.liver_standard = 0.0
        self.coag_standard = 0.3
        self.kidney_standard = 0.8

        self.sofa_standard = [0.1,0.0,0.3,0.8]

    def read_table(self,name):
        name = self.file_path + name
        self.patient_table = np.array(pd.read_table(name,sep="|"))
        if 1 in self.patient_table[:, 40]:
            label = 1
            self.sepsis_on_set_time = np.where(self.patient_table[:, 40] == 1)[0][0]
            if self.sepsis_on_set_time < 6:
                self.sepsis_on_set_time = np.shape(self.patient_table)[0]
            if self.sepsis_on_set_time > self.time_sequence:
                self.sepsis_on_set_time = self.time_sequence
                label = 0
        else:
            label = 0
            #self.sepsis_on_set_time = np.shape(self.patient_table)[0]/2
            self.sepsis_on_set_time = np.shape(self.patient_table)[0]
            if self.sepsis_on_set_time > self.time_sequence:
                self.sepsis_on_set_time = self.time_sequence

        self.one_data_logit = label
        self.return_value()



    def read_table_prospective(self,name):
        name = self.file_path + name
        self.patient_table = np.array(pd.read_table(name, sep="|"))
        self.one_data_mask = np.zeros(self.time_sequence)
        self.one_data_logit = np.zeros(self.time_sequence)
        self.one_data_logit_ = self.patient_table[:,40]
        length_time = self.one_data_logit_.shape[0]
        if 1 in self.one_data_logit_:
            self.one_data_logit = np.ones(self.time_sequence)
        if length_time > self.time_sequence:
            self.one_data_logit = self.one_data_logit_[0:self.time_sequence]
            self.one_data_mask = np.ones(self.time_sequence)
        else:
            self.one_data_logit[0:length_time] = self.one_data_logit_
            self.one_data_mask[0:length_time] = np.ones(length_time)

        self.return_value_prospective()

    def return_value_prospective(self):
        self.one_data_tensor = np.zeros((self.time_sequence, 34))
        self.one_data_tensor_origin = np.zeros((self.time_sequence, 34))
        length_data = self.patient_table.shape[0]
        length_final = np.min((length_data,self.time_sequence))
        for i in range(34):
            if self.std_all[i] == 0:
                continue
            else:
                for j in range(length_final):
                    time = j
                    if np.isnan(self.patient_table[time,i]):
                        continue
                    else:
                        self.one_data_tensor[j, i] = \
                            (self.patient_table[time, i] - self.ave_all[i]) / self.std_all[i]
                        self.one_data_tensor_origin[j, i] = self.patient_table[time, i]


    def aquire_data_prospect(self,data):
        length = len(data)
        self.data = np.zeros((length,self.time_sequence,34))
        self.data_origin = np.zeros((length,self.time_sequence,34))
        self.logit = np.zeros((length,self.time_sequence))
        self.mask = np.zeros((length,self.time_sequence))
        for i in range(length):
            name = data[i]
            self.read_table_prospective(name)
            self.data[i,:,:] = self.one_data_tensor
            self.data_origin[i,:,:] = self.one_data_tensor_origin
            self.logit[i,:] = self.one_data_logit
            self.mask[i,:] = self.one_data_mask

        for i in range(length):
            if 0 in self.mask[i,:]:
                index = np.where(self.mask[i,:]==0)[0][0]
                self.data[i,index:,:] = self.data[i,index-1,:]

    def aquire_data(self,data):
        length = len(data)
        self.data = np.zeros((length,self.time_sequence,34))
        self.data_origin = np.zeros((length,self.time_sequence,34))
        self.logit = np.zeros(length)
        self.on_site_time = np.zeros(length)
        self.mask = np.zeros((length,self.time_sequence))
        for i in range(length):
            print(i)
            name = data[i]
            self.read_table(name)
            self.data[i,:,:] = self.one_data_tensor
            self.data_origin[i,:,:] = self.one_data_tensor_origin
            self.logit[i] = self.one_data_logit
            self.on_site_time[i] = self.sepsis_on_set_time


    def store_train_data(self):
        file_path = '/home/tingyi/physionet_data/Interpolate_data/'
        with open(file_path + 'train.npy', 'wb') as f:
            np.save(f,self.data)

        with open(file_path + 'train_origin.npy', 'wb') as f:
            np.save(f,self.data_origin)

        with open(file_path + 'train_logit.npy', 'wb') as f:
            np.save(f,self.logit)

        with open(file_path + 'train_on_site_time.npy', 'wb') as f:
            np.save(f,self.on_site_time)

    def store_val_data(self):
        file_path = '/home/tingyi/physionet_data/Interpolate_data/'
        with open(file_path + 'val.npy', 'wb') as f:
            np.save(f,self.data)

        with open(file_path + 'val_origin.npy', 'wb') as f:
            np.save(f,self.data_origin)

        with open(file_path + 'val_logit.npy', 'wb') as f:
            np.save(f,self.logit)

        with open(file_path + 'val_on_site_time.npy', 'wb') as f:
            np.save(f,self.on_site_time)



    """
    def return_value(self):
        self.one_data_tensor = np.zeros((self.time_sequence, 34))
        self.one_data_tensor_origin = np.zeros((self.time_sequence, 34))
        self.one_data_sofa = np.zeros(4)
        self.one_data_sofa_score = np.zeros(4)
        #self.start_window = np.int(np.floor(self.sepsis_on_set_time - self.time_sequence + 1))
        self.start_window = 0
        if self.start_window < 0:
            self.start_window = 0

        for i in range(34):
            if self.std_all[i] == 0:
                continue
            else:
                for j in range(self.time_sequence):
                    time = j + self.start_window
                    if j + self.start_window > self.patient_table.shape[0]-1:
                        time = self.patient_table.shape[0]-1
                    if np.isnan(self.patient_table[time,i]):
                        continue
                    else:
                        self.one_data_tensor[j,i] = \
                            (self.patient_table[time,i] - self.ave_all[i])/self.std_all[i]
                        self.one_data_tensor_origin[j,i] = self.patient_table[time,i]
    """

    def return_value(self):
        self.one_data_tensor = np.zeros((self.time_sequence, 34))
        self.one_data_tensor_origin = np.zeros((self.time_sequence, 34))
        self.one_data_sofa = np.zeros(4)
        self.one_data_sofa_score = np.zeros(4)
        #self.start_window = np.int(np.floor(self.sepsis_on_set_time - self.time_sequence + 1))
        self.start_window = 0

        if self.start_window < 0:
            self.start_window = 0

        for i in range(34):
            if self.std_all[i] == 0:
                continue
            else:
                for j in range(self.time_sequence):
                    time = j + self.start_window
                    if j + self.start_window > self.patient_table.shape[0] - 1:
                        time = self.patient_table.shape[0] - 1
                    if np.isnan(self.patient_table[time, i]):
                        continue
                    else:
                        self.one_data_tensor[j, i] = \
                            (self.patient_table[time, i] - self.ave_all[i]) / self.std_all[i]
                        self.one_data_tensor_origin[j, i] = self.patient_table[time, i]

        for i in range(34):
            self.check_i = i
            zero_first = np.where(self.one_data_tensor[:,i]==0)[0]
            zero_num = zero_first.shape[0]
            self.check_zero_num = zero_num
            self.check_zero_first = zero_first
            if zero_num == self.time_sequence:
                continue
            elif zero_num == 0:
                continue
            elif zero_first[0] == 0:
                non_zero_first = np.where(self.one_data_tensor[:,i]!=0)[0][0]
                self.one_data_tensor[0,i] = self.one_data_tensor[non_zero_first,i]
                a = pd.Series(self.one_data_tensor[:,i])
                a.replace(0,np.NaN,inplace=True)
                self.one_data_tensor[:,i] = a.interpolate()
            else:
                a = pd.Series(self.one_data_tensor[:,i])
                a.replace(0, np.NaN, inplace=True)
                self.one_data_tensor[:,i] = a.interpolate()

        for i in range(34):
            self.check_i = i
            zero_first = np.where(self.one_data_tensor_origin[:,i]==0)[0]
            zero_num = zero_first.shape[0]
            self.check_zero_num = zero_num
            self.check_zero_first = zero_first
            if zero_num == self.time_sequence:
                continue
            elif zero_num == 0:
                continue
            elif zero_first[0] == 0:
                non_zero_first = np.where(self.one_data_tensor_origin[:,i]!=0)[0][0]
                self.one_data_tensor_origin[0,i] = self.one_data_tensor_origin[non_zero_first,i]
                a = pd.Series(self.one_data_tensor_origin[:,i])
                a.replace(0,np.NaN,inplace=True)
                self.one_data_tensor_origin[:,i] = a.interpolate()
            else:
                a = pd.Series(self.one_data_tensor_origin[:,i])
                a.replace(0, np.NaN, inplace=True)
                self.one_data_tensor_origin[:,i] = a.interpolate()



        """
        self.one_data_sofa[0] = np.mean(self.one_data_tensor_origin[:,self.cardiovas_index])
        self.one_data_sofa[1] = np.mean(self.one_data_tensor_origin[:,self.liver_index])
        self.one_data_sofa[2] = np.mean(self.one_data_tensor_origin[:,self.coagulation_index])
        self.one_data_sofa[3] = np.mean(self.one_data_tensor_origin[:,self.kidney_index])

        if self.one_data_sofa[0] > 70 or self.one_data_sofa[0] == 70:
            self.one_data_sofa_score[0] = 0
        else:
            self.one_data_sofa_score[0] = 1

        if self.one_data_sofa[1]< 1.2:
            self.one_data_sofa_score[1] = 0
        elif self.one_data_sofa[1] == 1.2 or self.one_data_sofa[1] >1.2 and self.one_data_sofa[1]<1.9:
            self.one_data_sofa_score[1] = 1
        elif self.one_data_sofa[1] == 1.9 or self.one_data_sofa[1] > 1.9 and self.one_data_sofa[1] <5.9:
            self.one_data_sofa_score[1] = 2
        elif self.one_data_sofa[1] == 5.9 or self.one_data_sofa[1] >5.9 and self.one_data_sofa[1] <11.9:
            self.one_data_sofa_score[1] = 3
        else:
            self.one_data_sofa_score[1] = 4

        if self.one_data_sofa[2]>150 or self.one_data_sofa[2]==150:
            self.one_data_sofa_score[2]=0
        elif self.one_data_sofa[2]<150 and self.one_data_sofa[2]>100:
            self.one_data_sofa_score[2] = 1
        elif self.one_data_sofa[2] == 100 or self.one_data_sofa[2]<100 and self.one_data_sofa[2]>50:
            self.one_data_sofa_score[2] = 2
        elif self.one_data_sofa[2] == 50 or self.one_data_sofa[2]<50 and self.one_data_sofa[2]>20:
            self.one_data_sofa_score[2] = 3
        else:
            self.one_data_sofa_score[2] = 4

        if self.one_data_sofa[3] < 1.2:
            self.one_data_sofa_score[3] = 0
        elif self.one_data_sofa[3] == 1.2 or self.one_data_sofa[3]>1.2 and self.one_data_sofa[3]<1.9:
            self.one_data_sofa_score[3] = 1
        elif self.one_data_sofa[3] == 1.9 or self.one_data_sofa[3]>1.9 and self.one_data_sofa[3]<3.4:
            self.one_data_sofa_score[3] = 2
        elif self.one_data_sofa[3] == 3.4 or self.one_data_sofa[3]>3.4 and self.one_data_sofa[3]<4.9:
            self.one_data_sofa_score[3] = 3
        else:
            self.one_data_sofa_score[3] = 4
        """

    def compute_missing(self):
        self.missingness = np.zeros(34)
        for i in range(34):
            missing_ratio = (np.where(np.isnan(self.patient_table[:,i]))[0].shape[0])/self.patient_table.shape[0]
            self.missingness[i] = missing_ratio


    def compute_missingness_all(self):
        self.missingness_all_ = np.zeros(34)
        self.index_missing = 0
        for i in self.train_data:
            self.read_table(i)
            self.compute_missing()
            self.missingness_all_ += self.missingness
            self.index_missing += 1
        self.missingness_all = self.missingness_all_/self.index_missing


    def compute_ave(self):
        self.single_ave_value = np.zeros(40)
        for i in range(40):
            single_column = [j for j in self.patient_table[:,i] if not np.isnan(j)]
            if single_column == []:
                continue
            else:
                ave = np.average(single_column)
                self.single_ave_value[i] = ave

    def compute_all_ave(self):
        self.index_count = np.zeros(40)
        self.ave_all_ = np.zeros(40)
        for i in self.file_names:
            #name = self.file_path + i
            self.read_table(i)
            self.compute_ave()
            self.ave_all_ += self.single_ave_value
            for j in range(40):
                if not self.single_ave_value[j] == 0:
                    self.index_count[j] += 1

        self.ave_all = self.ave_all_/self.index_count


    def generate_lib(self):
        count = 0
        for i in self.file_names:
            if count > self.total_size:
                break
            name = self.file_path+i
            patient_table = np.array(pd.read_table(name, sep="|"))

            if 1 in patient_table[:,40]:
                sepsis_on_set_time = np.where(patient_table[:, 40] == 1)[0][0]
                if sepsis_on_set_time < 5:
                    continue
                else:
                    self.sepsis_group.append(i)
                    self.total_data.append(i)
                    self.total_data_label.append(1)
            else:
                self.non_sepsis_group.append(i)
                self.total_data.append(i)
                self.total_data_label.append(0)

            if patient_table[0,35] == 0:
                self.female_group.append(i)
                self.total_gender_label.append(0)
            else:
                self.male_group.append(i)
                self.total_gender_label.append(1)

            for j in range(35):
                entry_mean = np.mean([l for l in patient_table[:, j] if not np.isnan(l)])
                if np.isnan(entry_mean):
                    continue
                self.dic_item.setdefault(j,[]).append(entry_mean)
                if patient_table[0, 35] == 0:
                    self.dic_item_female.setdefault(j,[]).append(entry_mean)
                else:
                    self.dic_item_male.setdefault(j,[]).append(entry_mean)

                if 1 in patient_table[:, 40]:
                    self.dic_item_sepsis.setdefault(j,[]).append(entry_mean)
                else:
                    self.dic_item_non_sepsis.setdefault(j, []).append(entry_mean)



            count += 1


    def compute_ave_vital(self):
        for j in range(35):
            #median = np.median([i for i in self.mean_vital[:,j] if not np.isnan(i)])
            #std = np.std([i for i in self.mean_vital[:,j] if not np.isnan(i)])
            if j in self.dic_item.keys():
                median = np.median(self.dic_item[j])
                std = np.std(self.dic_item[j])
                self.median_vital_signal[j] = median
                self.std_vital_signal[j] =std


    def divide_train_test(self):
        data_length = len(self.total_data)
        self.train_num = np.int(np.floor(data_length*self.train_prop))
        self.train_data = self.total_data[0:self.train_num]
        self.train_data_label = self.total_data_label[0:self.train_num]
        self.train_data_gender_label = self.total_gender_label[0:self.train_num]
        self.test_data = self.total_data[self.train_num:data_length]
        self.test_data_label = self.total_data_label[self.train_num:data_length]
        self.test_data_gender_label = self.total_gender_label[self.train_num:data_length]
        self.train_sepsis_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_label)==1)[0]])
        self.train_non_sepsis_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_label)==0)[0]])
        self.train_female_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_gender_label)==0)[0]])
        self.train_male_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_gender_label)==1)[0]])
        self.test_female_group = list(np.array(self.test_data)[np.where(np.array(self.test_data_gender_label)==0)[0]])
        self.test_male_group = list(np.array(self.test_data)[np.where(np.array(self.test_data_gender_label)==1)[0]])

        self.train_female_group_label = list(
            np.array(self.train_data_label)[np.where(np.array(self.train_data_gender_label)==0)[0]])
        self.train_male_group_label = list(
            np.array(self.train_data_label)[np.where(np.array(self.train_data_gender_label) == 1)[0]])
        self.test_female_group_label = list(
            np.array(self.test_data_label)[np.where(np.array(self.test_data_gender_label) == 0)[0]])
        self.test_male_group_label = list(
            np.array(self.test_data_label)[np.where(np.array(self.test_data_gender_label) == 1)[0]])






if __name__ == "__main__":
    #read_d = read_data()
    #seq = seq_seq_ehr(read_d)
    seq = protatype_ehr(projection,translation)