import matplotlib.pyplot as plt
from Subject import *
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, lfilter, filtfilt
from sklearn import tree
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score
###########
"""
Script used to create decision trees
"""


feature_vectors = np.load(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files\OND09_COMBINED_FEATURE_VECTOR_Chest_Oct15.npy')
labels = np.load(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files\OND09_COMBINED_LABELS_Chest_Oct15.npy')

# Use  First 10 participants (i.e. training data) for decision trees
decision_tree_features_train = feature_vectors[:10*60*60*24*6]
decision_tree_labels_train = labels[:10*60*60*24*6]

# use remaining 4 participants for testing
decision_tree_features_test = feature_vectors[10*60*60*24*6:]
decision_tree_labels_test = labels[10*60*60*24*6:]


def filter_signal(data, type, low_f=None, high_f=None, sample_f=None, filter_order=2):
    # From Kyle Weber
    """Function that creates bandpass filter to ECG data.
    Required arguments:
    -data: 3-column array with each column containing one accelerometer axis
    -type: "lowpass", "highpass" or "bandpass"
    -low_f, high_f: filter cut-offs, Hz
    -sample_f: sampling frequency, Hz
    -filter_order: order of filter; integer
    """
    nyquist_freq = 0.5 * sample_f
    if type == "lowpass":
        low = low_f / nyquist_freq
        b, a = butter(N=filter_order, Wn=low, btype="lowpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)
    if type == "highpass":
        high = high_f / nyquist_freq
        b, a = butter(N=filter_order, Wn=high, btype="highpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)
    if type == "bandpass":
        low = low_f / nyquist_freq
        high = high_f / nyquist_freq
        b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)
    return filtered_data


df_train = pd.DataFrame(decision_tree_features_train, columns=['x_values','y_values','z_values','x_std','y_std','z_std','temperature_values'])
df_test = pd.DataFrame(decision_tree_features_test, columns=['x_values','y_values','z_values','x_std','y_std','z_std','temperature_values'])

# Smooth temperature using an order 2 lowpass filter with a values of 0.0005 (visual inspection)
smoothed_temp_train = filter_signal(df_train["temperature_values"], 'lowpass', low_f=0.005,
                              sample_f=1)
smoothed_temp_test = filter_signal(df_test["temperature_values"], 'lowpass', low_f=0.005,
                              sample_f=1)

df_train["temperature_values"] = smoothed_temp_train
df_test["temperature_values"] = smoothed_temp_test

df_train["min_change_in_temp"] = df_train["temperature_values"] - df_train["temperature_values"].shift(15)
df_test["min_change_in_temp"] = df_test["temperature_values"] - df_test["temperature_values"].shift(15)

# df["x_std"] = df["x_values"].rolling(15).std()
# df["y_std"] = df["y_values"].rolling(15).std()
# df["z_std"] = df["z_values"].rolling(15).std()

df_train = df_train.replace(np.nan, 0)
df_test = df_test.replace(np.nan, 0)
# df_validation = df_validation.replace(np.nan, 0)
# df = df.replace(2,0)

labels_wear_and_start_train = decision_tree_labels_train[np.where((decision_tree_labels_train ==0)|(decision_tree_labels_train == 1))]
labels_wear_and_start_test = decision_tree_labels_test[np.where((decision_tree_labels_test ==0)|(decision_tree_labels_test == 1))]

df_wear_and_start_train = df_train.iloc[np.where((decision_tree_labels_train == 0)|(decision_tree_labels_train == 1))]
df_wear_and_start_test = df_test.iloc[np.where((decision_tree_labels_test == 0)|(decision_tree_labels_test == 1))]

feature_train = df_wear_and_start_train[['temperature_values','min_change_in_temp','x_std','y_std','z_std']]
feature_test = df_wear_and_start_test[['temperature_values','min_change_in_temp','x_std','y_std','z_std']]

label_train = labels_wear_and_start_train
label_test = labels_wear_and_start_test

# Wear and NW Start
fig_start, ax_start = plt.subplots(figsize=[25.6, 14.4])
clf_train = tree.DecisionTreeClassifier(max_depth=3)
clf_train.fit(feature_train, label_train)
tree.plot_tree(clf_train, feature_names = ['temperature_values','min_change_in_temp','x_std','y_std','z_std'],class_names=['Wear','NW Start'], fontsize = 15)
fig_start.suptitle("Classifying Wear and Non-wear Start for OND09 Chest Data")
plt.savefig(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\figs\OND09 Chest NW Start Decision Tree.png')
plt.close()

# NW End and NW
labels_nw_and_end_train = decision_tree_labels_train[np.where((decision_tree_labels_train ==2)|(decision_tree_labels_train == 3))]
labels_nw_and_end_test = decision_tree_labels_test[np.where((decision_tree_labels_test ==2)|(decision_tree_labels_test == 3))]

df_nw_and_end_train = df_train.iloc[np.where((decision_tree_labels_train ==2)|(decision_tree_labels_train == 3))]
df_nw_and_end_test = df_test.iloc[np.where((decision_tree_labels_test ==2)|(decision_tree_labels_test == 3))]

feature_train = df_nw_and_end_train[['temperature_values','min_change_in_temp','x_std','y_std','z_std']]
feature_test = df_nw_and_end_test[['temperature_values','min_change_in_temp','x_std','y_std','z_std']]

label_train = labels_nw_and_end_train
label_test = labels_nw_and_end_test

fig_end, ax_start = plt.subplots(figsize=[25.6, 14.4])
clf1_train = tree.DecisionTreeClassifier(max_depth=4)
clf1_train.fit(feature_train, label_train)
tree.plot_tree(clf1_train, feature_names = ['temperature_values','min_change_in_temp','x_std','y_std','z_std'],class_names=['NW End','NW'], fontsize = 10)
fig_end.suptitle("Classifying NW and NW End")
plt.savefig(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\figs\OND09 Chest NW End Decision Tree.png')
plt.close()