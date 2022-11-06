# from Subject import *
import sys
sys.path.insert(1, r'C:\Users\ahvert\PycharmProjects\nimbalwear-dev\src')
from nimbalwear.data import Device
import datetime as dt
import numpy as np
import pandas as pd
import os
from sklearn import tree
import bisect
from sklearn import metrics
from sklearn import model_selection
###########
"""
To creat .npy files for each OND06 participants accelerometer and temperature sensors for 6 days starting at 2pm on the
first day
"""
# ids = [2530]
master_x_values = []
master_y_values = []
master_z_values = []
master_x_std_values = []
master_y_std_values = []
master_z_std_values = []
master_temperature_values = []
master_labels = []

edf_path = "W:\Annotated nonwear\OND09"
edf_list = os.listdir(edf_path)

location = "Chest"

# Parse Out the unique ids and loop through them
ids = [file.split("_")[1] for file in edf_list]

for id in ids:
    device = Device()
    print("Starting %s..." % id)
    device.import_edf(os.path.join(edf_path, "OND09_%s_01_BF36_%s.edf" % (id, location)))

    orig_start_time = device.header['start_datetime']
    new_start_time = pd.Timestamp(device.header['start_datetime']).replace(minute = 0, hour = 14, second = 0)
    end_time = orig_start_time + dt.timedelta(seconds=len(device.signals[device.get_signal_index('Accelerometer x')]) / device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate'])
    accelerometer_timestamps = np.asarray(pd.date_range(orig_start_time, end_time, periods=len(device.signals[device.get_signal_index('Accelerometer x')])))

    accel_start_time_dp = np.where(accelerometer_timestamps >= new_start_time)[0]
    if len(accel_start_time_dp) == 0:
        accel_start_time_dp = 0
    else:
        accel_start_time_dp = accel_start_time_dp[0]
    accel_end_dp = int(accel_start_time_dp + device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate'] * 60 * 60 * 24 * 6) # 6 Day collection
    accel_values = np.array([accelerometer_timestamps[accel_start_time_dp:accel_end_dp],
                             device.signals[device.get_signal_index('Accelerometer x')][accel_start_time_dp:accel_end_dp],
                             device.signals[device.get_signal_index('Accelerometer y')][accel_start_time_dp:accel_end_dp],
                             device.signals[device.get_signal_index('Accelerometer z')][accel_start_time_dp:accel_end_dp]], float)


    temperature_timestamps = np.asarray(pd.date_range(orig_start_time, end_time, periods=len(device.signals[device.get_signal_index('Temperature')])))
    temp_start_time_dp = np.where(temperature_timestamps >= new_start_time)[0]
    if len(temp_start_time_dp) == 0:
        temp_start_time_dp = 0
    else:
        temp_start_time_dp = temp_start_time_dp[0]
    temp_end_dp = int(temp_start_time_dp + device.signal_headers[device.get_signal_index('Temperature')]['sample_rate'] * 60 * 60 * 24 * 6)
    temp_values = np.array([temperature_timestamps[temp_start_time_dp:temp_end_dp],
                            device.signals[device.get_signal_index('Temperature')][temp_start_time_dp:temp_end_dp]],float)

    np.save(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files\OND09_%s_%s_accel_Oct24' % (id,location), accel_values,allow_pickle = True)
    np.save(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files\OND09_%s_%s_temperature_Oct24' % (id,location), temp_values, allow_pickle = True)
    # read in GS nonwear
    gs_path = "W:\Annotated nonwear\OND09_VisuallyInspectedNonwear.xlsx"
    gs_df = pd.read_excel(gs_path)
    gs_df["start_time"] = pd.to_datetime(gs_df["start_timestamp"])
    gs_df["end_time"] = pd.to_datetime(gs_df["end_timestamp"])

    gs_parsed_df = gs_df.loc[gs_df["subject_id"].astype(int) == int(id)]

    dp_6days = int(device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate'] * 60 * 60 * 24 * 6)
    gs_nonwear_vector = np.zeros(dp_6days)

    new_start_time = pd.Timestamp(accelerometer_timestamps[0]).replace(minute = 0, hour = 14, second = 0)
    start_time_dp = int((new_start_time - pd.Timestamp(accelerometer_timestamps[0])).total_seconds()*device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate'])
    if start_time_dp <= 0:
        start_time_dp = 0
    new_end_time = new_start_time + pd.Timedelta(days=6)

    accel_timestamps = accelerometer_timestamps[start_time_dp:start_time_dp+dp_6days]

    for index, row in gs_parsed_df.iterrows():



        # Find indexs
        start_index = bisect.bisect_right(accel_timestamps, row["start_timestamp"])
        start_index_10 = int(start_index + device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate']*60*10)
        end_index = bisect.bisect_right(accel_timestamps, row["end_timestamp"])
        end_index_10 = int(end_index + device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate']*60*10)

        # If the start time is when the file ends, skip it since the EDF has been cropped before that
        if start_index == len(accel_timestamps):
            continue

        if end_index == 0:
            continue

        if start_index != 0:
            gs_nonwear_vector[start_index:start_index_10] =  1 # 1 = NW Start
        gs_nonwear_vector[end_index:end_index_10] = 2  # 2 = NW End
        gs_nonwear_vector[start_index_10:end_index] = 3  # 3 = NW Middle
    x_std = pd.Series(device.signals[device.get_signal_index('Accelerometer x')]).rolling(int(device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate']*60)).std()[start_time_dp:start_time_dp+dp_6days:25]
    y_std = pd.Series(device.signals[device.get_signal_index('Accelerometer y')]).rolling(int(device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate']*60)).std()[start_time_dp:start_time_dp+dp_6days:25]
    z_std = pd.Series(device.signals[device.get_signal_index('Accelerometer z')]).rolling(int(device.signal_headers[device.get_signal_index('Accelerometer x')]['sample_rate']*60)).std()[start_time_dp:start_time_dp+dp_6days:25]
    x_values = device.signals[device.get_signal_index('Accelerometer x')][start_time_dp:start_time_dp+dp_6days:25]
    y_values = device.signals[device.get_signal_index('Accelerometer y')][start_time_dp:start_time_dp+dp_6days:25]
    z_values = device.signals[device.get_signal_index('Accelerometer z')][start_time_dp:start_time_dp+dp_6days:25]


    temperature_values = device.signals[device.get_signal_index('Temperature')][int(start_time_dp/25):int((start_time_dp+dp_6days)/25)]

    labels = gs_nonwear_vector[::25] # Wear = 0, NW Start = 1, NW End = 2, NW Middle = 3
    if len(labels) != len(x_values):
        print("????")
        x = 1
    master_x_values.extend(x_values)
    master_y_values.extend(y_values)
    master_z_values.extend(z_values)
    master_x_std_values.extend(x_std)
    master_y_std_values.extend(y_std)
    master_z_std_values.extend(z_std)
    master_temperature_values.extend(temperature_values)
    master_labels.extend(labels)
    feature_vectors = np.stack((master_x_values, master_y_values, master_z_values, master_x_std_values,master_y_std_values,master_z_std_values, master_temperature_values), axis=1)
    np.save(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files/OND09_COMBINED_FEATURE_VECTOR_Chest_Oct15.npy', feature_vectors)
    np.save(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files/OND09_COMBINED_LABELS_Chest_Oct15.npy', master_labels)

feature_vectors = np.stack((master_x_values, master_y_values, master_z_values, master_x_std_values,master_y_std_values,master_z_std_values, master_temperature_values), axis=1)
np.save(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files/OND09_COMBINED_FEATURE_VECTOR_Chest_Oct15.npy',feature_vectors)
np.save(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files/OND09_COMBINED_LABELS_Chest_Oct15.npy',master_labels)
