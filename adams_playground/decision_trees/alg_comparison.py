# Adam Vert
# March 14, 2021

# ======================================== IMPORTS ========================================
from matplotlib.patches import Patch
# from Nonwear.Statistical_Outputs_for_Paper.GoodCopyPaperStuff.Algorithms_Good_Copy import *
import bisect
from sklearn import metrics
import seaborn as sns
import os
import pyedflib
from Subject import *
import datetime as dt
import numpy as np
import pandas as pd
import sys
from nimbalwear.nonwear import vanhees_nonwear, zhou_nonwear
from Nonwear.Nonwear_Main import vert_nonwear

# ======================================== CODE ==========================================
"""
To create an excel file with minute by minute results of each algorithm
"""


numpy_files_list = os.listdir(r"C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files")
master_array = np.array([[],[],[],[],[],[]])
vert_nw_dfs = pd.DataFrame({"PATIENT NUM": [],
                            "DEVICE LOCATION": [],
                            "NW Start Time": [],
                            "NW End Time": [],
                            "NW Duration": [],
                            "TP":[],
                            "FP":[]})
zhou_nw_dfs = pd.DataFrame({"PATIENT NUM": [],
                            "DEVICE LOCATION": [],
                            "NW Start Time": [],
                            "NW End Time": [],
                            "NW Duration": [],
                            "TP":[],
                            "FP":[]})
vh_nw_dfs = pd.DataFrame({"PATIENT NUM": [],
                          "DEVICE LOCATION": [],
                          "NW Start Time": [],
                          "NW End Time": [],
                          "NW Duration": [],
                            "TP":[],
                            "FP":[]})
gs_nw_dfs = pd.DataFrame({"PATIENT NUM": [],
                          "DEVICE LOCATION": [],
                          "NW Start Time": [],
                          "NW End Time": [],
                          "NW Duration": [],
                          "Vert Bout Perc": [],
                          "Zhou Bout Perc": [],
                          "Vanhees Bout Perc": []})

vert_start_paths_df = pd.DataFrame({"PATIENT NUM": [],
                                    "Decreasing Temperature Start Path": [],
                                    "Low Temperature Start Path": [],
                                    "Both Criterias Start Path":[]})
vert_end_paths_df = pd.DataFrame({"PATIENT NUM": [],
                                    "High Temperature End Path": [],
                                    "Both Criterias Met End Path": [],
                                    "End of File End Path":[]})

# Make list of only Chest files
location = "Chest"
chest_list = [file for file in numpy_files_list if location in file]

# Parse Out the unique ids and loop through them
ids = [file.split("_")[1] for file in chest_list if len(file.split("_")[1]) == 4]
ids  = np.unique(ids)
for id in ids:
    print('starting %s...' % id)
    # Read in 6 day .npy files
    accel_file = np.load(os.path.join(r"C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files", 'OND09_%s_%s_accel_Oct24.npy' % (id,location)))
    temp_file = np.load(os.path.join(r"C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files", 'OND09_%s_%s_temperature_Oct24.npy' % (id,location)))

    accelerometer_timestamps = pd.to_datetime(accel_file[0])
    x_values = accel_file[1]
    y_values = accel_file[2]
    z_values = accel_file[3]
    temperature_timestamps = pd.to_datetime(temp_file[0])
    temperature_values = temp_file[1]

    if id == '5919':
        new_end = dt.datetime(2019, 10, 12, 6, 59)
        temp_inds = np.where(temperature_timestamps < new_end)
        temperature_values = temperature_values[temp_inds]
        temperature_timestamps = temperature_timestamps[temp_inds]
        accel_inds = np.where(accelerometer_timestamps < new_end)
        x_values = x_values[accel_inds]
        y_values = y_values[accel_inds]
        z_values = z_values[accel_inds]
        accelerometer_timestamps = accelerometer_timestamps[accel_inds]
    if id == '2707':
        new_end = dt.datetime(2019, 7, 2, 12, 3)
        temp_inds = np.where(temperature_timestamps < new_end)
        temperature_values = temperature_values[temp_inds]
        temperature_timestamps = temperature_timestamps[temp_inds]
        accel_inds = np.where(accelerometer_timestamps < new_end)
        x_values = x_values[accel_inds]
        y_values = y_values[accel_inds]
        z_values = z_values[accel_inds]
        accelerometer_timestamps = accelerometer_timestamps[accel_inds]

    # Get todays date
    month = dt.datetime.today().strftime('%b').upper()
    day = dt.datetime.today().strftime('%d')

    # Perform the three different non-wear calculations
    vert_nonwear_start_stop, vert_nonwear_array, vert_start_paths, vert_start_paths_list, vert_end_paths, vert_end_paths_list = vert_nonwear(x_values, y_values, z_values, temperature_values, accel_freq = 25, temperature_freq=1, return_path_percentiles=True,
                                                                                                                                             temp_inc_roc=0.05, temp_dec_roc = -0.1, low_temperature_cutoff = 25, std_thresh_mg = 5.0)
    vanhees_nonwear_array = vanhees_nonwear(x_values,y_values,z_values, freq = 25)
    zhou_nonwear_array = zhou_nonwear(x_values,y_values,z_values,temperature_values, accelerometer_frequency = 25, temperature_frequency = 1)

    # read in GS nonwear
    gs_path = r"W:\Annotated nonwear\OND09_VisuallyInspectedNonwear.xlsx"
    gs_df = pd.read_excel(gs_path, dtype = {'subject_id':str})
    gs_df["start_time"] = pd.to_datetime(gs_df["start_timestamp"]) +pd.Timedelta(seconds = 30)
    gs_df["end_time"] = pd.to_datetime(gs_df["end_timestamp"]) + pd.Timedelta(seconds = 30)

    # Parse to relevant part
    gs_parsed_df = gs_df.loc[gs_df["subject_id"].astype(str) == str(id)]



    # Make a vector for the GS nonwear the same length as the accelerometer values
    gs_nonwear_vector = np.zeros(len(x_values), dtype = bool)

    vert_bout_perc = []
    vanhees_bout_perc = []
    zhou_bout_perc = []
    # loop through each gold standard non-wear row
    for index, row in gs_parsed_df.iterrows():
        # Find indexs
        start_index = bisect.bisect_right(accelerometer_timestamps, row["start_time"])
        end_index = bisect.bisect_right(accelerometer_timestamps, row["end_time"])

        # If the start time is when the file ends, skip it since the EDF has been cropped before that
        if start_index == len(accelerometer_timestamps):
            continue
        if end_index == 0:
            continue

        gs_nonwear_vector[start_index:end_index] = True

        vert_bout_perc.append(vert_nonwear_array[start_index:end_index].sum()/len(vert_nonwear_array[start_index:end_index]))
        vanhees_bout_perc.append(vanhees_nonwear_array[start_index:end_index].sum()/len(vanhees_nonwear_array[start_index:end_index]))
        zhou_bout_perc.append(
            zhou_nonwear_array[start_index:end_index].sum() / len(zhou_nonwear_array[start_index:end_index]))
    # append to master_array with all the  predictions and change to 1hz as that is still below the resoloution of any alg
    master_array = np.append(master_array,[[id]*len(gs_nonwear_vector[::25]),accelerometer_timestamps[::25],gs_nonwear_vector[::25],vert_nonwear_array[::25],zhou_nonwear_array[::25],vanhees_nonwear_array[::25]],1)
    # np.save(r'E:\Backup\D\PycharmProjects\OndriAtHome\Nonwear\Statistical_Outputs_for_Paper\GoodCopyPaperStuff\.npy Files\OND06_LWrist Master File GS VERT ZHOU HEES %s%s' % (month,day), master_array,allow_pickle = True)

    # Make start and stop time dataframes for non-wear
    def nw_df_maker(nonwear_vector):
        nw_starts_index = np.where(((nonwear_vector == True) & (np.roll(nonwear_vector, 1) == False)))
        nw_starts = accelerometer_timestamps[nw_starts_index]
        nw_ends_index = np.where(((nonwear_vector == True) & (np.roll(nonwear_vector, -1) == False)))
        nw_ends = accelerometer_timestamps[nw_ends_index]
        nw_duration = nw_ends - nw_starts

        nw_df = pd.DataFrame({"PATIENT NUM": id,
                              "DEVICE LOCATION": location,
                              "NW Start Time": nw_starts,
                              "NW End Time": nw_ends,
                              "NW Duration": nw_duration})
        return nw_df


    # Make individual non-wear figure
    vert_nonwear_df = nw_df_maker(vert_nonwear_array)
    vert_nonwear_df["NW Duration"] = np.round(vert_nonwear_df["NW Duration"].dt.total_seconds())
    vert_nonwear_df["Start Path Taken"] = vert_start_paths_list
    vert_nonwear_df["End Path Taken"] = vert_end_paths_list
    zhou_nonwear_df = nw_df_maker(zhou_nonwear_array)
    zhou_nonwear_df["NW Duration"] = np.round(zhou_nonwear_df["NW Duration"].dt.total_seconds())
    vanhees_nonwear_df = nw_df_maker(vanhees_nonwear_array)
    vanhees_nonwear_df["NW Duration"] = np.round(vanhees_nonwear_df["NW Duration"].dt.total_seconds())
    gs_nonwear_df = nw_df_maker(gs_nonwear_vector)
    gs_nonwear_df["NW Duration"] = np.round(gs_nonwear_df["NW Duration"].dt.total_seconds())
    gs_nonwear_df["Vert Bout Perc"] = vert_bout_perc
    gs_nonwear_df["Zhou Bout Perc"] = zhou_bout_perc
    gs_nonwear_df["Vanhees Bout Perc"] = vanhees_bout_perc

    temp_arr = [[id] * len(gs_nonwear_vector[::25]), accelerometer_timestamps[::25], gs_nonwear_vector[::25],
                vert_nonwear_array[::25], zhou_nonwear_array[::25], vanhees_nonwear_array[::25]]
    temp_df = pd.DataFrame(np.transpose(temp_arr),
                           columns=["PATIENT NUM", "Time Stamp", "Gold Standard Non-wear", "vert Non-wear",
                                    "zhou Non-wear", "vanhees Non-wear"])
    temp_df['Time Stamp'] = pd.to_datetime(temp_df['Time Stamp'])
    for name, df in zip(['vert', 'zhou', 'vanhees'], [vert_nonwear_df, zhou_nonwear_df, vanhees_nonwear_df]):
        fp_list = []
        tp_list = []
        for ind, row in df.iterrows():
            start_bout = row['NW Start Time']
            end_bout = row['NW End Time']
            cropped_df = temp_df.loc[(temp_df['Time Stamp'] >= start_bout) & (temp_df['Time Stamp'] <= end_bout)]
            tn, fp, fn, tp = metrics.confusion_matrix(cropped_df['Gold Standard Non-wear'].astype(bool),
                                                      cropped_df['%s Non-wear' % name].astype('bool'),
                                                      labels=[False, True]).ravel()
            if (fn != 0) or (tn != 0):
                raise ValueError
            fp_list.append(fp)
            tp_list.append(tp)
        df['TP'] = tp_list
        df['FP'] = fp_list
        exec("%s_nonwear_df = df" % name)


    vert_nw_dfs = pd.concat([vert_nw_dfs, vert_nonwear_df])
    zhou_nw_dfs = pd.concat([zhou_nw_dfs, zhou_nonwear_df])
    vh_nw_dfs = pd.concat([vh_nw_dfs, vanhees_nonwear_df])
    gs_nw_dfs = pd.concat([gs_nw_dfs, gs_nonwear_df])

    vert_start_paths["PATIENT NUM"] = [id]
    vert_start_paths_df = pd.concat([vert_start_paths_df, vert_start_paths])
    vert_end_paths["PATIENT NUM"] = [id]
    vert_end_paths_df = pd.concat([vert_end_paths_df, vert_end_paths])


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


    smoothed_temp = filter_signal(temperature_values, 'lowpass', low_f=0.005,
                                  sample_f=0.25)
    fig, ax = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [3, 5, 5, 5]})

    fig.suptitle("GENEACTIV NONWEAR SNAPSHOT\nParticipant: " + str(id) + "    Location: " + str(
        'Chest'))

    rolling_std = pd.Series(x_values)[::-1].rolling(60 * 25).std()[::-1]
    ax[1].plot(accelerometer_timestamps[::25 * 60], rolling_std[::25 * 60])
    ax[1].set_title("std for x-axis (g)")
    ax[2].plot(accelerometer_timestamps[::25], x_values[::25])
    ax[2].set_title("Accelerometer Values (g)")
    ax[3].plot(temperature_timestamps, smoothed_temp)
    ax[3].set_title("Smoothed Temperature Values (deg. Celcius)")
    for axis in range(4):
        for gs_index, gs_row in gs_parsed_df.iterrows():
            if gs_row['start_time'] < accelerometer_timestamps[0]:
                gs_row['start_time'] = pd.Timestamp(accelerometer_timestamps[0])
            if gs_row['start_time'] > accelerometer_timestamps[-1]:
                continue
            if gs_row['end_time'] > accelerometer_timestamps[-1]:
                gs_row['end_time'] = pd.Timestamp(accelerometer_timestamps[-1])
            if gs_row['end_time'] < accelerometer_timestamps[0]:
                continue
            ax[axis].axvspan(gs_row['start_time'], gs_row['end_time'], ymin=0, ymax=0.25, color='green',
                             alpha=0.5, zorder=1)

        for alg_index, alg_row in vert_nonwear_df.iterrows():
            ax[axis].axvspan(alg_row["NW Start Time"], alg_row["NW End Time"], ymin=0.25, ymax=0.5, color='red',
                             alpha=0.5, zorder=1)

        for alg_index, alg_row in zhou_nonwear_df.iterrows():
            ax[axis].axvspan(alg_row["NW Start Time"], alg_row["NW End Time"], ymin=0.5, ymax=0.75, color='purple',
                             alpha=0.5, zorder=1)

        for alg_index, alg_row in vanhees_nonwear_df.iterrows():
            ax[axis].axvspan(alg_row["NW Start Time"], alg_row["NW End Time"], ymin=0.75, ymax=1.0, color='orange',
                             alpha=0.5, zorder=1)

    custom_legend = [Patch(facecolor="green", edgecolor="green", alpha=1, label="Gold Standard NW Times"),
                     Patch(facecolor="red", edgecolor="red", alpha=1, label="Vert Detected NW Times"),
                     Patch(facecolor="purple", edgecolor="purple", alpha=1, label="Zhou Detected NW Times"),
                     Patch(facecolor="Orange", edgecolor="Orange", alpha=1, label="Vanhees Detected NW Times"),
                     ]

    fig.legend(handles=custom_legend)
    fig.set_size_inches(19.2, 9.8)
    fig.savefig(
        r"C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\figs\%s_%s_individual_compare_graphs_%s_%s.png" % (
            id, 'chest', month, day))
    print(id, 'plot saved')
    plt.close()
    x = 1
all_algs_mins_df = pd.DataFrame(np.transpose(master_array)[::60], columns = ["PATIENT NUM", "Time Stamp", "Gold Standard Non-wear","Vert Non-wear","Zhou Non-wear","Vanhees Non-wear"])
all_algs_mins_df["Time Stamp"] = pd.to_datetime(all_algs_mins_df["Time Stamp"])
np.save(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\npy_files\OND09_Chest Master File GS VERT ZHOU HEES %s%s' % (month, day), master_array,allow_pickle = True)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(r'C:\Users\ahvert\PycharmProjects\vertdetach\adams_playground\decision_trees\csvs\NW Bout Comparisons %s%s.xlsx' % (month,day), engine='xlsxwriter')

# Write each dataframe to a different worksheet.
all_algs_mins_df.to_excel(writer, sheet_name="All Algs By Minute",index = False)
gs_nw_dfs.to_excel(writer, sheet_name='GS', index = False)
vert_nw_dfs.to_excel(writer, sheet_name='Vert', index = False)
zhou_nw_dfs.to_excel(writer, sheet_name='Zhou', index = False)
vh_nw_dfs.to_excel(writer, sheet_name='Vanhees', index = False)
vert_start_paths_df.to_excel(writer, sheet_name= 'Vert Start Paths', index = False)
vert_end_paths_df.to_excel(writer, sheet_name='Vert End Paths', index = False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

x = 1
