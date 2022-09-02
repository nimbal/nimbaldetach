"""
This is a varitation of handds_curate.scripts.nonwear_review where instead of just loading in non-wear data from a csv
I also recalculate the data. This will help test its performancecompared to the most recently saved version
"""

from pathlib import Path
import json
from datetime import timedelta
import numpy as np
import pandas as pd
# from vertdetach import vertdetach
import datetime as dt
import sys
sys.path.insert(1, r'C:\Users\ahvert\PycharmProjects\vertdetach\src')
sys.path.insert(1, r'C:\Users\ahvert\PycharmProjects\nimbalwear\src')
from nimbalwear.data import Device as Data
# from nimbalwear import Data
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from dateutil.parser import ParserError
from dateutil.parser import parse as dparse
from scipy.signal import butter, filtfilt
from helpers import load_in_pipeline_nonwear_df
import os

study_save_dir='W:/NiMBaLWEAR/'
study_code='OND09'
subject_id = '0070' #0060 0065 0067 0069 0070 0078 0080 0092 0115
coll_id = '01'
device_type='AXV6'
device_location='RWrist'

epoch_length = 5
day_offset = 12
overnight_start = 22
overnight_end = 8

high_temp_cutoff = 30
low_temp_cutoff = 26
temp_inc_roc = 0.1
temp_dec_roc = -0.3

fig_size = (18, 9)

def filter_signal(data, filt_type, low_f=None, high_f=None, sample_f=None, filter_order=2):
    # From by Kyle Weber
    """Function that creates bandpass filter to ECG data.
    Required arguments:
    -data: 3-column array with each column containing one accelerometer axis
    -filt_type: "lowpass", "highpass" or "bandpass"
    -low_f, high_f: filter cut-offs, Hz
    -sample_f: sampling frequency, Hz
    -filter_order: order of filter; integer
    """
    nyquist_freq = 0.5 * sample_f
    low = (low_f / nyquist_freq) if low_f is not None else None
    high = (high_f / nyquist_freq) if high_f is not None else None
    if filt_type == 'lowpass':
        wn = low
    elif filt_type == 'highpass':
        wn = high
    elif filt_type == 'bandpass':
        wn = [low, high]
    b, a = butter(N=filter_order, Wn=wn, btype=filt_type)
    filtered_data = filtfilt(b, a, x=data)
    return filtered_data

log_save_dir = 'W:/OND09 (HANDDS-ONT)/Digitized logs'
removal_log_file_name = 'handds_device_removal_log.xlsx'

log_save_dir = Path(log_save_dir)
removal_log_path = log_save_dir / removal_log_file_name

# Read metadata files
study_save_dir = Path(study_save_dir)
study_dir = study_save_dir / study_code
settings_path = study_dir / 'pipeline/settings/settings.json'
devices_csv_path = study_dir / 'pipeline/devices.csv'

with open(settings_path, 'r') as f:
    settings_json = json.load(f)

dirs = settings_json['pipeline']['dirs']
dirs = {key: study_dir / value for key, value in dirs.items()}

devices_csv = pd.read_csv(devices_csv_path, dtype=str)

device_location = devices_csv['device_location'][(devices_csv['study_code'] == study_code) &
                                                 (devices_csv['subject_id'] == subject_id) &
                                                 (devices_csv['coll_id'] == coll_id) &
                                                 (devices_csv['device_type'] == device_type) &
                                                 (devices_csv['device_location'] == device_location)].item()

# Concatenate data file paths
device_edf_path = (dirs['device_edf_standard']
                   / ("_".join([study_code, subject_id, coll_id, device_type, device_location]) + ".edf"))

nonwear_csv_path = (dirs['nonwear_bouts_standard']
                    / ("_".join([study_code, subject_id, coll_id, device_type, device_location, "NONWEAR"])
                       + ".csv"))


# read device data
device = Data()
device.import_edf(device_edf_path)

print("Reading event and log files.")

# read event files
nonwear_old = pd.read_csv(nonwear_csv_path, dtype=str)

nonwear = load_in_pipeline_nonwear_df(device, study_code, subject_id, coll_id, device_type, device_location, low_temperature_cutoff=low_temp_cutoff,high_temperature_cutoff=high_temp_cutoff)

# Read log files
removal_log = pd.read_excel(removal_log_path, dtype=str)


############################
# Format and combine events
############################

events = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time', 'end_time',
                               'details', 'notes'])

# nonwear

nonwear['event'] = 'nonwear'
nonwear.rename(columns={'nonwear_bout_id': 'id'}, inplace=True)
nonwear['start_time'] = pd.to_datetime(nonwear['start_time'], format='%Y-%m-%d %H:%M:%S')
nonwear['end_time'] = pd.to_datetime(nonwear['end_time'], format='%Y-%m-%d %H:%M:%S')
nonwear['details'] = ['_'.join([row['device_type'], row['device_location']]) for index, row in nonwear.iterrows()]
nonwear.drop(columns=['device_type', 'device_location'], inplace=True)
nonwear.reset_index(drop=True, inplace=True)

events = pd.concat([events, nonwear], ignore_index=True)


# device removal
removal_log = removal_log[(removal_log['SUBJECT'] == subject_id) & (
            (removal_log['SENSOR_RW'] == 'yes') | (removal_log['SENSOR_LW'] == 'yes'))]
removal_log.reset_index(inplace=True, drop=True)

removal_log['COLL_ID'] = removal_log['COLL_ID'].str.zfill(2)

time_cols = ['TIME_REMOVED', 'TIME_REATTACHED']

for index, row in removal_log.iterrows():
    for col in time_cols:
        try:
            row[col] = dparse(row[col], fuzzy=True)
        except (ParserError, TypeError):
            row[col] = np.datetime64('NaT')


device_removal = removal_log[['STUDY_CODE', 'SUBJECT', 'COLL_ID', 'TIME_REMOVED', 'TIME_REATTACHED',
                              'REASON_REMOVAL']].copy()
device_removal['event'] = 'removal'
device_removal['id'] = range(1, len(device_removal['event']) + 1)
device_removal = device_removal.rename(columns={'STUDY_CODE': 'study_code', 'SUBJECT': 'subject_id',
                                                'COLL_ID': 'coll_id','TIME_REMOVED': 'start_time',
                                                'TIME_REATTACHED': 'end_time', 'REASON_REMOVAL': 'details'})

events = pd.concat([events, device_removal], ignore_index=True)

# sort events
events.sort_values(by='start_time', inplace=True)
events.reset_index(inplace=True, drop=True)


night_device = device
x_ind = night_device.get_signal_index('Accelerometer x')
y_ind = night_device.get_signal_index('Accelerometer y')
z_ind = night_device.get_signal_index('Accelerometer z')
temp_ind = night_device.get_signal_index('Temperature')
x = night_device.signals[x_ind]
y = night_device.signals[y_ind]
z = night_device.signals[z_ind]
temp = night_device.signals[temp_ind]
start_datetime = night_device.header['start_datetime']
accel_sample_rate = night_device.signal_headers[x_ind]['sample_rate']
temp_sample_rate = night_device.signal_headers[temp_ind]['sample_rate']

accel = [x, y, z]

vm = np.sqrt(np.square(accel).sum(axis=0)) - 1
#vm[vm < 0] = 0

# # get start and sample rate
# start_datetime = night_device.header['start_datetime']
# accel_sample_rate = night_device.signal_headers[x_ind]['sample_rate']
# temp_sample_rate = night_device.signal_headers[temp_ind]['sample_rate']
end_datetime = start_datetime + timedelta(seconds = (len(vm))/accel_sample_rate)

accel_times = pd.date_range(start_datetime, end_datetime, periods = len(vm))

smoothed_temperature = filter_signal(temp, 'lowpass', low_f=0.005, sample_f=temp_sample_rate)
smoothed_temp_deg_per_min = np.diff(smoothed_temperature, prepend=1) * 60 * temp_sample_rate
# temp_change_idx = pd.date_range(start_datetime, periods=len(smoothed_temp_deg_per_min), freq=f'{1 / temp_sample_rate}S')
temp_change = pd.Series(smoothed_temp_deg_per_min[::-1]).rolling(int(5 * 60 * temp_sample_rate)).mean()[::-1]
temp_times = pd.date_range(start_datetime, end_datetime, periods = len(temp_change))
# temp_change.index = temp_change_idx

fig, ax = plt.subplots(2, 3, sharex='all', figsize=fig_size)

ax[0][0].plot_date(accel_times, vm, fmt='', linewidth=0.25, color='black')
ax[0][0].set_title('Accelerometer Vector Magnitude')
ax[0][1].plot_date(temp_times, temp, fmt='', linewidth=0.25, color= 'black')
ax[0][1].set_title('Temperature Absolute Value')
ax[0][2].plot_date(temp_times, temp_change, fmt='', linewidth=0.25, color= 'black')
ax[0][2].set_title('Temperature 5 minute mean change')
ax[1][0].plot_date(accel_times, vm, fmt='', linewidth=0.25, color='black')
ax[1][1].plot_date(temp_times, temp, fmt='', linewidth=0.25, color= 'black')
ax[1][2].plot_date(temp_times, temp_change, fmt='', linewidth=0.25, color= 'black')

for index, row in nonwear.iterrows():

    row['end_time'] = row['start_time'] if pd.isnull(row['end_time']) else row['end_time']
    row['start_time'] = row['end_time'] if pd.isnull(row['start_time']) else row['start_time']

    ax[1][0].axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=1, alpha=1, color='grey')
    ax[1][1].axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=1, alpha=1, color='grey')
    ax[1][2].axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=1, alpha=1, color='grey')

for index, row in nonwear_old.iterrows():

    row['end_time'] = row['start_time'] if pd.isnull(row['end_time']) else row['end_time']
    row['start_time'] = row['end_time'] if pd.isnull(row['start_time']) else row['start_time']

    ax[0][0].axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=1, alpha=1, color='grey')
    ax[0][1].axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=1, alpha=1, color='grey')
    ax[0][2].axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=1, alpha=1, color='grey')

ax[0][1].axhline(high_temp_cutoff, color='red', linewidth=0.25, linestyle='-')
ax[0][1].axhline(low_temp_cutoff, color='blue', linewidth=0.25, linestyle='-')
ax[0][2].axhline(temp_inc_roc, color='red', linewidth=0.25, linestyle='-')
ax[0][2].axhline(temp_dec_roc, color='blue', linewidth=0.25, linestyle='-')

ax[1][1].axhline(high_temp_cutoff, color='red', linewidth=0.25, linestyle='-')
ax[1][1].axhline(low_temp_cutoff, color='blue', linewidth=0.25, linestyle='-')
ax[1][2].axhline(temp_inc_roc, color='red', linewidth=0.25, linestyle='-')
ax[1][2].axhline(temp_dec_roc, color='blue', linewidth=0.25, linestyle='-')
ax[0][0].set_ylabel('ORIGINAL DETACH NONWEAR')
ax[1][0].set_ylabel('BOUTS COMBINED DETACH NONWEAR')
fig.tight_layout()
file_name = "_".join([study_code,subject_id,coll_id,device_type,device_location,'COMBINED_BOUTS.png'])
plt.savefig(os.path.join(r'images\combine_bouts_figs',file_name))
plt.show()
x = 1


############## OTHER USEFUL PLOTS ###################
# temp_array = np.array(temp)
#
# plt.figure()
# plt.title(subject_id)
# plt.hist(temp_array, bins=range(0,40), density=True)
# plt.show()
#
# #
# # ncols = int(night_device.signal_headers[temp_ind]['sample_rate'] * 60)
# # nrows = int(np.ceil(temp_array.shape[0] / ncols))
# # temp_mins = np.append(temp_array, np.zeros(nrows * ncols - temp_array.shape[0]))
# # temp_mins = temp_mins.reshape([nrows, ncols])
# #
# # temp_mins = temp_mins.mean(axis=1)
# #
# # temp_slope = np.gradient(temp_mins)
# temp_diff = np.diff(temp_array, 75) / 75 # int(night_device.signal_headers[temp_ind]['sample_rate'] * 60 * 3))
#
# a = temp_array[:-225]
# b = temp_array[225:]
#
# temp_diff = (b - a) / 3
#
# plt.figure()
# plt.title(subject_id)
# plt.plot(temp_diff)
# plt.show()
#
# plt.figure()
# plt.title(subject_id)
# plt.hist(temp_diff, bins=np.arange(-1.2, 1.2, 0.1), density=True)
# plt.show()