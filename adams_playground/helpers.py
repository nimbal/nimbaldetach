import sys
sys.path.insert(1, r'C:\Users\ahvert\PycharmProjects\vertdetach\src')
from vertdetach import vertdetach as vertdetach
import datetime as dt

def load_in_pipeline_nonwear_df(data_object,study_code, subject_id, coll_id, device_type, device_location, low_temperature_cutoff=26, high_temperature_cutoff=30):
    night_device = data_object
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

    nonwear_times, nonwear_array = vertdetach(
        x_values=x,
        y_values=y,
        z_values=z,
        temperature_values=temp,
        accel_freq=accel_sample_rate,
        temperature_freq=temp_sample_rate)
    # low_temperature_cutoff=low_temperature_cutoff,
    # high_temperature_cutoff=high_temperature_cutoff,
    # temp_dec_roc=temp_dec_roc,
    # temp_inc_roc=temp_inc_roc,
    # quiet=quiet)
    algorithm_name = 'DETACH'

    nonwear_times['nonwear_bout_id'] = nonwear_times.index
    nonwear_times.rename(columns={'Start Datapoint': 'start_datapoint', 'End Datapoint': 'end_datapoint'},
                         inplace=True)

    bout_count = nonwear_times.shape[0]
    # convert datapoints to times

    start_times = []
    end_times = []

    for nw_index, nw_row in nonwear_times.iterrows():
        start_times.append(start_datetime + dt.timedelta(seconds=(nw_row['start_datapoint'] / accel_sample_rate)))
        end_times.append(start_datetime + dt.timedelta(seconds=(nw_row['end_datapoint'] / accel_sample_rate)))

    nonwear_times['start_time'] = start_times
    nonwear_times['end_time'] = end_times

    # add study_code
    nonwear_times['study_code'] = study_code
    nonwear_times['subject_id'] = subject_id
    nonwear_times['coll_id'] = coll_id
    nonwear_times['device_type'] = device_type
    nonwear_times['device_location'] = device_location

    # reorder columns
    nonwear = nonwear_times[['study_code', 'subject_id', 'coll_id', 'device_type', 'device_location',
                             'nonwear_bout_id', 'start_time', 'end_time']]

    return nonwear
