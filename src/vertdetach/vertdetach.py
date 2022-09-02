import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def vertdetach(x_values, y_values, z_values, temperature_values, accel_freq=75,
               temperature_freq=0.25, std_thresh_mg=8.0, low_temperature_cutoff=26, high_temperature_cutoff=30,
               temp_dec_roc=-0.2, temp_inc_roc=0.1, num_axes=2, border_criteria=False, quiet=False):
    """
    Non-wear algorithm with a 5 minute minimum non-wear duration using absolute temperature, temperature rate of
    change and accelerometer standard deviation of 3 axes to detect start and stop times of non-wear periods.

    Args:
        x_values (numpy array): accelerometer x values
        y_values (numpy array): accelerometer y values
        z_values (numpy array): accelerometer z values
        temperature_values (numpy array): temperature values
        accel_freq (float): frequency of accelerometer in hz
        temperature_freq (float): frequency of temperature sensor in hz
        std_thresh_mg (float): the value which the std of an axis in the window must be below to trigger non-wear
        low_temperature_cutoff (float): Low temperature for non-wear classification (see paper for details)
        high_temperature_cutoff (float): High temperature cut off for wear classification (see paper for details)
        num_axes (int): the number of axes that must be below the std threshold to be considered NW
        quiet (bool): Whether or not to quiet print statements

    Returns:
        A tuple with (start_stop_df, vert_nonwear_array) as defined below

        start_stop_df: A dataframe with the start and end datapoints of the non-wear bouts
        vert_nonwear_array: numpy array with length of the accelerometer data marked as either wear (0) or non-wear (1)
    """
    if not quiet:
        print("Starting Vert Calculation...")

    # Create generic filter script for smoothing temperature
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

    vert_nonwear_array = np.zeros(len(x_values))  # This array will contain 0 for wear and 1 for non-wear
    vert_nonwear_start_datapoints = []
    vert_nonwear_end_datapoints = []

    # mg to g
    std_thresh_g = std_thresh_mg / 1000

    # Create seies of 1 minute rolling std
    x_std_fwd = pd.Series(x_values)[::-1].rolling(round(accel_freq * 60)).std()[::-1]  # 1 minute forward looking STD
    y_std_fwd = pd.Series(y_values)[::-1].rolling(round(accel_freq * 60)).std()[::-1]
    z_std_fwd = pd.Series(z_values)[::-1].rolling(round(accel_freq * 60)).std()[::-1]
    x_std_back = pd.Series(x_values).rolling(round(accel_freq * 60)).std()  # 1 minute backward looking STD
    y_std_back = pd.Series(y_values).rolling(round(accel_freq * 60)).std()
    z_std_back = pd.Series(z_values).rolling(round(accel_freq * 60)).std()

    std_df = pd.DataFrame({"x_std_fwd": x_std_fwd, "y_std_fwd": y_std_fwd, "z_std_fwd": z_std_fwd,
                           "x_std_back": x_std_back, "y_std_back": y_std_back, "z_std_back": z_std_back, })

    # Create Smoothed Temperature array
    smoothed_temperature = filter_signal(temperature_values, 'lowpass', low_f=0.005, sample_f=temperature_freq)
    smoothed_temp_deg_per_min = np.diff(smoothed_temperature, prepend=1) * 60 * temperature_freq

    # Create a constant which converts accel datapoint to temp_datapoint

    # Create DF column counting number of axes below std_thresh
    std_df["Num Axes Fwd"] = np.sum(np.array([std_df["x_std_fwd"] < std_thresh_g,
                                              std_df["y_std_fwd"] < std_thresh_g,
                                              std_df["z_std_fwd"] < std_thresh_g], int), axis=0)

    std_df["Num Axes Backwards"] = np.sum(np.array([std_df["x_std_back"] < std_thresh_g,
                                                    std_df["y_std_back"] < std_thresh_g,
                                                    std_df["z_std_back"] < std_thresh_g], int), axis=0)

    # Find spots where at least num_axes are below the STD threshold for 90% of the next 5 minutes
    std_df["Perc Num Axes >= num_axes for Next 5 Mins (fwd looking)"] = \
        (std_df["Num Axes Fwd"][::-1] >= num_axes).rolling(int(accel_freq * 60 * 5)).mean()[::-1]
    std_df["Perc Num Axes >= num_axes for Next 5 Mins (backward looking)"] = \
        (std_df["Num Axes Backwards"][::-1] >= num_axes).rolling(int(accel_freq * 60 * 5)).mean()[::-1]

    # Make Accelerometer Datapoints have the same number of datapoints as the temperature values
    full_df = std_df[::int(accel_freq / temperature_freq)]
    full_df = full_df[:len(temperature_values)]
    full_df = full_df.reset_index()
    if len(full_df) != len(smoothed_temperature):
        smoothed_temperature = smoothed_temperature[:len(full_df)]
        smoothed_temp_deg_per_min = smoothed_temp_deg_per_min[:len(full_df)]
    full_df["smoothed temperature values"] = smoothed_temperature
    full_df["Max Temp in next five mins"] = full_df["smoothed temperature values"][::-1].rolling(
        int(5 * 60 * temperature_freq)).max()[::-1]
    full_df["Min Temp in next five mins"] = full_df["smoothed temperature values"][::-1].rolling(
        int(5 * 60 * temperature_freq)).min()[::-1]
    full_df["smoothed temperature deg per min"] = smoothed_temp_deg_per_min
    full_df["Mean Five minute Temp Change"] = full_df["smoothed temperature deg per min"][::-1].rolling(int(
        5 * 60 * temperature_freq)).mean()[::-1]

    # Get Candidate NW Start Times
    candidate_nw_starts = np.where((full_df["Num Axes Fwd"] >= num_axes) & (
            full_df["Perc Num Axes >= num_axes for Next 5 Mins (fwd looking)"] >= 0.9))

    # Construct Arrays that will find indexs where a NW bout would end
    # First End Criteria: Rate of Change Path

    end_crit_1 = np.array(np.where((full_df["Num Axes Backwards"] == 0) &
                                   (full_df["Perc Num Axes >= num_axes for Next 5 Mins (backward looking)"] <= 0.50) &
                                   (full_df["Mean Five minute Temp Change"] > temp_inc_roc)))[0]

    # Second End Criteria: Absolute Temperature Path
    end_crit_2 = np.array(np.where((full_df["Num Axes Backwards"] == 0) &
                                   (full_df["Perc Num Axes >= num_axes for Next 5 Mins (backward looking)"] <= 0.50) &
                                   (full_df["Min Temp in next five mins"] > low_temperature_cutoff)))[0]

    end_crit_combined = np.sort(np.unique(np.concatenate((end_crit_1, end_crit_2))))

    # Loop through Candidate NW starts to find bouts
    previous_end = 0
    for ind in candidate_nw_starts[0]:
        if ind < previous_end:
            continue  # Skip if previous bout is already past it

        valid_start = False
        start_ind = int(ind)
        end_ind = int(ind + temperature_freq * 60 * 5)

        # Start Criteria 1: Rate of Change Path
        if (full_df["Max Temp in next five mins"][start_ind] < high_temperature_cutoff) & (
                full_df["Mean Five minute Temp Change"][start_ind] < temp_dec_roc):
            valid_start = True

        # Start Criteria 2: Absolute Temperature Path
        elif full_df["Max Temp in next five mins"][start_ind] < low_temperature_cutoff:
            valid_start = True

        if not valid_start:
            continue

        # If you get to this point its a valid bout, now we find the nearest end_time
        end_crit = end_crit_combined[end_crit_combined > end_ind]

        if (len(end_crit) > 0):
            bout_end_index = end_crit[0]

        else:
            bout_end_index = full_df.last_valid_index()

        accel_start_dp = int(start_ind * accel_freq / temperature_freq)
        accel_end_dp = int(bout_end_index * accel_freq / temperature_freq)
        vert_nonwear_array[accel_start_dp:accel_end_dp] = 1
        vert_nonwear_start_datapoints.append(accel_start_dp)
        vert_nonwear_end_datapoints.append(accel_end_dp)

        previous_end = bout_end_index

    # Vanhees Border Criteria
    if border_criteria:
        df = pd.DataFrame({'NW Vector': vert_nonwear_array,
                           'idx': np.arange(len(vert_nonwear_array)),
                           "Unique Period Key": (pd.Series(vert_nonwear_array).diff(1) != 0).astype('int').cumsum(),
                           'Duration': np.ones(len(vert_nonwear_array))})
        period_durations = df.groupby('Unique Period Key').sum() / (accel_freq * 60)
        period_durations['Wear'] = [False if val == 0 else True for val in period_durations['NW Vector']]
        period_durations['Adjacent Sum'] = period_durations['Duration'].shift(1, fill_value=0) + period_durations[
            'Duration'].shift(-1, fill_value=0)
        period_durations['Period Start'] = df.groupby('Unique Period Key').min()['idx']
        period_durations['Period End'] = df.groupby('Unique Period Key').max()['idx'] + 1
        for index, row in period_durations.iterrows():
            if not row['Wear']:
                if row['Duration'] <= 180:
                    if row['Duration'] / row['Adjacent Sum'] < 0.8:
                        vert_nonwear_array[row['Period Start']:row['Period End']] = True
                elif row['Duration'] <= 360:
                    if row['Duration'] / row['Adjacent Sum'] < 0.3:
                        vert_nonwear_array[row['Period Start']:row['Period End']] = True

        vert_nonwear_start_datapoints = \
        np.where(pd.Series(vert_nonwear_array) - pd.Series(vert_nonwear_array).shift(1) == 1)[
            0]
        if vert_nonwear_array[0] == 1:
            vert_nonwear_start_datapoints = np.insert(vert_nonwear_start_datapoints, 0, 0)
        vert_nonwear_end_datapoints = \
            np.where(pd.Series(vert_nonwear_array) - pd.Series(vert_nonwear_array).shift(1) == -1)[0]
        if vert_nonwear_array[-1] == 1:
            vert_nonwear_end_datapoints = np.append(vert_nonwear_end_datapoints, len(vert_nonwear_array))

    start_stop_df = pd.DataFrame({"Start Datapoint": vert_nonwear_start_datapoints,
                                  "End Datapoint": vert_nonwear_end_datapoints},
                                 index=range(1, len(vert_nonwear_start_datapoints) + 1))
    vert_nonwear_array = np.array(vert_nonwear_array, bool)

    if not quiet:
        print("Finished Vert Calculation.")

    return start_stop_df, vert_nonwear_array
