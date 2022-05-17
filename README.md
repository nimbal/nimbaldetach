# vertdetach
DETACH (Device Temperature and Acceleration Change) algorithm for detecting non-wear from raw accelerometer data as
discussed in the paper **A novel method to detect periods of non-wear for body-worn inertial measurement units** which
can be accessed at [doi.org/10.1186/s12874-022-01633-6]()


## Using the vertdetach Algorithm
The DETACH algorithm is stored as a function called vertdetach within the /src/vertdetach/vertdetach.py Python file.

The function itself is defined as follows
```python
vertdetach(x_values, y_values, z_values, temperature_values, accel_freq=75,
               temperature_freq=0.25, std_thresh_mg=8.0, low_temperature_cutoff=26, high_temperature_cutoff=30,
               temp_dec_roc=-0.2, temp_inc_roc=0.1, num_axes=2, quiet=False)
```

### Input Arguments
The input arguments and there definitions are listed in the table below

| Argument                | Data Type   | Optional (default value) or Required | Description                                                                                             |
|-------------------------|-------------|--------------------------------------|---------------------------------------------------------------------------------------------------------|
| x_values                | NumPy Array | Required                             | x-axis accelerometer values                                                                             |
| y_values                | NumPy Array | Required                             | y-axis accelerometer values                                                                             |
| z_values                | NumPy Array | Required                             | x-axis accelerometer values                                                                             |
| temperature_values      | NumPy Array | Required                             | Temperature values                                                                                      |
| accel_freq              | float       | Optional (75)                        | Frequency of the accelerometer in Hz                                                                    |
| temperature_freq        | float       | Optional (0.25)                      | Frequency of the temperature in Hz                                                                      |
| std_thresh_mg           | float       | Optional (8.0)                       | The value which the standard deviation (STD) of an axis in the window must be below to trigger non-wear |
| low_temperature_cutoff  | float       | Optional (26.0)                      | Low temperature threshold for non-wear classification (see paper for more details)                      |
| high_temperature_cutoff | float       | Optional (30.0)                      | High temperature threshold for non-wear classification (see paper for more details)                     |
| num_axes                | int         | Optional (2)                         | Number of axes that must be below the STD threshold to be considered non-wear                           |
| quiet                   | bool        | Optional (False)                     | Whether or not to quiet print statements.                                                               |
### Returns
The algorithm returns a tuple with two objects:
1. start_stop_df: A pandas DataFrame with the start and end datapoints of the non-wear bouts.
2. vert_nonwear_array: A numpy array with the same length of the accelerometer data marked as either wear (0) or non-wear (1)

### Example
Example python code to determine the percentage of non-wear time from 
```python
from src.vertdetach.vertdetach import vertdetach
import numpy as np

# Load Data
accelerometer_vals = np.load("path\to\raw\accelerometer.npy")
temperature_values = np.load("path\to\raw\temperature.npy")
x_values = accelerometer_vals[0]
y_values = accelerometer_vals[1]
z_values = accelerometer_vals[2]

# Define Frequencies
accel_freq = 75
temperature_freq = 0.25

# Calculate Non-wear
start_stop_df, nonwear_array = vertdetach(x_values = x_values, y_values = y_values, z_values = z_values, temperature_values = temperature_values,
                                               accel_freq = accel_freq, temperature_freq = temperature_freq)

# Analysis
total_wear_time = np.sum(nonwear_array)
pct_worn = total_wear_time/len(nonwear_array) * 100

print("The device was not worn of %s percent of the time" % pct_worn)

```
## Version/Changelog

v0.1.1
- testing zenodo

## Installation

## Package Dependencies
- numpy
- pandas
- scipy
