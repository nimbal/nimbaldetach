# vertdetach 
[![DOI](https://zenodo.org/badge/447634532.svg)](https://zenodo.org/badge/latestdoi/447634532)

DETACH (Device Temperature and Acceleration Change) algorithm detects non-wear periods for body-worn accelerometers 
by integrating a ‘rate-of-change’ criterion for temperature into a combined temperature-acceleration algorithm.

### Publication

Vert, A., Weber, K. S., Thai, V., Turner, E., Beyer, K. B., Cornish, B. F., Godkin, F. E., Wong, C., McIlroy, W. E., 
& Van Ooteghem, K. (2022). Detecting accelerometer non-wear periods using change in acceleration combined with 
rate-of-change in temperature. *BMC Medical Research Methodology, 22*. https://doi.org/10.1186/s12874-022-01633-6

## Using the vertdetach Algorithm
The DETACH algorithm is stored as a function called vertdetach within the /src/vertdetach/vertdetach.py Python file.

The function itself is defined as follows
```python
vertdetach(x_values, y_values, z_values, temperature_values, accel_freq=75,
           temperature_freq=0.25, std_thresh_mg=8.0, low_temperature_cutoff=26, high_temperature_cutoff=30,
           temp_dec_roc=-0.2, temp_inc_roc=0.1, num_axes=2, border_criteria = False, quiet=False)
```

### Input Arguments
The input arguments and their definitions are listed in the table below:

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
| temp_dec_roc            | float       | Optional (-0.2)                      | Temperature decrease rate-of-change threshold for non-wear classification (see paper for more details)  |
| temp_inc_roc            | float       | Optional (0.1)                       | Temperature increase rate-of-change threshold for non-wear classification (see paper for more details)  |
| num_axes                | int         | Optional (2)                         | Number of axes that must be below the STD threshold to be considered non-wear                           |
| border_criteria         | bool        | Optional (False)                     | Determines whether or not to use the non-wear border criteria inspired by van Hees here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061691 <br><br>For additional information, look at this discussion: https://github.com/nimbal/vertdetach/issues/1|
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
start_stop_df, nonwear_array = vertdetach(x_values = x_values, y_values = y_values, z_values = z_values, 
                                          temperature_values = temperature_values, accel_freq = accel_freq, 
                                          temperature_freq = temperature_freq)

# Analysis
total_wear_time = np.sum(nonwear_array)
pct_worn = total_wear_time/len(nonwear_array) * 100

print("The device was not worn %s percent of the time" % pct_worn)

```
## Version/Changelog

v1.0.2
- [Detecting accelerometer non-wear periods using change in acceleration combined with rate-of-change in temperature.](https://doi.org/10.1186/s12874-022-01633-6)

## Installation

To install the latest release of vertdetach directly from GitHub using pip, run the following line in terminal or 
console:

`pip install git+https://github.com/nimbal/vertdetach`

To install a specific release, insert `@v#.#.#` after the repository name replacing with the tag associated with that 
release. For example:

`pip install git+https://github.com/nimbal/vertdetach@v1.0.0`

## Include vertdetach as Package Dependency

To include the latest release of vertdetach as a dependency in your Python package, include the following
string within the list alongside your other dependencies:

`install_requires=['vertdetach@git+https://github.com/nimbal/vertdetach@[version]']`

To include a specific release, replace `[version]` with the tag associated with that release.

## vertdetach Package Dependencies
- numpy
- pandas
- scipy
