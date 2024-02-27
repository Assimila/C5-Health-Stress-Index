import xarray as xr
import metview as mv
import numpy as np
import matplotlib.pyplot as plt
import thermofeel as tf
import glob
from rasterstats import zonal_stats
from affine import Affine
import pandas as pd
import shapefile
import datetime as dt
import os
import matplotlib.dates as mdates


########################################################################################################################
# Functions
########################################################################################################################
def roundPartial(numerator, denominator):
    """
    Rounds one value (numerator) to the nearest multiple of another value (denominator).
    :param numerator:          int or float  Number to be rounded to closest multiple of another number.
    :param denominator:        int or float  numerator will be rounded to a multiple of this number.
    :return closest_multiple:  int or float  Closest multiple to value of interest (numerator).
    """
    quotient = round(numerator / denominator)
    closest_multiple = quotient * denominator

    return (closest_multiple)


def seq(start, stop, step, n_decs):
    """
    Create arithmetic sequence of numbers between a start and end value (has a constant difference between each step).
    :param start:      float           Start value of sequence (first value of sequence will be exactly this value).
    :param stop:       float           End boundary of sequence. May not be exact value of end of sequence.
    :param step:       float           Size of difference between each step in the sequence.
    :param n_decs:     integer         Set number of decimal places for each value in sequence.
    :return sequence:  list of floats  Arithmetic sequence of numbers between start and stop with difference between
                                       consecutive numbers equal to "step".
    """
    # Calculate number of steps in sequence
    num_of_steps = int(round((stop - start) / float(step)))

    # Calculate sequence
    if num_of_steps > 1:
        sequence = np.round([start + step * i for i in range(num_of_steps + 1)], n_decs)
    elif num_of_steps == 1:
        sequence = np.round([start], n_decs)
    else:
        sequence = []

    return (sequence)


def create_gridline_vals(coordinates, resolution, dec_num):
    """
    Calculate custom gridline locations between 2 locations that are multiples of desired resolution.
    :param coordinates:         list of floats  Two or more ordered coordinate values. Lowest first (south if latitude -
                                                west if longitude),.
    :param resolution:          float           Resolution of required gridline locations.
    :param dec_num:             float           Number of decimal places relevant to gridline locations.
    :return all_gridline_vals:  1D numpy array  Array of gridline locations.
    """
    # Calculate minimum and maximum gridline locations. Will be closest multiples of resolution value to coordinate
    # values.
    min_gridline = roundPartial(coordinates[0], resolution)
    max_gridline = roundPartial(coordinates[-1], resolution)

    # Add a buffer to min and max gridlines, so gridline locations definitely cover entire area to be mapped.
    buffer_min_gridline = min_gridline - resolution
    buffer_max_gridline = max_gridline + resolution

    # Calculate sequence of values between min and max buffer gridline locations, with steps equal resolution.
    all_gridline_vals = seq(buffer_min_gridline, buffer_max_gridline, resolution, dec_num)

    return (all_gridline_vals)


def read_shapefile(sf_shape):
    """
    Read a shapefile into a Pandas dataframe with a 'coords'
    column holding the geometry information. This uses the pyshp
    package
    """

    fields = [x[0] for x in sf_shape.fields][1:]
    records = [y[:] for y in sf_shape.records()]
    shps = [s.points for s in sf_shape.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df


def get_resolution_from_xarray(x_array):
    """
    Method to create a tuple (x resolution, y resolution) in x and y
    dimensions.

    :param x_array: X-array with latitude and longitude variables.

    :return: tuple with x and y resolutions
    """

    x_res = x_array.longitude.values[1] - x_array.longitude.values[0]
    y_res = x_array.latitude.values[1] - x_array.latitude.values[0]

    return (x_res, y_res)


def get_zonal_stats(datasets_list, vector_path):
    """
    Get statistics of a given dataset (xarray) for every feature
    in a vector (shapefile)
    :param datasets_list: list of xarray dataset or lists of xarrays
    :param vector_path: full path of a shapefile
    :return: xarray with statistics
    """

    # Output list
    stats = []

    # Dataset is a list containing one of more xarray datasets
    for _dataset in datasets_list:
        for data_var in _dataset.data_vars:
            # Create geo_transform - perhaps the user requested a
            # spatial subset, therefore is compulsory to update it

            # GeoTransform -- case of a "north up" image without
            #                 any rotation or shearing
            #  GeoTransform[0] top left x
            #  GeoTransform[1] w-e pixel resolution
            #  GeoTransform[2] 0
            #  GeoTransform[3] top left y
            #  GeoTransform[4] 0
            #  GeoTransform[5] n-s pixel resolution (negative value)

            _xarray = getattr(_dataset, data_var)
            x_res, y_res = get_resolution_from_xarray(_xarray)

            gt = (_xarray.longitude.data[0] - (x_res / 2.),
                  x_res, 0.0,
                  _xarray.latitude.data[0] - (y_res / 2.),
                  0.0, y_res)

            affine = Affine.from_gdal(*gt)

            tmp_stats_list = []

            # For each time step...
            for i in range(_xarray.data.shape[0]):
                data = _xarray[i].data

                tmp_stats = zonal_stats(vector_path, data,
                                        affine=affine,
                                        nodata=np.nan,
                                        geojson_out=True,
                                        stats=['count', 'min', 'max',
                                               'mean', 'sum'])
                # raster_out=True),
                # all_touched=True)

                # Add the date for each feature in the shapefile
                for j, feature in enumerate(tmp_stats):
                    feature['properties']['date'] = \
                        str(_xarray.time.data[i])

                    tmp_stats_list.append(feature['properties'])

        # Convert ordered dictionary to DataFrame
        stats.append(pd.DataFrame.from_dict(tmp_stats_list))

    return stats


def plot_heat_index(hi_df, title_loc, save_plot, plot_fpath=None):
    """
    Plot heat index and heat risk category thresholds.
    :param hi_df:      pd.Dataframe.  Dataframe containing 'date' column (in datetime format) and column called 'mean'
                                      containing heat index in degrees celsius.
    :param title_loc:  string.        Title (location of data)
    :param save_plot:  boolean.       Whether plot should be saved.
    :param plot_fpath: string.        Optional file path to where plot should be saved (only needed if save_plot = True)
    """
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(hi_df['date'].values, hi_df['mean'].values, color='black')
    ax.tick_params(axis='x', labelsize=18, rotation=45)
    ax.tick_params(axis='y', labelsize=18)

    # Make ticks on occurrences of each day
    ax.xaxis.set_major_locator(mdates.DayLocator())

    # Plot major and minor x grid lines
    ax.grid(axis='x', which='both', ls='--')

    plt.xlabel('Time', fontsize=30)
    plt.ylabel('Heat Index', fontsize=30)
    plt.title(title_loc, fontsize=30)

    # Plot boundaries of heat risk categories
    ax.axhline(26.7, color='black', alpha=0.7, linestyle='--')
    ax.axhline(32.2, color='black', alpha=0.7, linestyle='--')
    ax.axhline(39.4, color='black', alpha=0.7, linestyle='--')
    ax.axhline(51.1, color='black', alpha=0.7, linestyle='--')

    # Add colour to heat risk categories
    if np.min(hi_df['mean'].values) < 18:
        ax.axhspan(np.min(hi_df['mean'].values) - 1, 26.7, facecolor='green', alpha=0.5)
    else:
        ax.axhspan(18, 26.7, facecolor='green', alpha=0.5)
    ax.axhspan(26.7, 32.2, facecolor='yellow', alpha=0.5)
    ax.axhspan(32.2, 39.4, facecolor='orange', alpha=0.5)
    ax.axhspan(39.4, 51.1, facecolor='red', alpha=0.5)
    ax.axhspan(51.1, 53, facecolor='maroon', alpha=0.5)

    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_fpath)

    plt.close()


def plot_airtemp(t2m_dataframe, title_loc, save_plot, hline_true, plot_fpath=None, hline_val=None):
    """
    Plot air temperature with optional threshold line (e.g. 38 C heat wave threshold).
    :param t2m_dataframe:  pd.Dataframe.  Dataframe containing 'date' column (in datetime format) and column called 'mean'
                                          containing 2m air temperature in degrees celsius.
    :param title_loc:      string.        Title (location of data)
    :param save_plot:      boolean.       Whether plot should be saved.
    :param hline_true:     boolean.       Whether to plot a horizontal threshold line on the graph
    :param plot_fpath:     string.        Optional file path to where plot should be saved (only needed if save_plot =
                                          True)
    :param hline_val:      float.         Value of horizontal threshold line on graph (only needed if hline_true = True).
    """

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(t2m_dataframe['date'].values, t2m_dataframe['mean'].values, color='black')
    ax.tick_params(axis='x', labelsize=18, rotation=45)
    ax.tick_params(axis='y', labelsize=18)

    # Make ticks on occurrences of each day
    ax.xaxis.set_major_locator(mdates.DayLocator())

    # Plot major and minor x grid lines
    ax.grid(axis='x', which='both', ls='--')

    # Add dashed line showing heatwave threshold
    if hline_true:
        ax.axhline(hline_val, color='black', alpha=0.7, linestyle='--')

    plt.xlabel('Time', fontsize=30)
    plt.ylabel('Air temperature (\N{degree sign}C)', fontsize=30)
    plt.title(title_loc, fontsize=30)

    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_fpath)

    plt.close()


def apply_heat_index_thresholds(array):
    """
    Convert heat index array to risk categories based on thresholds.
    :param array:  pd.Series. Column of pandas dataframe containing heat index values (floats or integers).
    """
    array = array.copy()
    array.loc[array < 26.7] = 1
    array.loc[(array >= 26.7) & (array < 32.2)] = 2
    array.loc[(array >= 32.2) & (array < 39.4)] = 3
    array.loc[(array >= 39.4) & (array < 51.1)] = 4
    array.loc[(array >= 51.1)] = 5

    return array


def calculate_daily_summary_stats(df_hi, df_t2m):
    """
    Calculate daily minimums and maximums for air temperature, heat index and heat risk category.
    :param df_hi:         pd.Dataframe.  Zonal stats heat index output for single district - date column must be converted
                                         to datetime.
    :param df_t2m:        pd.Dataframe.  Zonal stats air temperature output for single district - date column must be
                                         converted to datetime.

    :return daily_stats:  pd.Dataframe.  Dataframe containing daily minimums and maximums.
    """
    #########################################################################################################
    # Heat index daily calculations
    #########################################################################################################
    df_hi['date'] = pd.to_datetime(df_hi['date'], errors='coerce')
    new = df_hi[['ADM2_EN', 'date', 'mean']].set_index('date')

    # Maximum
    daily_max = new.resample('D').max()
    daily_hi_max = daily_max.rename(columns={"mean": "Maximum Heat Index", "ADM2_EN": "District"})
    daily_hi_max.reset_index(inplace=True)

    # Minimum
    daily_min = new.resample('D').min()
    daily_hi_min = daily_min.rename(columns={"mean": "Minimum Heat Index", "ADM2_EN": "District"})
    daily_hi_min.reset_index(inplace=True)

    # Heat risk categories
    daily_hi_max['Maximum Heat Risk Category'] = daily_hi_max["Maximum Heat Index"]
    daily_hi_max['Maximum Heat Risk Category'] = apply_heat_index_thresholds(daily_hi_max["Maximum Heat Risk Category"])
    daily_hi_min['Minimum Heat Risk Category'] = daily_hi_min["Minimum Heat Index"]
    daily_hi_min['Minimum Heat Risk Category'] = apply_heat_index_thresholds(daily_hi_min["Minimum Heat Risk Category"])

    #########################################################################################################
    # 2m Air temperature daily calculations
    #########################################################################################################
    df_t2m['date'] = pd.to_datetime(df_t2m['date'], errors='coerce')
    new = df_t2m[['ADM2_EN', 'date', 'mean']].set_index('date')

    # Maximum
    daily_max = new.resample('D').max()
    daily_t2m_max = daily_max.rename(columns={"mean": "Maximum Air Temperature (C)", "ADM2_EN": "District"})
    daily_t2m_max.reset_index(inplace=True)

    # Minimum
    daily_min = new.resample('D').min()
    daily_t2m_min = daily_min.rename(columns={"mean": "Minimum Air Temperature (C)", "ADM2_EN": "District"})
    daily_t2m_min.reset_index(inplace=True)

    #########################################################################################################
    # Merge dataframes together
    daily_stats = daily_hi_min.merge(daily_hi_max)
    daily_stats = daily_stats.merge(daily_t2m_min)
    daily_stats = daily_stats.merge(daily_t2m_max)

    # Reorder columns
    cols = ['date', 'District', 'Minimum Heat Index', 'Maximum Heat Index',
            'Minimum Heat Risk Category', 'Maximum Heat Risk Category',
            'Minimum Air Temperature (C)', 'Maximum Air Temperature (C)']
    daily_stats = daily_stats[cols]

    return daily_stats


def get_compass_direction(coord_value, lat_or_lon):
    global compass_coord
    if lat_or_lon == 'lat':
        if coord_value >= 0:
            compass_coord = f'{coord_value}N'
        else:
            compass_coord = f'{abs(coord_value)}S'
    if lat_or_lon == 'lon':
        if coord_value >= 0:
            compass_coord = f'{coord_value}E'
        else:
            compass_coord = f'{abs(coord_value)}W'
    return compass_coord


def make_output_filename(parameters, atm_level, n_lat, s_lat, w_lon, e_lon, timestep, fcast_date, fcast_time):
    param_string = '_'.join(parameters)

    n_string = get_compass_direction(n_lat, 'lat')
    s_string = get_compass_direction(s_lat, 'lat')
    w_string = get_compass_direction(w_lon, 'lon')
    e_string = get_compass_direction(e_lon, 'lon')

    fcast_filename = f"gfs_{param_string}_{atm_level}_{n_string}_{s_string}_{w_string}_{e_string}_{fcast_date}T{fcast_time:02d}_step_{timestep:03d}.grib"

    return fcast_filename


def open_multiple_grib_file_single_variable(grib_fpath_list, var_name):
    for file_name in grib_fpath_list:

        tmp = mv.read(file_name)[var_name].to_dataset()

        if file_name == grib_fpath_list[0]:
            new_dat = tmp.copy(deep=True)
        else:
            new_dat = xr.concat([new_dat, tmp], dim='valid_time')

    return new_dat


def make_complete_days_df(df_with_datetime_cols, expected_no_of_timesteps):
    """
    Make a dataframe containing only the dates that have the number of timesteps that would be expected if
    all the hours of a day are represented.
    :param df_with_datetime_cols:     pandas.Dataframe.  Dataframe with date columns in datetime format.
    :param expected_no_of_timesteps:  integer.           Number of timesteps that represent a 24 hour period. 24 for hourly
                                                         data, 8 for 3 hourly, etc.
    :return complete_days_df:         pandas.Dataframe.  Dataframe containing dates and timestep counts.
    """

    # Count number of values available for each day
    date_counts = df_with_datetime_cols['date'].dt.date.value_counts().rename_axis('dates').reset_index(
        name='timestep_counts')

    # Make a dataframe of which dates have expected number of timesteps that represent a whole day
    complete_days_df = date_counts[date_counts['timestep_counts'] == expected_no_of_timesteps]

    return complete_days_df


def make_complete_timesteps_column(hourly_df, num_of_timesteps_in_day):
    """
    Add a column to a dataframe (extracted from the BMD json files) which shows whether each timestep is part of a
    day in which all the timesteps are present that would be expected if an entire day was present.
    :param hourly_df:                pandas.Dataframe.  Dataframe of single variable (e.g. air temperature) from BMD json
                                                        output.
    :param num_of_timesteps_in_day:  integer.           Number of timesteps that represent a 24 hour period. 24 for hourly
                                                        data, 8 for 3 hourly, etc.
    :return output_df:               pandas.Dataframe.  Dataframe containing an extra column
    """
    # Convert time columns from string to datetime
    hourly_df['date'] = pd.to_datetime(hourly_df['date'], errors='coerce')

    # Make a dataframe of which dates have expected number of timesteps that represent a whole day (assumes
    # all subdistricts have the same number of timesteps)
    complete_days = make_complete_days_df(hourly_df, num_of_timesteps_in_day)

    # Make column stating whether measurement is part of a day when all timesteps are available
    hourly_df['complete_days'] = hourly_df['date'].dt.date.isin(complete_days['dates'].tolist())

    # Rename dataframe
    output_df = hourly_df.copy()

    return output_df


def convert_to_only_complete_days(hourly_df, expected_timestep_no):
    """
    Drop timesteps that are from days when not every timestep is present.
    :param hourly_df:             pandas.Dataframe.  Dataframe containing forecast timestep column.
    :param expected_timestep_no:  int.               Number of timesteps that represent a 24 hour period. 24 for hourly
                                                     data, 8 for 3 hourly, etc.
    :return output_df:            pandas.Dataframe.  Same as input dataframe, except it now has an extra column called
                                                     "complete_days", and all timesteps from dates without a complete
                                                     set of forecast steps are dropped.
    """
    # Remove incomplete days from temperature data
    temp = make_complete_timesteps_column(hourly_df, expected_timestep_no)
    output_df = temp[temp['complete_days'] == True]

    return output_df


def make_0p05_res_lat_lons(n_lat, s_lat, w_lon, e_lon):
    """
    Makes numpy arrays representing latitude and longitude arrays (with 0.05 degree grid spacings) within specified
    ranges.
    :param n_lat:           float.                 Northern latitude.
    :param s_lat:           float.                 Southern latitude.
    :param w_lon:           float.                 Western longitude.
    :param e_lon:           float.                 Eastern longitude.
    :return lat_lons_0p05:  list of numpy arrays.  Two numpy arrays, representing latitude values and longitude
                                                   values with grid spacings of 0.05 degrees between n_lat-s_lat and
                                                   w_lon-e_lon.
    """
    # Calculate new latitudes
    lats_new = np.arange(n_lat, (s_lat - 0.01), -0.05)
    # Calculate new longitudes
    lons_new = np.arange(w_lon, (e_lon + 0.01), 0.05)
    # Make list containing new lats and lons
    lat_lons_0p05 = [lats_new, lons_new]
    # Return list
    return lat_lons_0p05


def make_plots_and_csv(date: str, timestep: int, grib_paths: str, shp_path: str, output_csv_dir: str,
                       output_plt_dir: str, make_output_dir: bool, save_plots: bool, save_csv: bool):
    """
    Makes and optionally saves summary statistics and plots of heat conditions for three districts of Bangladesh
    (Chuadanga, Kushtia and Meherpur) from GFS forecast data.
    :param date:             string.   Initialisation date of forecast (format 'yyyy-mm-dd').
    :param timestep:         integer.  Initialisation time of forecast in UTC.
    :param grib_paths:       string.   Path to forecast grib files.
    :param shp_path:         string.   Path to shapefile with Bangladesh administrative boundaries (level 2).
    :param output_csv_dir:   string.   Path to where output csv will be stored.
    :param output_plt_dir:   string.   Path to where output png files will be stored.
    :param make_output_dir:  boolean.  Whether subdirectories for output data should be created if not already present.
    :param save_plots:       boolean.  Whether plots png files should be saved.
    :param save_csv:         boolean.  Whether csv file should be saved.
    :return:
    """
    ####################################################################################################################
    # Open files
    ####################################################################################################################
    # List grib files for individual forecast
    grib_file_list = sorted(glob.glob(grib_paths))

    # Open individual variables
    # 2m air temperature
    new_2t = open_multiple_grib_file_single_variable(grib_file_list, '2t')

    # 2m relative humidity
    new_2r = open_multiple_grib_file_single_variable(grib_file_list, '2r')

    # 2m dewpoint temperature
    new_2d = open_multiple_grib_file_single_variable(grib_file_list, '2d')

    ####################################################################################################################
    # Convert data to Bangladesh local time
    ####################################################################################################################
    new_2t = new_2t.assign_coords({'valid_time': (new_2t.valid_time.values + np.timedelta64(6, 'h'))})
    new_2r = new_2r.assign_coords({'valid_time': (new_2r.valid_time.values + np.timedelta64(6, 'h'))})
    new_2d = new_2d.assign_coords({'valid_time': (new_2d.valid_time.values + np.timedelta64(6, 'h'))})

    ####################################################################################################################
    # Calculate Heat Index
    ####################################################################################################################
    hi = tf.calculate_heat_index_adjusted(new_2t['t2m'].values, new_2d['d2m'].values)

    # Convert to xarray
    hi_xr = xr.DataArray(data=hi,
                         dims=["time", "latitude", "longitude"],
                         coords=dict(time=new_2t.valid_time.values,
                                     latitude=new_2t.latitude.values,
                                     longitude=new_2t.longitude.values
                                     )
                         )

    ####################################################################################################################
    # Convert 2m air temperature from K to C
    ####################################################################################################################
    new_2t_c = new_2t.copy(deep=True)
    new_2t_c['t2m'].values = new_2t_c['t2m'].values - 273.15

    t2m_xr = xr.DataArray(data=new_2t_c['t2m'].values,
                          dims=["time", "latitude", "longitude"],
                          coords=dict(time=new_2t_c.valid_time.values,
                                      latitude=new_2t_c.latitude.values,
                                      longitude=new_2t_c.longitude.values
                                      )
                          ).to_dataset(name='t2m')

    ####################################################################################################################
    # District averaging
    ####################################################################################################################
    # Create high resolution lat and lon values across Bangladesh, to allow resampling on scale that matches district
    # sizes
    new_lats, new_lons = make_0p05_res_lat_lons(27.0, 20.2, 87.8, 93.0)

    # Resample to higher spatial resolution (of new_lats and new_lons defined earlier) to allow zonal stats to calculate
    # averages across multiple grid cells
    t2m_0p05 = t2m_xr.interp(latitude=new_lats, longitude=new_lons, method='nearest')
    hi_0p05 = hi_xr.interp(latitude=new_lats, longitude=new_lons, method='nearest')

    # Calculate averages across all districts
    t2m_dat = get_zonal_stats([t2m_0p05], shp_path)
    hi_dat = get_zonal_stats([hi_0p05.to_dataset(name='var_name')], shp_path)

    # Get out of list form and into dataframe form
    hi_dat1 = hi_dat[0]
    t2m_dat1 = t2m_dat[0]

    # Convert date column from string to datetime
    hi_dat1['date'] = pd.to_datetime(hi_dat1['date'])
    t2m_dat1['date'] = pd.to_datetime(t2m_dat1['date'])

    # Create t2m tables for relevant districts
    kushtia_t2m = t2m_dat1[(t2m_dat1['ADM2_EN'] == 'Kushtia')]
    meherpur_t2m = t2m_dat1[(t2m_dat1['ADM2_EN'] == 'Meherpur')]
    chuadanga_t2m = t2m_dat1[(t2m_dat1['ADM2_EN'] == 'Chuadanga')]

    # Create heat index tables for relevant districts
    kushtia_hi = hi_dat1[(hi_dat1['ADM2_EN'] == 'Kushtia')]
    meherpur_hi = hi_dat1[(hi_dat1['ADM2_EN'] == 'Meherpur')]
    chuadanga_hi = hi_dat1[(hi_dat1['ADM2_EN'] == 'Chuadanga')]

    ####################################################################################################################
    # Make output directories
    ####################################################################################################################
    if make_output_dir:
        if not os.path.isdir(output_csv_dir):
            os.mkdir(output_csv_dir)
        if not os.path.isdir(output_plt_dir):
            os.mkdir(output_plt_dir)

    ####################################################################################################################
    # Generate plots for districts
    ####################################################################################################################
    # Heat Index plots
    plot_heat_index(kushtia_hi, 'Kushtia', save_plots,
                    plot_fpath=f'{output_plt_dir}/Kushtia_hi_{date}T{timestep:02d}.png')
    plot_heat_index(meherpur_hi, 'Meherpur', save_plots,
                    plot_fpath=f'{output_plt_dir}/Meherpur_hi_{date}T{timestep:02d}.png')
    plot_heat_index(chuadanga_hi, 'Chuadanga', save_plots,
                    plot_fpath=f'{output_plt_dir}/Chuadanga_hi_{date}T{timestep:02d}.png')

    # Air temperature plots
    plot_airtemp(kushtia_t2m, 'Kushtia', save_plots, False,
                 plot_fpath=f'{output_plt_dir}/Kushtia_t2m_{date}T{timestep:02d}.png', hline_val=None)
    plot_airtemp(meherpur_t2m, 'Meherpur', save_plots, False,
                 plot_fpath=f'{output_plt_dir}/Meherpur_t2m_{date}T{timestep:02d}.png', hline_val=None)
    plot_airtemp(chuadanga_t2m, 'Chuadanga', save_plots, False,
                 plot_fpath=f'{output_plt_dir}/Chuadanga_t2m_{date}T{timestep:02d}.png', hline_val=None)

    ####################################################################################################################
    # Remove days with incomplete number of timesteps from daily summaries
    ####################################################################################################################
    kushtia_hi = convert_to_only_complete_days(kushtia_hi, 8)
    meherpur_hi = convert_to_only_complete_days(meherpur_hi, 8)
    chuadanga_hi = convert_to_only_complete_days(chuadanga_hi, 8)

    kushtia_t2m = convert_to_only_complete_days(kushtia_t2m, 8)
    meherpur_t2m = convert_to_only_complete_days(meherpur_t2m, 8)
    chuadanga_t2m = convert_to_only_complete_days(chuadanga_t2m, 8)

    ####################################################################################################################
    # Make daily summary statistics dataframes
    ####################################################################################################################
    kushtia_daily = calculate_daily_summary_stats(kushtia_hi.copy(), kushtia_t2m.copy())
    meherpur_daily = calculate_daily_summary_stats(meherpur_hi.copy(), meherpur_t2m.copy())
    chuadanga_daily = calculate_daily_summary_stats(chuadanga_hi.copy(), chuadanga_t2m.copy())
    all_daily = pd.concat([kushtia_daily, meherpur_daily, chuadanga_daily])

    ####################################################################################################################
    # Save to csv
    ####################################################################################################################
    if save_csv:
        all_daily.to_csv(f'{output_csv_dir}/all_daily_gfs_{date}T{timestep:02d}.csv', index=False)


########################################################################################################################
########################################################################################################################
if __name__ == "__main__":
    ####################################################################################################################
    # Set date and file paths
    ####################################################################################################################
    # Forecast initialisation date and time - data for this date must already be downloaded in grib file format in an
    # accessible directory. Data can be downloaded by running download_gfs_forecast_files.py.
    forecast_date = '2024-02-23'  # format YYYY-mm-dd
    forecast_timestep = 18  # 0, 6, 12 or 18

    # File paths
    # Location of grib forecast files (assumes file structure if download_gfs_forecast_files.py has just been run for
    # the date and time above)
    grib_fpaths = f"./forecast_files/fc_{forecast_date}T{forecast_timestep:02d}/*.grib"
    # Path to shapefile with administrative boundaries
    shp_fpath = "./shapefiles/bangladesh/bgd_admbnda_adm2_bbs_20201113.shp"

    # Output directory paths
    output_csv_directory = f"./output/csv_files/fc_{forecast_date}T{forecast_timestep:02d}"
    output_plot_directory = f"./output/plots/fc_{forecast_date}T{forecast_timestep:02d}"

    # Whether to make output directories if they don't currently exist
    make_output_directory = True

    # Whether to save plots and csvs
    save_plots_as_pngs = True
    save_table_as_csv = True

    make_plots_and_csv(forecast_date, forecast_timestep, grib_fpaths, shp_fpath, output_csv_directory,
                       output_plot_directory, make_output_directory, save_plots_as_pngs,
                       save_table_as_csv)

