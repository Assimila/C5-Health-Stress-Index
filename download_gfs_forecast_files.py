import requests
import os
import datetime as dt
import numpy as np


########################################################################################################################
# Functions
########################################################################################################################
def download_file(url, target_dir, make_dir, local_filename):
    """
    Download file from the internet to local drive.
    :param url:             string.   URL which points to the file to be downloaded
    :param target_dir:      string.   Directory in which data should be saved.
    :param make_dir:        boolean.  Whether to make a directory in which the data will be saved.
    :param local_filename:  string.   Name that the file will be saved as
    """
    # Requests data from url (which was previously stored as a string)
    # (stackoverflow note: NOTE the stream=True parameter below)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Option to make output directory
        if make_dir:
            # Check if directory exists and create it if necessary
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
        # Open file where the data will be saved
        with open(os.path.join(target_dir, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
            f.close()


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


def make_link(parameters, atm_level, n_lat, s_lat, w_lon, e_lon, timestep, fcast_date, fcast_time):
    param_string = '=on&var_' + '=on&var_'.join(parameters)

    fcast_filename = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t{fcast_time:02d}z.pgrb2full.0p50.f{timestep:03d}&lev_{atm_level}{param_string}=on&subregion=&leftlon={w_lon}&rightlon={e_lon}&toplat={n_lat}&bottomlat={s_lat}&dir=%2Fgfs.{fcast_date.replace('-', '')}%2F{fcast_time:02d}%2Fatmos"

    return fcast_filename


########################################################################################################################
# Set custom values
########################################################################################################################
# Output directory
forecast_date = "2024-02-23"  # yyyy-mm-dd format
forecast_time = 18  # 0, 6, 12 or 18
download_directory_path = f"./forecast_files/fc_{forecast_date}T{forecast_time:02d}"

# Whether to make output directory
make_directory = True

# Subregion
north_lat = 27
south_lat = 20
west_lon = 87
east_lon = 93

# Level in atmosphere
level = '2_m_above_ground'

# Forecast variables
variables = ['DPT', 'TMP', 'RH']

# Relevant timesteps
timesteps = np.arange(0, 387, 3)

########################################################################################################################
# Run download code
########################################################################################################################
# Generate URLs and filenames for filtered GFS downloads (customised by variables, timesteps, sub-region and forecast
# date and time)
links = []
filenames = []
for i in np.arange(0, 387, 3):
    # link_i = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t00z.pgrb2full.0p50.f{i:03d}&lev_2_m_above_ground=on&var_DPT=on&var_TMP=on&subregion=&leftlon=87&rightlon=93&toplat=27&bottomlat=20&dir=%2Fgfs.{forecast_date.replace('-', '')}%2F00%2Fatmos"
    link_i = make_link(variables, level, north_lat, south_lat, west_lon, east_lon, i, forecast_date, forecast_time)
    file_i = make_output_filename(variables, level, north_lat, south_lat, west_lon, east_lon, i, forecast_date,
                                  forecast_time)
    links.append(link_i)
    filenames.append(file_i)

# Download the files from the list of links
for link in links:
    download_file(link, os.path.abspath(download_directory_path), True, filenames[links.index(link)])
