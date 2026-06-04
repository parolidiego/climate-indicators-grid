# %%

# ---------------------------------------------------------------------------
# PACKAGES
# ---------------------------------------------------------------------------

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar

# Suppress expected RuntimeWarnings from NaN-only ocean cells:
#   - nanmax/nanmin on all-NaN chunks (TXX, TNN, PX1, PX5, PXM, PNM)
#   - divide-by-zero/NaN in PWA (W=0), PWVAR (W=1), TVAR/PVAR before .where() mask
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')


# %%

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("climate_variables_yearly.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# %%

# ---------------------------------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_DIR    = PROJECT_ROOT / "data/in/era5_daily"
OUTPUT_DIR   = PROJECT_ROOT / "data/out"

YEARS           = range(1950, 2024)   # (start: inclusuve, end: exclusive -> ends 31/12 year -1)
YEAR_START      = YEARS.start
YEAR_END        = YEARS.stop - 1
PERC_YEAR_START = 1965                # percentile reference period start
PERC_YEAR_END   = 1994                # percentile reference period end (inclusive)

OUT_PERC    = OUTPUT_DIR / "percentiles"
OUT_DAILY   = OUTPUT_DIR / "daily"
OUT_MONTHLY = OUTPUT_DIR / "monthly"
OUT_YEARLY  = OUTPUT_DIR / "yearly"
for path in [OUT_PERC, OUT_DAILY, OUT_MONTHLY, OUT_YEARLY]:
    path.mkdir(parents=True, exist_ok=True)

dask.config.set(scheduler='threads')


# %%

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

logger.info("Opening ERA5 files (lazy)...")

# File list
files = sorted(INPUT_DIR.glob("era5_daily_*.nc"))
# Lazy opening
ds = xr.open_mfdataset(files, combine='by_coords', engine="netcdf4")
# Slicing
ds = ds.sel(valid_time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

logger.info(f"Graph opened: {ds}")


# %%

# ===========================================================================
# YEARLY VARIABLES
# ===========================================================================


def calculate_TM(ds: xr.Dataset):
    """
    Input: t_mean [Grid, Day] - mean daily 2-metre temperature at the grid level
    Output: TM [Grid, Year] - mean annual temperature at the grid level

    Takes the average of all daily mean-temperature values within each year per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_mean` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `TM` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TM (mean annual temperature at grid cell level)")

    if 't_mean' not in ds:
        raise KeyError(f"'t_mean' not found in dataset. Available variables: {list(ds.data_vars)}")

    # .resample(valid_time='YE') groups data into annual blocks.
    TM = (
        ds['t_mean']
        .resample(valid_time='YE')
        .mean(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .assign_coords(year=np.unique(ds['valid_time'].dt.year.values))
        .assign_attrs(long_name='Mean annual temperature', units='Celsius')
        .to_dataset(name='TM')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return TM

def calculate_TX(ds: xr.Dataset):
    """
    Input: t_max [Grid, Day] - maximum daily 2-metre temperature at the grid level
    Output: TX [Grid, Year] - mean annual maximum temperature at the grid level

    Takes the average of all daily maximum-temperature values within each year per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `TX` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TX (mean annual maximum temperature at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")

    TX = (
        ds['t_max']
        .resample(valid_time='YE')
        .mean(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .assign_coords(year=np.unique(ds['valid_time'].dt.year.values))
        .assign_attrs(long_name='Mean annual maximum temperature', units='Celsius')
        .to_dataset(name='TX')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return TX

def calculate_TN(ds: xr.Dataset):
    """
    Input: t_min [Grid, Day] - minimum daily 2-metre temperature at the grid level
    Output: TN [Grid, Year] - mean annual minimum temperature at the grid level

    Takes the average of all daily minimum-temperature values within each year per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `TN` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TN (mean annual minimum temperature at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")

    TN = (
        ds['t_min']
        .resample(valid_time='YE')
        .mean(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .assign_coords(year=np.unique(ds['valid_time'].dt.year.values))
        .assign_attrs(long_name='Mean annual minimum temperature', units='Celsius')
        .to_dataset(name='TN')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return TN

def calculate_TVAR(ds: xr.Dataset, ds_TM: xr.Dataset):
    """
    Input: t_mean [Grid, Day] - mean daily 2-metre temperature at the grid level
           TM     [Grid, Year] - mean annual temperature (output of calculate_TM)
    Output: TVAR [Grid, Year]  - annual temperature variance at the grid level

    Calculates temperature variability.
    Computed as: TVAR = sum((t_mean_d - TM)^2) / (N - 1)
    where t_mean_d is daily mean temperature, TM is the annual mean temperature,
    and N is the actual number of days in the year (365 or 366 for leap years).

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_mean` with a `valid_time` dimension (daily frequency).
    ds_TM : xr.Dataset
        Dataset containing the variable `TM` with a `year` dimension (annual frequency).
        Typically the output of calculate_TM().

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `TVAR` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TVAR (annual temperature variance at grid cell level)")

    if 't_mean' not in ds:
        raise KeyError(f"'t_mean' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TM' not in ds_TM:
        raise KeyError(f"'TM' not found in dataset. Available variables: {list(ds_TM.data_vars)}")

    # sum((t_mean_d - TM)^2) / (N - 1) per year per grid cell
    # .groupby('valid_time.year') groups days into annual blocks.
    # The subtraction broadcasts TM (annual) back to daily frequency and then squares the result of the subtraction.
    # Finally divide by the number of days in a year, however:
    # groupby.sum(skipna=True) returns 0 for all-NaN groups (ocean), so mask with TM.
    # N is the actual number of days per year (365 or 366 for leap years).
    days_per_year = ds['t_mean'].groupby('valid_time.year').count()
    TVAR = (
        (
            ((ds['t_mean'].groupby('valid_time.year') - ds_TM['TM']) ** 2)
            .groupby('valid_time.year')
            .sum(skipna=True)
            / (days_per_year - 1)
        )
        .where(ds_TM['TM'].notnull())
    ).assign_attrs(long_name='Annual temperature variance', units='Celsius^2').to_dataset(name='TVAR')

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return TVAR

def calculate_DTR(ds_DTR_d: xr.Dataset):
    """
    Input: DTR_d [Grid, Day] - diurnal temperature range at the grid level (output of calculate_DTR_d)
    Output: DTR [Grid, Year] - mean annual diurnal temperature range at the grid level

    Takes the average of all daily DTR (temperature range) values within each year per grid cell.

    Parameters
    ----------
    ds_DTR_d : xr.Dataset
        Lazily-loaded dataset containing the variable `DTR_d` with a `valid_time`
        dimension (daily frequency). Typically the output of calculate_DTR_d().

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `DTR` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating DTR (mean annual diurnal temperature range at grid cell level)")

    if 'DTR_d' not in ds_DTR_d:
        raise KeyError(f"'DTR_d' not found in dataset. Available variables: {list(ds_DTR_d.data_vars)}")

    DTR = (
        ds_DTR_d['DTR_d']
        .resample(valid_time='YE')
        .mean(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .assign_coords(year=np.unique(ds_DTR_d['valid_time'].dt.year.values))
        .assign_attrs(long_name='Mean annual temperature range', units='Celsius')
        .to_dataset(name='DTR')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return DTR

def calculate_TNN_TXX(ds: xr.Dataset):
    """
    Input: t_min [Grid, Day] - minimum daily 2-metre temperature at the grid level
           t_max [Grid, Day] - maximum daily 2-metre temperature at the grid level
    Output: TNN [Grid, Year] - coldest night of the year (annual minimum of daily minimum temperature)
            TXX [Grid, Year] - hottest day of the year (annual maximum of daily maximum temperature)

    Takes the minimum of all daily minimum temperatures and the maximum of all daily maximum
    temperatures within each year per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variables `t_min` and `t_max` with a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables `TNN` and `TXX` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TNN, TXX (coldest night and hottest day per year at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")

    # Extracts a list of years to then be assigned as coordinates
    years = np.unique(ds['valid_time'].dt.year.values)

    # Gets the minimum TN across the year
    TNN = (
        ds['t_min']
        .resample(valid_time='YE')
        .min(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .rename('TNN')
        .assign_coords(year=years)
        .assign_attrs(long_name='Annual minimum of daily minimum temperature', units='Celsius')
    )

    # Gets the maximum TX across the year
    TXX = (
        ds['t_max']
        .resample(valid_time='YE')
        .max(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .rename('TXX')
        .assign_coords(year=years)
        .assign_attrs(long_name='Annual maximum of daily maximum temperature', units='Celsius')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset({'TNN': TNN, 'TXX': TXX})

def calculate_coldwarm_nightsdays(ds: xr.Dataset, ds_T_pkd_5w: xr.Dataset):
    """
    Input: t_min    [Grid, Day]       - daily minimum temperature at the grid level
           t_max    [Grid, Day]       - daily maximum temperature at the grid level
           TN_10p_5w [Grid, DayOfYear] - 10th percentile of t_min in a 5-day window
           TX_10p_5w [Grid, DayOfYear] - 10th percentile of t_max in a 5-day window
           TN_90p_5w [Grid, DayOfYear] - 90th percentile of t_min in a 5-day window
           TX_90p_5w [Grid, DayOfYear] - 90th percentile of t_max in a 5-day window
    Output: CN10 [Grid, Year] - number of cold nights (days where t_min < TN_10p_5w)
            CD10 [Grid, Year] - number of cold days   (days where t_max < TX_10p_5w)
            WN90 [Grid, Year] - number of warm nights (days where t_min > TN_90p_5w)
            WD90 [Grid, Year] - number of warm days   (days where t_max > TX_90p_5w)

    Calculates the number of cold/warm nights or days.
    For each day, the threshold is selected by matching the calendar day-of-year to the
    corresponding entry in ds_T_pkd_5w. Days are then counted per year and various count variables
    of cold/warm nights/days are calculated

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` and `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_T_pkd_5w : xr.Dataset
        Dataset containing `TN_10p_5w`, `TX_10p_5w`, `TN_90p_5w`, `TX_90p_5w`
        with a `day_of_year` dimension (1-366). Typically temp_perc (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables CN10, CD10, WN90, WD90 and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating CN10, CD10, WN90, WD90 (cold/warm night and day counts at grid cell level)")

    for var in ['t_min', 't_max']:
        if var not in ds:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in ['TN_10p_5w', 'TX_10p_5w', 'TN_90p_5w', 'TX_90p_5w']:
        if var not in ds_T_pkd_5w:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_T_pkd_5w.data_vars)}")

    # Remove Feb 29: the percentile thresholds are indexed 1-365, so Feb 29 has no valid threshold
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds.sel(valid_time=~feb29)

    # Map each day's normalized day-of-year to the corresponding percentile threshold.
    # For leap year dates after Feb 28, dt.dayofyear is 1 higher than the equivalent
    # non-leap year date; subtracting 1 gives a consistent DOY 1-365 for the threshold lookup.
    is_leap_and_late = ds['valid_time'].dt.is_leap_year & (ds['valid_time'].dt.month > 2)
    doy = ds['valid_time'].dt.dayofyear - is_leap_and_late.astype(int)
    # For each day in the time series, look up the corresponding percentile threshold
    # by its day-of-year. doy is an array of integers (1-365), one per timestep.
    # .sel(day_of_year=doy) maps each timestep to its threshold, producing arrays
    # of shape (time, latitude, longitude) aligned with the daily data.
    tn10 = ds_T_pkd_5w['TN_10p_5w'].sel(day_of_year=doy)
    tx10 = ds_T_pkd_5w['TX_10p_5w'].sel(day_of_year=doy)
    tn90 = ds_T_pkd_5w['TN_90p_5w'].sel(day_of_year=doy)
    tx90 = ds_T_pkd_5w['TX_90p_5w'].sel(day_of_year=doy)

    years = np.unique(ds['valid_time'].dt.year.values)

    CN10 = (
        (ds['t_min'] < tn10).where(ds['t_min'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of cold nights (t_min < TN_10p_5w)', units='days')
        .rename('CN10')
    )
    CD10 = (
        (ds['t_max'] < tx10).where(ds['t_max'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of cold days (t_max < TX_10p_5w)', units='days')
        .rename('CD10')
    )
    WN90 = (
        (ds['t_min'] > tn90).where(ds['t_min'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of warm nights (t_min > TN_90p_5w)', units='days')
        .rename('WN90')
    )
    WD90 = (
        (ds['t_max'] > tx90).where(ds['t_max'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of warm days (t_max > TX_90p_5w)', units='days')
        .rename('WD90')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset({'CN10': CN10, 'CD10': CD10, 'WN90': WN90, 'WD90': WD90})

def calculate_day_heatwaves(ds: xr.Dataset, ds_TX90p15w: xr.Dataset):
    """
    Input: t_max       [Grid, Day]       - daily maximum temperature at the grid level
           TX_90p_15w  [Grid, DayOfYear] - 90th percentile of t_max in a 15-day window
    Output: DDHW [Grid, Year] - number of days in day heatwave events in the year
            LDHW [Grid, Year] - number of days in the longest day heatwave of the year
            NDHW [Grid, Year] - number of day heatwave events in the year
            TDHW [Grid, Year] - average TX during day heatwave days in the year

    Calculates several day heatwaves variables.
    A day heatwave event is defined as a run of at least 3 consecutive days where daily
    maximum temperature (TX) exceeds the calendar-day-specific 90th percentile computed
    over a 15-day centred window (TX90p15w). Streak detection is applied per grid cell
    and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_TX90p15w : xr.Dataset
        Dataset containing `TX_90p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TX_90p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables DDHW, LDHW, NDHW, TDHW and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating DDHW, LDHW, NDHW, TDHW (day heatwave indicators at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TX_90p_15w' not in ds_TX90p15w:
        raise KeyError(f"'TX_90p_15w' not found in dataset. Available variables: {list(ds_TX90p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_max']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tx90_table = ds_TX90p15w['TX_90p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tx90_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tx90_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmax for this block
        tx = ds_block['t_max'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tx90_arr = tx90_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays, initialised to NaN
        DDHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # total heatwave days
        LDHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest heatwave (days)
        NDHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # number of heatwave events
        TDHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # mean temp during heatwaves

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue
                # Look up the daily 90th-percentile threshold for each day of this year at that grid point
                tx90_1d = tx90_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing
                # Creates binary variables: 1 where t_max exceeds the daily 90th percentile, 0 otherwise
                hot = (tx_1d > tx90_1d).astype(np.int8)
                # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                padded = np.concatenate([[0], hot, [0]])
                # Take the diff between consecutive days, +1 marks the start of a hot spell; -1 marks the end
                d = np.diff(padded)
                # Get all the start dates (value = +1)
                starts = np.where(d == 1)[0]
                # Get all the end dates (value = -1)
                ends = np.where(d == -1)[0]
                # Get the length of heatwaves
                lengths = ends - starts
                # A heatwave requires at least 3 consecutive hot days, so mask
                hw_mask = lengths >= 3
                # Assign 0 values if no heatwave is detected
                if not hw_mask.any():
                    DDHW[0, i, j] = 0
                    LDHW[0, i, j] = 0
                    NDHW[0, i, j] = 0
                    continue
                # Use the mask to get the start and the end date of the valid heatwaves only
                hw_starts  = starts[hw_mask]
                hw_lengths = lengths[hw_mask]
                # Calculate the variables
                DDHW[0, i, j] = hw_lengths.sum() # total days across all heatwave events
                LDHW[0, i, j] = hw_lengths.max() # duration of the longest single event
                NDHW[0, i, j] = hw_mask.sum() # count of distinct events
                # Concatenate the actual t_max values from all heatwave periods to compute mean
                hw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                # Calculates mean
                TDHW[0, i, j] = hw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({
            'DDHW': xr.DataArray(DDHW, dims=dims, coords=coords),
            'LDHW': xr.DataArray(LDHW, dims=dims, coords=coords),
            'NDHW': xr.DataArray(NDHW, dims=dims, coords=coords),
            'TDHW': xr.DataArray(TDHW, dims=dims, coords=coords),
        })

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in ['DDHW', 'LDHW', 'NDHW', 'TDHW']
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx90_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {
        'DDHW': ('Number of days in day heatwave events (TX > TX_90p_15w for >= 3 consecutive days)', 'days'),
        'LDHW': ('Number of days in the longest day heatwave event of the year', 'days'),
        'NDHW': ('Number of day heatwave events in the year', 'count'),
        'TDHW': ('Average TX during day heatwave days', 'Celsius'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_night_heatwaves(ds: xr.Dataset, ds_TN90p15w: xr.Dataset):
    """
    Input: t_min       [Grid, Day]       - daily minimum temperature at the grid level
           TN90p15w  [Grid, DayOfYear] - 90th percentile of t_min in a 15-day window
                                           (from compute_rolling_percentiles)
    Output: DNHW [Grid, Year] - number of nights in night heatwave events in the year
            LNHW [Grid, Year] - number of nights in the longest night heatwave of the year
            NNHW [Grid, Year] - number of night heatwave events in the year
            TNHW [Grid, Year] - average TN during night heatwave nights in the year

    Calculates several night heatwaves variables.
    A night heatwave event is defined as a run of at least 3 consecutive nights where daily
    minimum temperature (TN) exceeds the calendar-day-specific 90th percentile computed
    over a 15-day centred window (TN90p15w). Streak detection is applied per grid cell
    and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with
        a `valid_time` dimension (daily frequency).
    ds_TN90p15w : xr.Dataset
        Dataset containing `TN_90p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TN_90p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables DNHW, LNHW, NNHW, TNHW and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating DNHW, LNHW, NNHW, TNHW (night heatwave indicators at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TN_90p_15w' not in ds_TN90p15w:
        raise KeyError(f"'TN_90p_15w' not found in dataset. Available variables: {list(ds_TN90p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_min']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tn90_table = ds_TN90p15w['TN_90p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tn90_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tn90_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmin for this block
        tn = ds_block['t_min'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tn90_arr = tn90_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays, initialised to NaN
        DNHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # total heatwave nights
        LNHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest heatwave (nights)
        NNHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # number of heatwave events
        TNHW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # mean temp during heatwaves

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue
                # Look up the daily 90th-percentile threshold for each day of this year at that grid point
                tn90_1d = tn90_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing
                # Creates binary variables: 1 where t_min exceeds the daily 90th percentile, 0 otherwise
                hot = (tn_1d > tn90_1d).astype(np.int8)
                # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                padded = np.concatenate([[0], hot, [0]])
                # Take the diff between consecutive days, +1 marks the start of a hot spell; -1 marks the end
                d = np.diff(padded)
                # Get all the start dates (value = +1)
                starts = np.where(d == 1)[0]
                # Get all the end dates (value = -1)
                ends = np.where(d == -1)[0]
                # Get the length of heatwaves
                lengths = ends - starts
                # A heatwave requires at least 3 consecutive hot nights, so mask
                hw_mask = lengths >= 3
                # Assign 0 values if no heatwave is detected
                if not hw_mask.any():
                    DNHW[0, i, j] = 0
                    LNHW[0, i, j] = 0
                    NNHW[0, i, j] = 0
                    continue
                # Use the mask to get the start and the end date of the valid heatwaves only
                hw_starts  = starts[hw_mask]
                hw_lengths = lengths[hw_mask]
                # Calculate the variables
                DNHW[0, i, j] = hw_lengths.sum() # total nights across all heatwave events
                LNHW[0, i, j] = hw_lengths.max() # duration of the longest single event
                NNHW[0, i, j] = hw_mask.sum() # count of distinct events
                # Concatenate the actual t_min values from all heatwave periods to compute mean
                hw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                # Calculates mean
                TNHW[0, i, j] = hw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({
            'DNHW': xr.DataArray(DNHW, dims=dims, coords=coords),
            'LNHW': xr.DataArray(LNHW, dims=dims, coords=coords),
            'NNHW': xr.DataArray(NNHW, dims=dims, coords=coords),
            'TNHW': xr.DataArray(TNHW, dims=dims, coords=coords),
        })

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in ['DNHW', 'LNHW', 'NNHW', 'TNHW']
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn90_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {
        'DNHW': ('Number of nights in night heatwave events (TN > TN_90p_15w for >= 3 consecutive nights)', 'nights'),
        'LNHW': ('Number of nights in the longest night heatwave event of the year', 'nights'),
        'NNHW': ('Number of night heatwave events in the year', 'count'),
        'TNHW': ('Average TN during night heatwave nights', 'Celsius'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_day_coldwaves(ds: xr.Dataset, ds_TX10p15w: xr.Dataset):
    """
    Input: t_max       [Grid, Day]       - daily maximum temperature at the grid level
           TX_10p_15w  [Grid, DayOfYear] - 10th percentile of t_max in a 15-day window
    Output: DDCW [Grid, Year] - number of days in day coldwave events in the year
            LDCW [Grid, Year] - number of days in the longest day coldwave of the year
            NDCW [Grid, Year] - number of day coldwave events in the year
            TDCW [Grid, Year] - average TX during day coldwave days in the year

    Calculates several day coldwaves variables.
    A day coldwave event is defined as a run of at least 3 consecutive days where daily
    maximum temperature (TX) falls below the calendar-day-specific 10th percentile computed
    over a 15-day centred window (TX_10p_15w). Streak detection is applied per grid cell
    and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_TX10p15w : xr.Dataset
        Dataset containing `TX_10p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TX_10p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables DDCW, LDCW, NDCW, TDCW and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating DDCW, LDCW, NDCW, TDCW (day coldwave indicators at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TX_10p_15w' not in ds_TX10p15w:
        raise KeyError(f"'TX_10p_15w' not found in dataset. Available variables: {list(ds_TX10p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_max']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tx10_table = ds_TX10p15w['TX_10p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tx10_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tx10_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmax for this block
        tx = ds_block['t_max'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tx10_arr = tx10_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays, initialised to NaN
        DDCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # total coldwave days
        LDCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest coldwave (days)
        NDCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # number of coldwave events
        TDCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # mean temp during coldwaves

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue
                # Look up the daily 10th-percentile threshold for each day of this year at that grid point
                tx10_1d = tx10_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing
                # Creates binary variables: 1 where t_max falls below the daily 10th percentile, 0 otherwise
                cold = (tx_1d < tx10_1d).astype(np.int8)
                # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                padded = np.concatenate([[0], cold, [0]])
                # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                d = np.diff(padded)
                # Get all the start dates (value = +1)
                starts = np.where(d == 1)[0]
                # Get all the end dates (value = -1)
                ends = np.where(d == -1)[0]
                # Get the length of coldwaves
                lengths = ends - starts
                # A coldwave requires at least 3 consecutive cold days, so mask
                cw_mask = lengths >= 3
                # Assign 0 values if no coldwave is detected
                if not cw_mask.any():
                    DDCW[0, i, j] = 0
                    LDCW[0, i, j] = 0
                    NDCW[0, i, j] = 0
                    continue
                # Use the mask to get the start and the end date of the valid coldwaves only
                cw_starts  = starts[cw_mask]
                cw_lengths = lengths[cw_mask]
                # Calculate the variables
                DDCW[0, i, j] = cw_lengths.sum() # total days across all coldwave events
                LDCW[0, i, j] = cw_lengths.max() # duration of the longest single event
                NDCW[0, i, j] = cw_mask.sum() # count of distinct events
                # Concatenate the actual t_max values from all coldwave periods to compute mean
                cw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                # Calculates mean
                TDCW[0, i, j] = cw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({
            'DDCW': xr.DataArray(DDCW, dims=dims, coords=coords),
            'LDCW': xr.DataArray(LDCW, dims=dims, coords=coords),
            'NDCW': xr.DataArray(NDCW, dims=dims, coords=coords),
            'TDCW': xr.DataArray(TDCW, dims=dims, coords=coords),
        })

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in ['DDCW', 'LDCW', 'NDCW', 'TDCW']
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx10_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {
        'DDCW': ('Number of days in day coldwave events (TX < TX_10p_15w for >= 3 consecutive days)', 'days'),
        'LDCW': ('Number of days in the longest day coldwave event of the year', 'days'),
        'NDCW': ('Number of day coldwave events in the year', 'count'),
        'TDCW': ('Average TX during day coldwave days', 'Celsius'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_night_coldwaves(ds: xr.Dataset, ds_TN10p15w: xr.Dataset):
    """
    Input: t_min        [Grid, Day]       - daily minimum temperature at the grid level
           TN_10p_15w  [Grid, DayOfYear] - 10th percentile of t_min in a 15-day window
    Output: DNCW [Grid, Year] - number of nights in night coldwave events in the year
            LNCW [Grid, Year] - number of nights in the longest night coldwave of the year
            NNCW [Grid, Year] - number of night coldwave events in the year
            TNCW [Grid, Year] - average TN during night coldwave nights in the year

    Calculates several night coldwaves variables.
    A night coldwave event is defined as a run of at least 3 consecutive nights where daily
    minimum temperature (TN) falls below the calendar-day-specific 10th percentile computed
    over a 15-day centred window (TN_10p_15w). Streak detection is applied per grid cell
    and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with
        a `valid_time` dimension (daily frequency).
    ds_TN10p15w : xr.Dataset
        Dataset containing `TN_10p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TN_10p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables DNCW, LNCW, NNCW, TNCW and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating DNCW, LNCW, NNCW, TNCW (night coldwave indicators at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TN_10p_15w' not in ds_TN10p15w:
        raise KeyError(f"'TN_10p_15w' not found in dataset. Available variables: {list(ds_TN10p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_min']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tn10_table = ds_TN10p15w['TN_10p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tn10_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tn10_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmin for this block
        tn = ds_block['t_min'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tn10_arr = tn10_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays, initialised to NaN
        DNCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # total coldwave nights
        LNCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest coldwave (nights)
        NNCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # number of coldwave events
        TNCW = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # mean temp during coldwaves

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue
                # Look up the daily 10th-percentile threshold for each day of this year at that grid point
                tn10_1d = tn10_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing
                # Creates binary variables: 1 where t_min falls below the daily 10th percentile, 0 otherwise
                cold = (tn_1d < tn10_1d).astype(np.int8)
                # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                padded = np.concatenate([[0], cold, [0]])
                # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                d = np.diff(padded)
                # Get all the start dates (value = +1)
                starts = np.where(d == 1)[0]
                # Get all the end dates (value = -1)
                ends = np.where(d == -1)[0]
                # Get the length of coldwaves
                lengths = ends - starts
                # A coldwave requires at least 3 consecutive cold nights, so mask
                cw_mask = lengths >= 3
                # Assign 0 values if no coldwave is detected
                if not cw_mask.any():
                    DNCW[0, i, j] = 0
                    LNCW[0, i, j] = 0
                    NNCW[0, i, j] = 0
                    continue
                # Use the mask to get the start and the end date of the valid coldwaves only
                cw_starts  = starts[cw_mask]
                cw_lengths = lengths[cw_mask]
                # Calculate the variables
                DNCW[0, i, j] = cw_lengths.sum() # total nights across all coldwave events
                LNCW[0, i, j] = cw_lengths.max() # duration of the longest single event
                NNCW[0, i, j] = cw_mask.sum() # count of distinct events
                # Concatenate the actual t_min values from all coldwave periods to compute mean
                cw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                # Calculates mean
                TNCW[0, i, j] = cw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({
            'DNCW': xr.DataArray(DNCW, dims=dims, coords=coords),
            'LNCW': xr.DataArray(LNCW, dims=dims, coords=coords),
            'NNCW': xr.DataArray(NNCW, dims=dims, coords=coords),
            'TNCW': xr.DataArray(TNCW, dims=dims, coords=coords),
        })

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in ['DNCW', 'LNCW', 'NNCW', 'TNCW']
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn10_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {
        'DNCW': ('Number of nights in night coldwave events (TN < TN_10p_15w for >= 3 consecutive nights)', 'nights'),
        'LNCW': ('Number of nights in the longest night coldwave event of the year', 'nights'),
        'NNCW': ('Number of night coldwave events in the year', 'count'),
        'TNCW': ('Average TN during night coldwave nights', 'Celsius'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_CSD(ds: xr.Dataset, ds_TN10p5w: xr.Dataset):
    """
    Input: t_min       [Grid, Day]       - daily minimum temperature at the grid level
           TN_10p_5w  [Grid, DayOfYear] - 10th percentile of t_min in a 5-day window
    Output: CSD [Grid, Year] - number of days in which TN < TN_10p_5w is observed
                               in intervals of at least 6 consecutive days

    Calculates the number of days in cold spells (intervals of at least 6 days where TN < TN_10p_5w).
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with
        a `valid_time` dimension (daily frequency).
    ds_TN10p5w : xr.Dataset
        Dataset containing `TN_10p_5w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TN_10p_5w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable CSD and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating CSD (cold spell duration at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TN_10p_5w' not in ds_TN10p5w:
        raise KeyError(f"'TN_10p_5w' not found in dataset. Available variables: {list(ds_TN10p5w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_min']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tn10_table = ds_TN10p5w['TN_10p_5w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tn10_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tn10_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmin for this block
        tn = ds_block['t_min'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tn10_arr = tn10_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output array, initialised to NaN
        CSD = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # total days in cold spells

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue
                # Look up the daily 10th-percentile threshold for each day of this year at that grid point
                tn10_1d = tn10_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing
                # Creates binary variables: 1 where t_min falls below the daily 10th percentile, 0 otherwise
                cold = (tn_1d < tn10_1d).astype(np.int8)
                # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                padded = np.concatenate([[0], cold, [0]])
                # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                d = np.diff(padded)
                # Get all the start dates (value = +1)
                starts = np.where(d == 1)[0]
                # Get all the end dates (value = -1)
                ends = np.where(d == -1)[0]
                # Get the length of each spell
                lengths = ends - starts
                # A cold spell requires at least 6 consecutive cold days, so mask
                spell_mask = lengths >= 6
                # Sum the total days across all valid cold spells (0 if none)
                CSD[0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        return xr.Dataset({'CSD': xr.DataArray(CSD, dims=['valid_time', 'latitude', 'longitude'],
                                               coords=coords)})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        'CSD': xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name='template-CSD',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn10_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {
        'CSD': ('Number of days in cold spell events (TN < TN_10p_5w for >= 6 consecutive days)', 'days'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_WSD(ds: xr.Dataset, ds_TX90p5w: xr.Dataset):
    """
    Input: t_max       [Grid, Day]       - daily maximum temperature at the grid level
           TX_90p_5w  [Grid, DayOfYear] - 90th percentile of t_max in a 5-day window
    Output: WSD [Grid, Year] - number of days in which TX > TX_90p_5w is observed
                               in intervals of at least 6 consecutive days

    Calculates the number of days in warm spells (intervals of at least 6 days where TX > TX_90p_5w).
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_TX90p5w : xr.Dataset
        Dataset containing `TX_90p_5w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TX_90p_5w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable WSD and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating WSD (warm spell duration at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TX_90p_5w' not in ds_TX90p5w:
        raise KeyError(f"'TX_90p_5w' not found in dataset. Available variables: {list(ds_TX90p5w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_max']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tx90_table = ds_TX90p5w['TX_90p_5w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tx90_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tx90_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmax for this block
        tx = ds_block['t_max'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tx90_arr = tx90_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output array, initialised to NaN
        WSD = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # total days in warm spells

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue
                # Look up the daily 90th-percentile threshold for each day of this year at that grid point
                tx90_1d = tx90_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing
                # Creates binary variables: 1 where t_max exceeds the daily 90th percentile, 0 otherwise
                warm = (tx_1d > tx90_1d).astype(np.int8)
                # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                padded = np.concatenate([[0], warm, [0]])
                # Take the diff between consecutive days, +1 marks the start of a warm spell; -1 marks the end
                d = np.diff(padded)
                # Get all the start dates (value = +1)
                starts = np.where(d == 1)[0]
                # Get all the end dates (value = -1)
                ends = np.where(d == -1)[0]
                # Get the length of each spell
                lengths = ends - starts
                # A warm spell requires at least 6 consecutive warm days, so mask
                spell_mask = lengths >= 6
                # Sum the total days across all valid warm spells (0 if none)
                WSD[0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        return xr.Dataset({'WSD': xr.DataArray(WSD, dims=['valid_time', 'latitude', 'longitude'],
                                               coords=coords)})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        'WSD': xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name='template-WSD',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx90_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {
        'WSD': ('Number of days in warm spell events (TX > TX_90p_5w for >= 6 consecutive days)', 'days'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_TNnumber_TXnumber(ds: xr.Dataset):
    """
    Input: t_min [Grid, Day] - daily minimum 2-metre temperature at the grid level
           t_max [Grid, Day] - daily maximum 2-metre temperature at the grid level
    Output: TN0   [Grid, Year] - number of days with t_min < 0°C
            TNm5  [Grid, Year] - number of days with t_min < -5°C
            TNm10 [Grid, Year] - number of days with t_min < -10°C
            TX30  [Grid, Year] - number of days with t_max > 30°C
            TX35  [Grid, Year] - number of days with t_max > 35°C
            TX40  [Grid, Year] - number of days with t_max > 40°C

    Counts days per year where daily minimum or maximum temperature exceeds fixed thresholds.
    Unlike percentile-based counts, Feb 29 is retained because thresholds are not day-of-year specific.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` and `t_max` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables TN0, TX35, TX40 and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TN0, TNm5, TNm10, TX30, TX35, TX40 (fixed temperature threshold exceedance counts at grid cell level)")

    for var in ['t_min', 't_max']:
        if var not in ds:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds.data_vars)}")

    years = np.unique(ds['valid_time'].dt.year.values)

    TN0 = (
        (ds['t_min'] < 0).where(ds['t_min'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of days with daily minimum temperature below 0°C', units='days')
        .rename('TN0')
    )
    TNm5 = (
        (ds['t_min'] < -5).where(ds['t_min'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of days with daily minimum temperature below -5°C', units='days')
        .rename('TNm5')
    )
    TNm10 = (
        (ds['t_min'] < -10).where(ds['t_min'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of days with daily minimum temperature below -10°C', units='days')
        .rename('TNm10')
    )
    TX30 = (
        (ds['t_max'] > 30).where(ds['t_max'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of days with daily maximum temperature above 30°C', units='days')
        .rename('TX30')
    )
    TX35 = (
        (ds['t_max'] > 35).where(ds['t_max'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of days with daily maximum temperature above 35°C', units='days')
        .rename('TX35')
    )
    TX40 = (
        (ds['t_max'] > 40).where(ds['t_max'].notnull()).resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of days with daily maximum temperature above 40°C', units='days')
        .rename('TX40')
    )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset({'TN0': TN0, 'TNm5': TNm5, 'TNm10': TNm10, 'TX30': TX30, 'TX35': TX35, 'TX40': TX40})

def calculate_TM_bins(ds: xr.Dataset):
    """
    Input: t_mean [Grid, Day] - daily mean 2-metre temperature at the grid level
    Output: 15 variables [Grid, Year] counting days falling in each 3°C-wide bin:
            TM_below_minus9     - t_mean < -9°C
            TM_minus6_to_minus9 - -9 <= t_mean < -6°C
            TM_minus3_to_minus6 - -6 <= t_mean < -3°C
            TM_minus3_to_0      - -3 <= t_mean < 0°C
            TM_0_to_3           - 0 <= t_mean < 3°C
            TM_3_to_6           - 3 <= t_mean < 6°C
            TM_6_to_9           - 6 <= t_mean < 9°C
            TM_9_to_12          - 9 <= t_mean < 12°C
            TM_12_to_15         - 12 <= t_mean < 15°C
            TM_15_to_18         - 15 <= t_mean < 18°C
            TM_18_to_21         - 18 <= t_mean < 21°C
            TM_21_to_24         - 21 <= t_mean < 24°C
            TM_24_to_27         - 24 <= t_mean < 27°C
            TM_27_to_30         - 27 <= t_mean < 30°C
            TM_above_30         - t_mean >= 30°C

    Bins are left-inclusive, right-exclusive. Feb 29 is retained because thresholds are
    not day-of-year specific.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_mean` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with 15 TM_bin variables and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TM bin counts (daily mean temperature distribution at grid cell level)")

    if 't_mean' not in ds:
        raise KeyError(f"'t_mean' not found in dataset. Available variables: {list(ds.data_vars)}")

    years = np.unique(ds['valid_time'].dt.year.values)
    tm = ds['t_mean']

    bins = [
        ('TM_below_minus9',     None, -9,  'Number of days with daily mean temperature below -9°C'),
        ('TM_minus6_to_minus9',   -9, -6,  'Number of days with daily mean temperature in [-9°C, -6°C)'),
        ('TM_minus3_to_minus6',   -6, -3,  'Number of days with daily mean temperature in [-6°C, -3°C)'),
        ('TM_minus3_to_0',        -3,  0,  'Number of days with daily mean temperature in [-3°C, 0°C)'),
        ('TM_0_to_3',              0,  3,  'Number of days with daily mean temperature in [0°C, 3°C)'),
        ('TM_3_to_6',              3,  6,  'Number of days with daily mean temperature in [3°C, 6°C)'),
        ('TM_6_to_9',              6,  9,  'Number of days with daily mean temperature in [6°C, 9°C)'),
        ('TM_9_to_12',             9, 12,  'Number of days with daily mean temperature in [9°C, 12°C)'),
        ('TM_12_to_15',           12, 15,  'Number of days with daily mean temperature in [12°C, 15°C)'),
        ('TM_15_to_18',           15, 18,  'Number of days with daily mean temperature in [15°C, 18°C)'),
        ('TM_18_to_21',           18, 21,  'Number of days with daily mean temperature in [18°C, 21°C)'),
        ('TM_21_to_24',           21, 24,  'Number of days with daily mean temperature in [21°C, 24°C)'),
        ('TM_24_to_27',           24, 27,  'Number of days with daily mean temperature in [24°C, 27°C)'),
        ('TM_27_to_30',           27, 30,  'Number of days with daily mean temperature in [27°C, 30°C)'),
        ('TM_above_30',           30, None, 'Number of days with daily mean temperature above 30°C'),
    ]

    result = {}
    for name, lower, upper, long_name in bins:
        if lower is None:
            cond = tm < upper
        elif upper is None:
            cond = tm >= lower
        else:
            cond = (tm >= lower) & (tm < upper)

        result[name] = (
            cond.where(tm.notnull()).resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=long_name, units='days')
            .rename(name)
        )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset(result)

def calculate_PA(ds: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
    Output: PA [Grid, Year] - mean daily precipitation per year at the grid level

    Takes the mean of all daily precipitation values within each year per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `PA` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PA (mean annual precipitation at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")

    PA = (
        ds['tot_prec']
        .resample(valid_time='YE')
        .mean(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .assign_coords(year=np.unique(ds['valid_time'].dt.year.values))
        .assign_attrs(long_name='Mean annual precipitation', units='m')
        .to_dataset(name='PA')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return PA

def calculate_PWT(ds: xr.Dataset, ds_orig: xr.Dataset):
    """
    Input: PW_d     [Grid, Day] - daily precipitation on wet days only (NaN on dry days)
           tot_prec [Grid, Day] - total daily precipitation (used to build land mask)
    Output: PWT [Grid, Year] - total annual precipitation on wet days at the grid level

    Sums wet-day precipitation values over each year per grid cell.
    Land cells with no wet days in a year receive 0; ocean cells receive NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Lazy dataset containing the variable `PW_d` with a `valid_time`
        dimension (daily frequency). Typically the output of calculate_PW_d().
    ds_orig : xr.Dataset
        Dataset containing `tot_prec` with a `valid_time` dimension. Used to build
        the land mask: cells with at least one non-NaN tot_prec value are treated as land.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `PWT` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PWT (total annual wet day precipitation at grid cell level)")

    if 'PW_d' not in ds:
        raise KeyError(f"'PW_d' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'tot_prec' not in ds_orig:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds_orig.data_vars)}")

    # True for cells that have at least one non-NaN tot_prec value (i.e. land cells)
    land_mask = ds_orig['tot_prec'].notnull().any(dim='valid_time')

    # PW_d is NaN on both dry days and ocean cells, so an all-NaN year is ambiguous:
    # it could mean "no wet days this year" (land) or "no data at all" (ocean).
    # min_count=0 resolves this by returning 0 for any all-NaN group; the land_mask
    # then restores NaN specifically for ocean cells.
    PWT = (
        ds['PW_d']
        .resample(valid_time='YE')
        .sum(dim='valid_time', skipna=True, min_count=0)
        .where(land_mask)
        .rename({'valid_time': 'year'})
        .assign_coords(year=np.unique(ds['valid_time'].dt.year.values))
        .assign_attrs(long_name='Total annual wet day precipitation', units='m')
        .to_dataset(name='PWT')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return PWT

def calculate_W(ds: xr.Dataset, ds_orig: xr.Dataset):
    """
    Input: PW_d     [Grid, Day] - daily precipitation on wet days only (NaN on dry days)
           tot_prec [Grid, Day] - total daily precipitation (used to build land mask)
    Output: W [Grid, Year] - number of wet days per year at the grid level

    Counts the number of wet days per year at grid cell level.
    Land cells with no wet days in a year receive 0; ocean cells receive NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Lazy dataset containing the variable `PW_d` with a `valid_time`
        dimension (daily frequency). Typically the output of calculate_PW_d().
    ds_orig : xr.Dataset
        Dataset containing `tot_prec` with a `valid_time` dimension. Used to build
        the land mask: cells with at least one non-NaN tot_prec value are treated as land.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `W` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating W (number of wet days per year at grid cell level)")

    if 'PW_d' not in ds:
        raise KeyError(f"'PW_d' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'tot_prec' not in ds_orig:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds_orig.data_vars)}")

    # True for cells that have at least one non-NaN tot_prec value (i.e. land cells)
    land_mask = ds_orig['tot_prec'].notnull().any(dim='valid_time')

    # PW_d is NaN on both dry days and ocean cells, so count() alone cannot distinguish
    # "no wet days this year" (land, should be 0) from "no data at all" (ocean, should be NaN).
    # count() naturally returns 0 for all-NaN groups; the land_mask then restores NaN
    # specifically for ocean cells.
    W = (
        ds['PW_d']
        .resample(valid_time='YE')
        .count(dim='valid_time')
        .where(land_mask)
        .rename({'valid_time': 'year'})
        .assign_coords(year=np.unique(ds['valid_time'].dt.year.values))
        .assign_attrs(long_name='Number of wet days', units='count')
        .to_dataset(name='W')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return W

def calculate_PWA(ds_PWT: xr.Dataset, ds_W: xr.Dataset):
    """
    Input: PWT [Grid, Year] - total annual precipitation on wet days
           W   [Grid, Year] - number of wet days per year
    Output: PWA [Grid, Year] - average daily precipitation on wet days

    Computing average daily precipitation on wet days as PWA = PWT / W.
    Land cells with zero wet days in a year yield PWT=0 and W=0, so PWA=NaN (undefined average).
    Ocean cells are NaN in both inputs and therefore NaN in PWA as well.

    Parameters
    ----------
    ds_PWT : xr.Dataset
        Dataset containing the variable `PWT` with dimensions (year, latitude, longitude).
    ds_W : xr.Dataset
        Dataset containing the variable `W` with dimensions (year, latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `PWA` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PWA (average daily precipitation on wet days at grid cell level)")

    if 'PWT' not in ds_PWT:
        raise KeyError(f"'PWT' not found in dataset. Available variables: {list(ds_PWT.data_vars)}")
    if 'W' not in ds_W:
        raise KeyError(f"'W' not found in dataset. Available variables: {list(ds_W.data_vars)}")

    # Division is intentionally 0/0 for land cells with no wet days, producing NaN —
    # the average is undefined when there are no wet days to average over.
    PWA = (
        (ds_PWT['PWT'] / ds_W['W'])
        .assign_attrs(long_name='Average daily precipitation on wet days', units='m')
        .to_dataset(name='PWA')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return PWA

def calculate_PVAR(ds: xr.Dataset, ds_PA: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
           PA [Grid, Year]      - mean annual precipitation (output of calculate_PA)
    Output: PVAR [Grid, Year]   - annual precipitation variance at the grid level

    Calculates precipitation variance
    Computed as: PVAR = sum((Pd - PA)^2) / (N - 1)
    where Pd is daily precipitation, PA is the annual mean precipitation,
    and N is the actual number of days in the year (365 or 366 for leap years).

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec`.
    ds_PA : xr.Dataset
        Dataset containing the variable `PA` with a `year` dimension.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `PVAR` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PVAR (annual precipitation variance at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'PA' not in ds_PA:
        raise KeyError(f"'PA' not found in dataset. Available variables: {list(ds_PA.data_vars)}")

    # sum((Pd - PA)^2) / (N - 1) per year per grid cell.
    # .groupby('valid_time.year') groups days into annual blocks.
    # The subtraction broadcasts PA (annual) back to daily frequency and then squares the result of the subtraction.
    # Finally divide by the number of days in a year minus one, however:
    # groupby.sum(skipna=True) returns 0 for all-NaN groups (ocean), so mask with PA.
    # N is the actual number of days per year (365 or 366 for leap years).
    days_per_year = ds['tot_prec'].groupby('valid_time.year').count()
    PVAR = (
        (
            ((ds['tot_prec'].groupby('valid_time.year') - ds_PA['PA']) ** 2)
            .groupby('valid_time.year')
            .sum(skipna=True)
            / (days_per_year - 1)
        )
        .where(ds_PA['PA'].notnull())
    ).assign_attrs(long_name='Annual precipitation variance', units='m^2').to_dataset(name='PVAR')

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return PVAR

def calculate_PWVAR(ds_PWd: xr.Dataset, ds_PWA: xr.Dataset, ds_W: xr.Dataset):
    """
    Input: PW_d [Grid, Day]  - daily precipitation on wet days only
           PWA  [Grid, Year] - average daily precipitation on wet days
           W    [Grid, Year] - number of wet days per year
    Output: PWVAR [Grid, Year] - annual wet day precipitation variance

    Calculates wet days precipitation variance
    Computed as: PWVAR = sum((PW_d - PWA)^2) / (W - 1)
    where PW_d is daily wet-day precipitation, PWA is the annual mean wet-day precipitation,
    and W is the number of wet days in the year. Years with zero wet days yield NaN (undefined).

    Parameters
    ----------
    ds_PWd : xr.Dataset
        LAZY Dataset containing the variable `PW_d`.
    ds_PWA : xr.Dataset
        LAZY Dataset containing the variable `PWA`.
    ds_W : xr.Dataset
        LAZY Dataset containing the variable `W`.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `PWVAR` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PWVar (annual wet day precipitation variance at grid cell level)")

    if 'PW_d' not in ds_PWd:
        raise KeyError(f"'PW_d' not found in dataset. Available variables: {list(ds_PWd.data_vars)}")
    if 'PWA' not in ds_PWA:
        raise KeyError(f"'PWA' not found in dataset. Available variables: {list(ds_PWA.data_vars)}")
    if 'W' not in ds_W:
        raise KeyError(f"'W' not found in dataset. Available variables: {list(ds_W.data_vars)}")

    # sum((PW_d - PWA)^2) / (W - 1) per year per grid cell.
    # .groupby('valid_time.year') groups days into annual blocks.
    # The subtraction broadcasts PWA (annual) back to daily frequency and then squares the result of the subtraction.
    # skipna=True ignores dry days (NaN in PW_d), so only wet days contribute to the sum.
    # Finally divide by (W - 1); mask with PWA which is NaN for both ocean cells and dry-year land cells.
    PWVAR = (
        (
            ((ds_PWd['PW_d'].groupby('valid_time.year') - ds_PWA['PWA']) ** 2)
            .groupby('valid_time.year')
            .sum(skipna=True)
            / (ds_W['W'] - 1)
        )
        .where(ds_PWA['PWA'].notnull())
    ).assign_attrs(long_name='Annual wet day precipitation variance', units='m^2').to_dataset(name='PWVAR')

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return PWVAR

def calculate_P95WT_P99WT(ds: xr.Dataset, ds_PW_pj: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
           PW_95p  [Grid]       - 95th percentile of wet day precipitation
           PW_99p  [Grid]       - 99th percentile of wet day precipitation
    Output: P95WT [Grid, Year] - total annual precipitation on very wet days (tot_prec >= PW_95p)
            P99WT [Grid, Year] - total annual precipitation on extremely wet days (tot_prec >= PW_99p)

    Calculates total annual precipitation in very wet days and extremely wet days.
    Land cells with no days exceeding the threshold in a given year receive 0; ocean cells receive NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec`.
    ds_PW_pj : xr.Dataset
        Dataset containing `PW_95p` and `PW_99p` with dimensions (latitude, longitude).
        Typically wet_days_perc (computed).

    Returns
    -------
    xr.Dataset
        Lazy Dataset with variables `P95WT` and `P99WT` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating P95WT, P99WT (very and extremely wet day annual precipitation totals at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'PW_95p' not in ds_PW_pj:
        raise KeyError(f"'PW_95p' not found in dataset. Available variables: {list(ds_PW_pj.data_vars)}")
    if 'PW_99p' not in ds_PW_pj:
        raise KeyError(f"'PW_99p' not found in dataset. Available variables: {list(ds_PW_pj.data_vars)}")

    years = np.unique(ds['valid_time'].dt.year.values)
    prec = ds['tot_prec']

    # True for cells that have at least one non-NaN tot_prec value (i.e. land cells)
    land_mask = prec.notnull().any(dim='valid_time')

    # .where() masks days below the threshold to NaN, leaving only exceedance days.
    # This produces NaN for both ocean cells and land cells with no exceedances in a year.
    # min_count=0 resolves this by returning 0 for any all-NaN group; the land_mask
    # then restores NaN specifically for ocean cells.
    P95WT = (
        prec
        .where(prec > ds_PW_pj['PW_95p'])
        .resample(valid_time='YE')
        .sum(dim='valid_time', skipna=True, min_count=0)
        .where(land_mask)
        .rename({'valid_time': 'year'})
        .rename('P95WT')
        .assign_coords(year=years)
        .assign_attrs(long_name='Total annual precipitation on very wet days (> PW_95p)', units='m')
    )

    P99WT = (
        prec
        .where(prec > ds_PW_pj['PW_99p'])
        .resample(valid_time='YE')
        .sum(dim='valid_time', skipna=True, min_count=0)
        .where(land_mask)
        .rename({'valid_time': 'year'})
        .rename('P99WT')
        .assign_coords(year=years)
        .assign_attrs(long_name='Total annual precipitation on extremely wet days (> PW_99p)', units='m')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset({'P95WT': P95WT, 'P99WT': P99WT})

def calculate_consecutive_precip_counts(ds: xr.Dataset, ds_PW_pj: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
           PW_95p  [Grid]       - 95th percentile wet day threshold
           PW_99p  [Grid]       - 99th percentile wet day threshold
    Output: CDD   [Grid, Year] - largest number of consecutive dry days (Pd < 1mm)
            CWD   [Grid, Year] - largest number of consecutive wet days (Pd >= 1mm)
            C95WD [Grid, Year] - largest number of consecutive very wet days (Pd > PW_95p)
            C99WD [Grid, Year] - largest number of consecutive extremely wet days (Pd > PW_99p)

    Calculates largest number of consecutive dry/wet/very wet/extremely wet days in a year.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec`.
    ds_PW_pj : xr.Dataset
        Dataset containing `PW_95p` and `PW_99p` with dimensions (latitude, longitude).
        Typically wet_days_perc (computed).

    Returns
    -------
    xr.Dataset
        Lazy Dataset with variables CDD, CWD, C95WD, C99WD
        and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating CDD, CWD, C95WD, C99WD (consecutive precipitation day counts at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'PW_95p' not in ds_PW_pj:
        raise KeyError(f"'PW_95p' not found in dataset. Available variables: {list(ds_PW_pj.data_vars)}")
    if 'PW_99p' not in ds_PW_pj:
        raise KeyError(f"'PW_99p' not found in dataset. Available variables: {list(ds_PW_pj.data_vars)}")

    # Precipitation variables keep Feb 29 (because the percentiles are not doy-based)
    # so years are 365 or 366 days long. Rechunk with the actual per-year day counts
    # so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk according to the number of days in each year
    prec = ds['tot_prec'].chunk({'valid_time': year_lengths.tolist()})

    # Thresholds are 2D (lat, lon) — rechunk to match prec spatial chunks so map_blocks aligns blocks correctly
    pw95 = ds_PW_pj['PW_95p'].chunk({'latitude': prec.chunksizes['latitude'], 'longitude': prec.chunksizes['longitude']})
    pw99 = ds_PW_pj['PW_99p'].chunk({'latitude': prec.chunksizes['latitude'], 'longitude': prec.chunksizes['longitude']})

    # pw95_block/pw99_block (thresholds) and da_block (daily values to be evaluated) should be aligned
    def process_block_counts(da_block, pw95_block, pw99_block):
        # Wet day threshold: 1mm expressed in metres
        wet_threshold = 0.001
        # Extract precipitation and threshold arrays for this block
        p = da_block.values # (n_days, n_lat, n_lon)
        pw95_arr = pw95_block.values # (n_lat, n_lon)
        pw99_arr = pw99_block.values # (n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = p.shape[1], p.shape[2]

        # Output arrays, initialised to NaN
        CDD   = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest dry spell (days)
        CWD   = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest wet spell (days)
        C95WD = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest very wet spell (days)
        C99WD = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # longest extremely wet spell (days)

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the precipitation data throughout the year for that location
                p_1d = p[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(p_1d)):
                    continue
                # Loop over the four conditions, each writing into its own output array
                for cond, out in [
                    (p_1d <  wet_threshold,   CDD  ),   # dry day
                    (p_1d >= wet_threshold,   CWD  ),   # wet day
                    (p_1d >  pw95_arr[i, j],  C95WD),   # very wet day
                    (p_1d >  pw99_arr[i, j],  C99WD),   # extremely wet day
                ]:
                    # Assign 0 if the condition is never met
                    if not np.any(cond):
                        out[0, i, j] = 0
                        continue
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    cond_int = cond.astype(np.int8)
                    padded   = np.concatenate([[0], cond_int, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a spell; -1 marks the end
                    d        = np.diff(padded)
                    # Get all the start and end dates
                    starts   = np.where(d ==  1)[0]
                    ends     = np.where(d == -1)[0]
                    # Get the length of each spell and keep the longest
                    lengths  = ends - starts
                    out[0, i, j] = lengths[np.argmax(lengths)]

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': da_block['valid_time'].values[[0]],
                  'latitude': da_block['latitude'].values,
                  'longitude': da_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({
            'CDD':   xr.DataArray(CDD,   dims=dims, coords=coords),
            'CWD':   xr.DataArray(CWD,   dims=dims, coords=coords),
            'C95WD': xr.DataArray(C95WD, dims=dims, coords=coords),
            'C99WD': xr.DataArray(C99WD, dims=dims, coords=coords),
        })

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = prec['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = prec.chunksizes['latitude']
    lon_chunks = prec.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(prec['latitude']), len(prec['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': prec['latitude'], 'longitude': prec['longitude']},
        )
        for var in ['CDD', 'CWD', 'C95WD', 'C99WD']
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block_counts, prec, args=[pw95, pw99], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)
    result = result.chunk({'year': 1})

    # Assigns attributes to each variable
    attrs = {
        'CDD':   ('Largest number of consecutive dry days (Pd < 1mm)', 'days'),
        'CWD':   ('Largest number of consecutive wet days (Pd >= 1mm)', 'days'),
        'C95WD': ('Largest number of consecutive very wet days (Pd > PW_95p)', 'days'),
        'C99WD': ('Largest number of consecutive extremely wet days (Pd > PW_99p)', 'days'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_consecutive_precip_totals(ds: xr.Dataset, ds_PW_pj: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
           PW_95p  [Grid]       - 95th percentile wet day threshold
           PW_99p  [Grid]       - 99th percentile wet day threshold
    Output: PCWD   [Grid, Year] - total precipitation during the longest consecutive wet day period
            PC95WD [Grid, Year] - total precipitation during the longest consecutive very wet day period
            PC99WD [Grid, Year] - total precipitation during the longest consecutive extremely wet day period

    Calculates total precipitation during the largest consecutive wet/very wet/extremely days in a year.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec`.
    ds_PW_pj : xr.Dataset
        Dataset containing `PW_95p` and `PW_99p` with dimensions (latitude, longitude).
        Typically wet_days_perc (computed).

    Returns
    -------
    xr.Dataset
        Lazy Dataset with variables PCWD, PC95WD, PC99WD
        and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PCWD, PC95WD, PC99WD (precipitation totals over longest consecutive wet periods at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'PW_95p' not in ds_PW_pj:
        raise KeyError(f"'PW_95p' not found in dataset. Available variables: {list(ds_PW_pj.data_vars)}")
    if 'PW_99p' not in ds_PW_pj:
        raise KeyError(f"'PW_99p' not found in dataset. Available variables: {list(ds_PW_pj.data_vars)}")

    # Precipitation keeps Feb 29, so years are 365 or 366 days long.
    # Rechunk with the actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    prec = ds['tot_prec'].chunk({'valid_time': year_lengths.tolist()})

    # Thresholds are 2D (lat, lon) — rechunk to match prec spatial chunks so map_blocks aligns blocks correctly
    pw95 = ds_PW_pj['PW_95p'].chunk({'latitude': prec.chunksizes['latitude'], 'longitude': prec.chunksizes['longitude']})
    pw99 = ds_PW_pj['PW_99p'].chunk({'latitude': prec.chunksizes['latitude'], 'longitude': prec.chunksizes['longitude']})

    # pw95_block/pw99_block (thresholds) and da_block (daily values to be evaluated) should be aligned
    def process_block_totals(da_block, pw95_block, pw99_block):
        # Wet day threshold: 1mm expressed in metres
        wet_threshold = 0.001
        # Extract precipitation and threshold arrays for this block
        p        = da_block.values              # (n_days, n_lat, n_lon)
        pw95_arr = pw95_block.values            # (n_lat, n_lon)
        pw99_arr = pw99_block.values            # (n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = p.shape[1], p.shape[2]

        # Output arrays, initialised to NaN
        PCWD   = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # precip total in longest wet spell
        PC95WD = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # precip total in longest very wet spell
        PC99WD = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)  # precip total in longest extremely wet spell

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the precipitation data throughout the year for that location
                p_1d = p[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(p_1d)):
                    continue
                # Loop over the three conditions, each writing into its own output array
                for cond, out in [
                    (p_1d >= wet_threshold,  PCWD  ),   # wet day
                    (p_1d >  pw95_arr[i, j], PC95WD),   # very wet day
                    (p_1d >  pw99_arr[i, j], PC99WD),   # extremely wet day
                ]:
                    # Assign 0 if the condition is never met
                    if not np.any(cond):
                        out[0, i, j] = 0.0
                        continue
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    cond_int = cond.astype(np.int8)
                    padded   = np.concatenate([[0], cond_int, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a spell; -1 marks the end
                    d        = np.diff(padded)
                    # Get all the start and end dates
                    starts   = np.where(d ==  1)[0]
                    ends     = np.where(d == -1)[0]
                    # Find the longest spell and sum precipitation over it
                    lengths  = ends - starts
                    k        = np.argmax(lengths)
                    out[0, i, j] = float(np.nansum(p_1d[starts[k]:ends[k]]))

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': da_block['valid_time'].values[[0]],
                  'latitude': da_block['latitude'].values,
                  'longitude': da_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({
            'PCWD':   xr.DataArray(PCWD,   dims=dims, coords=coords),
            'PC95WD': xr.DataArray(PC95WD, dims=dims, coords=coords),
            'PC99WD': xr.DataArray(PC99WD, dims=dims, coords=coords),
        })

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = prec['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = prec.chunksizes['latitude']
    lon_chunks = prec.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(prec['latitude']), len(prec['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': prec['latitude'], 'longitude': prec['longitude']},
        )
        for var in ['PCWD', 'PC95WD', 'PC99WD']
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block_totals, prec, args=[pw95, pw99], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)
    result = result.chunk({'year': 1})

    # Assigns attributes to each variable
    attrs = {
        'PCWD':   ('Total precipitation during longest consecutive wet day period', 'm'),
        'PC95WD': ('Total precipitation during longest consecutive very wet day period (> PW_95p)', 'm'),
        'PC99WD': ('Total precipitation during longest consecutive extremely wet day period (> PW_99p)', 'm'),
    }
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result

def calculate_PX1_PX5(ds: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
    Output: PX1 [Grid, Year] - maximum 1-day precipitation total within each year
            PX5 [Grid, Year] - maximum 5-day precipitation total within each year

    PX1 is the annual maximum of daily precipitation.
    PX5 is the annual maximum of rolling 5-day precipitation sums (min 5 valid days required).

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec`.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables `PX1` and `PX5` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PX1, PX5 (max 1-day and 5-day precipitation totals at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")

    years = np.unique(ds['valid_time'].dt.year.values)
    prec = ds['tot_prec']

    PX1 = (
        prec
        .resample(valid_time='YE')
        .max(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .rename('PX1')
        .assign_coords(year=years)
        .assign_attrs(long_name='Annual maximum 1-day precipitation', units='m')
    )

    PX5 = (
        prec
        .rolling(valid_time=5, min_periods=5)
        .sum()
        .resample(valid_time='YE')
        .max(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .rename('PX5')
        .assign_coords(year=years)
        .assign_attrs(long_name='Annual maximum 5-day precipitation total', units='m')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset({'PX1': PX1, 'PX5': PX5})

def calculate_PXM_PNM(ds_P_jm: xr.Dataset):
    """
    Input: P_jm [Grid, Month] - total monthly precipitation (output of calculate_P_jm)
    Output: PXM [Grid, Year] - maximum monthly total precipitation within each year
            PNM [Grid, Year] - minimum monthly total precipitation within each year

    Calculates maximum and minimum monthly precipitation within a year

    Parameters
    ----------
    ds_P_jm : xr.Dataset
        Lazy dataset containing the variable `P_jm` at monthly frequency.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables `PXM` and `PNM` and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating PXM and PNM (max/min monthly precipitation per year at grid cell level)")

    if 'P_jm' not in ds_P_jm:
        raise KeyError(f"'P_jm' not found in dataset. Available variables: {list(ds_P_jm.data_vars)}")

    PXM = (
        ds_P_jm['P_jm']
        .resample(valid_time='YE')
        .max(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .rename('PXM')
        .assign_coords(year=np.unique(ds_P_jm['valid_time'].dt.year.values))
        .assign_attrs(long_name='Maximum monthly total precipitation', units='m')
    )

    PNM = (
        ds_P_jm['P_jm']
        .resample(valid_time='YE')
        .min(dim='valid_time', skipna=True)
        .rename({'valid_time': 'year'})
        .rename('PNM')
        .assign_coords(year=np.unique(ds_P_jm['valid_time'].dt.year.values))
        .assign_attrs(long_name='Minimum monthly total precipitation', units='m')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset({'PXM': PXM, 'PNM': PNM})

def calculate_P_bins(ds: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level (metres)
    Output: P_1_to_10  [Grid, Year] - number of days with 1mm <= precipitation < 10mm
            P_10_to_20 [Grid, Year] - number of days with 10mm <= precipitation < 20mm
            P_above_20 [Grid, Year] - number of days with precipitation >= 20mm

    Bins are left-inclusive, right-exclusive. Thresholds are applied in metres
    (1mm = 0.001m, 10mm = 0.010m, 20mm = 0.020m).

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `tot_prec` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables P_1_to_10, P_10_to_20, P_above_20
        and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating P bin counts (daily precipitation distribution at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")

    years = np.unique(ds['valid_time'].dt.year.values)
    p = ds['tot_prec']

    bins = [
        ('P_1_to_10',  0.001, 0.010, 'Number of days with daily precipitation in [1mm, 10mm)'),
        ('P_10_to_20', 0.010, 0.020, 'Number of days with daily precipitation in [10mm, 20mm)'),
        ('P_above_20', 0.020, None,  'Number of days with daily precipitation >= 20mm'),
    ]

    result = {}
    for name, lower, upper, long_name in bins:
        if upper is None:
            cond = p >= lower
        else:
            cond = (p >= lower) & (p < upper)

        result[name] = (
            cond.where(p.notnull()).resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=long_name, units='days')
            .rename(name)
        )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset(result)


def calculate_coldwarm_nightsdays_abs(ds: xr.Dataset, ds_T_pkd_5w: xr.Dataset):
    """
    Input: t_min      [Grid, Day]       - daily minimum temperature at the grid level
           t_max      [Grid, Day]       - daily maximum temperature at the grid level
           TN_10p_5w  [Grid, DayOfYear] - 10th percentile of t_min in a 5-day window
           TX_10p_5w  [Grid, DayOfYear] - 10th percentile of t_max in a 5-day window
           TN_90p_5w  [Grid, DayOfYear] - 90th percentile of t_min in a 5-day window
           TX_90p_5w  [Grid, DayOfYear] - 90th percentile of t_max in a 5-day window
    Output: CN10_0C,  CN10_5C,  CN10_10C  [Grid, Year] - cold nights: t_min < TN_10p_5w and t_min < abs_thr
            CD10_10C, CD10_15C, CD10_20C  [Grid, Year] - cold days:   t_max < TX_10p_5w and t_max < abs_thr
            WN90_15C, WN90_20C, WN90_25C  [Grid, Year] - warm nights: t_min > TN_90p_5w and t_min > abs_thr
            WD90_25C, WD90_30C, WD90_35C  [Grid, Year] - warm days:   t_max > TX_90p_5w and t_max > abs_thr

    Extends CN10, CD10, WN90, WD90 with a combined relative+absolute threshold condition.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` and `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_T_pkd_5w : xr.Dataset
        Dataset containing `TN_10p_5w`, `TX_10p_5w`, `TN_90p_5w`, `TX_90p_5w`
        with a `day_of_year` dimension (1-366). Typically temp_perc (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with CN10_*C, CD10_*C, WN90_*C, WD90_*C variables
        and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating CN10_*C, CD10_*C, WN90_*C, WD90_*C (cold/warm night and day counts with absolute thresholds)")

    for var in ['t_min', 't_max']:
        if var not in ds:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in ['TN_10p_5w', 'TX_10p_5w', 'TN_90p_5w', 'TX_90p_5w']:
        if var not in ds_T_pkd_5w:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_T_pkd_5w.data_vars)}")

    # Remove Feb 29: the percentile thresholds are indexed 1-365, so Feb 29 has no valid threshold
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds.sel(valid_time=~feb29)

    # Map each day's normalized day-of-year to the corresponding percentile threshold.
    # For leap year dates after Feb 28, dt.dayofyear is 1 higher than the equivalent
    # non-leap year date; subtracting 1 gives a consistent DOY 1-365 for the threshold lookup.
    is_leap_and_late = ds['valid_time'].dt.is_leap_year & (ds['valid_time'].dt.month > 2)
    doy = ds['valid_time'].dt.dayofyear - is_leap_and_late.astype(int)
    # For each day in the time series, look up the corresponding percentile threshold
    # by its day-of-year. doy is an array of integers (1-365), one per timestep.
    # .sel(day_of_year=doy) maps each timestep to its threshold, producing arrays
    # of shape (time, latitude, longitude) aligned with the daily data.
    tn10 = ds_T_pkd_5w['TN_10p_5w'].sel(day_of_year=doy)
    tx10 = ds_T_pkd_5w['TX_10p_5w'].sel(day_of_year=doy)
    tn90 = ds_T_pkd_5w['TN_90p_5w'].sel(day_of_year=doy)
    tx90 = ds_T_pkd_5w['TX_90p_5w'].sel(day_of_year=doy)

    years = np.unique(ds['valid_time'].dt.year.values)

    result = {}

    for thr in [0, 5, 10]:
        name = f'CN10_{thr}C'
        result[name] = (
            ((ds['t_min'] < tn10) & (ds['t_min'] < thr)).where(ds['t_min'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of cold nights (t_min < TN_10p_5w and t_min < {thr}°C)', units='days')
            .rename(name)
        )

    for thr in [10, 15, 20]:
        name = f'CD10_{thr}C'
        result[name] = (
            ((ds['t_max'] < tx10) & (ds['t_max'] < thr)).where(ds['t_max'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of cold days (t_max < TX_10p_5w and t_max < {thr}°C)', units='days')
            .rename(name)
        )

    for thr in [15, 20, 25]:
        name = f'WN90_{thr}C'
        result[name] = (
            ((ds['t_min'] > tn90) & (ds['t_min'] > thr)).where(ds['t_min'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of warm nights (t_min > TN_90p_5w and t_min > {thr}°C)', units='days')
            .rename(name)
        )

    for thr in [25, 30, 35]:
        name = f'WD90_{thr}C'
        result[name] = (
            ((ds['t_max'] > tx90) & (ds['t_max'] > thr)).where(ds['t_max'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of warm days (t_max > TX_90p_5w and t_max > {thr}°C)', units='days')
            .rename(name)
        )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset(result)


def calculate_day_heatwaves_abs(ds: xr.Dataset, ds_TX90p15w: xr.Dataset):
    """
    Input: t_max       [Grid, Day]       - daily maximum temperature at the grid level
           TX_90p_15w  [Grid, DayOfYear] - 90th percentile of t_max in a 15-day window
    Output: DDHW_25C, DDHW_30C, DDHW_35C [Grid, Year] - days in heatwave events with absolute t_max threshold
            LDHW_25C, LDHW_30C, LDHW_35C [Grid, Year] - days in longest heatwave with absolute t_max threshold
            NDHW_25C, NDHW_30C, NDHW_35C [Grid, Year] - number of heatwave events with absolute t_max threshold
            TDHW_25C, TDHW_30C, TDHW_35C [Grid, Year] - average TX during heatwave days with absolute threshold

    Extends day heatwave indicators with a combined relative+absolute threshold condition.
    A heatwave event requires at least 3 consecutive days where t_max > TX_90p_15w AND t_max > abs_thr.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_TX90p15w : xr.Dataset
        Dataset containing `TX_90p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TX_90p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DDHW_*C, LDHW_*C, NDHW_*C, TDHW_*C variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS = [25, 30, 35]
    all_vars = [f'{base}_{t}C' for t in THRESHOLDS for base in ['DDHW', 'LDHW', 'NDHW', 'TDHW']]
    logger.info(f"Calculating DDHW_*C, LDHW_*C, NDHW_*C, TDHW_*C (day heatwave indicators with absolute thresholds at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TX_90p_15w' not in ds_TX90p15w:
        raise KeyError(f"'TX_90p_15w' not found in dataset. Available variables: {list(ds_TX90p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_max']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tx90_table = ds_TX90p15w['TX_90p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tx90_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tx90_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmax for this block
        tx = ds_block['t_max'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tx90_arr = tx90_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue
                # Look up the daily 90th-percentile threshold for each day of this year at that grid point
                tx90_1d = tx90_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing

                for thr in THRESHOLDS:
                    suffix = f'_{thr}C'
                    # Creates binary variable: 1 where t_max exceeds both the daily 90th percentile and the absolute threshold, 0 otherwise
                    hot = ((tx_1d > tx90_1d) & (tx_1d > thr)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], hot, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a hot spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of heatwaves
                    lengths = ends - starts
                    # A heatwave requires at least 3 consecutive hot days, so mask
                    hw_mask = lengths >= 3
                    # Assign 0 values if no heatwave is detected
                    if not hw_mask.any():
                        out[f'DDHW{suffix}'][0, i, j] = 0
                        out[f'LDHW{suffix}'][0, i, j] = 0
                        out[f'NDHW{suffix}'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the end date of the valid heatwaves only
                    hw_starts  = starts[hw_mask]
                    hw_lengths = lengths[hw_mask]
                    # Calculate the variables
                    out[f'DDHW{suffix}'][0, i, j] = hw_lengths.sum() # total days across all heatwave events
                    out[f'LDHW{suffix}'][0, i, j] = hw_lengths.max() # duration of the longest single event
                    out[f'NDHW{suffix}'][0, i, j] = hw_mask.sum() # count of distinct events
                    # Concatenate the actual t_max values from all heatwave periods to compute mean
                    hw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                    # Calculates mean
                    out[f'TDHW{suffix}'][0, i, j] = hw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx90_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for thr in THRESHOLDS:
        suffix = f'_{thr}C'
        attrs = {
            f'DDHW{suffix}': (f'Number of days in day heatwave events (TX > TX_90p_15w and TX > {thr}°C for >= 3 consecutive days)', 'days'),
            f'LDHW{suffix}': (f'Number of days in the longest day heatwave event of the year (TX > TX_90p_15w and TX > {thr}°C)', 'days'),
            f'NDHW{suffix}': (f'Number of day heatwave events in the year (TX > TX_90p_15w and TX > {thr}°C)', 'count'),
            f'TDHW{suffix}': (f'Average TX during day heatwave days (TX > TX_90p_15w and TX > {thr}°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_night_heatwaves_abs(ds: xr.Dataset, ds_TN90p15w: xr.Dataset):
    """
    Input: t_min       [Grid, Day]       - daily minimum temperature at the grid level
           TN_90p_15w  [Grid, DayOfYear] - 90th percentile of t_min in a 15-day window
    Output: DNHW_15C, DNHW_20C, DNHW_25C [Grid, Year] - nights in heatwave events with absolute t_min threshold
            LNHW_15C, LNHW_20C, LNHW_25C [Grid, Year] - nights in longest heatwave with absolute t_min threshold
            NNHW_15C, NNHW_20C, NNHW_25C [Grid, Year] - number of heatwave events with absolute t_min threshold
            TNHW_15C, TNHW_20C, TNHW_25C [Grid, Year] - average TN during heatwave nights with absolute threshold

    Extends night heatwave indicators with a combined relative+absolute threshold condition.
    A heatwave event requires at least 3 consecutive nights where t_min > TN_90p_15w AND t_min > abs_thr.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with
        a `valid_time` dimension (daily frequency).
    ds_TN90p15w : xr.Dataset
        Dataset containing `TN_90p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TN_90p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DNHW_*C, LNHW_*C, NNHW_*C, TNHW_*C variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS = [15, 20, 25]
    all_vars = [f'{base}_{t}C' for t in THRESHOLDS for base in ['DNHW', 'LNHW', 'NNHW', 'TNHW']]
    logger.info("Calculating DNHW_*C, LNHW_*C, NNHW_*C, TNHW_*C (night heatwave indicators with absolute thresholds at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TN_90p_15w' not in ds_TN90p15w:
        raise KeyError(f"'TN_90p_15w' not found in dataset. Available variables: {list(ds_TN90p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_min']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tn90_table = ds_TN90p15w['TN_90p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tn90_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tn90_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmin for this block
        tn = ds_block['t_min'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tn90_arr = tn90_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue
                # Look up the daily 90th-percentile threshold for each day of this year at that grid point
                tn90_1d = tn90_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing

                for thr in THRESHOLDS:
                    suffix = f'_{thr}C'
                    # Creates binary variable: 1 where t_min exceeds both the daily 90th percentile and the absolute threshold, 0 otherwise
                    hot = ((tn_1d > tn90_1d) & (tn_1d > thr)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], hot, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a hot spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of heatwaves
                    lengths = ends - starts
                    # A heatwave requires at least 3 consecutive hot nights, so mask
                    hw_mask = lengths >= 3
                    # Assign 0 values if no heatwave is detected
                    if not hw_mask.any():
                        out[f'DNHW{suffix}'][0, i, j] = 0
                        out[f'LNHW{suffix}'][0, i, j] = 0
                        out[f'NNHW{suffix}'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the end date of the valid heatwaves only
                    hw_starts  = starts[hw_mask]
                    hw_lengths = lengths[hw_mask]
                    # Calculate the variables
                    out[f'DNHW{suffix}'][0, i, j] = hw_lengths.sum() # total nights across all heatwave events
                    out[f'LNHW{suffix}'][0, i, j] = hw_lengths.max() # duration of the longest single event
                    out[f'NNHW{suffix}'][0, i, j] = hw_mask.sum() # count of distinct events
                    # Concatenate the actual t_min values from all heatwave periods to compute mean
                    hw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                    # Calculates mean
                    out[f'TNHW{suffix}'][0, i, j] = hw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn90_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for thr in THRESHOLDS:
        suffix = f'_{thr}C'
        attrs = {
            f'DNHW{suffix}': (f'Number of nights in night heatwave events (TN > TN_90p_15w and TN > {thr}°C for >= 3 consecutive nights)', 'nights'),
            f'LNHW{suffix}': (f'Number of nights in the longest night heatwave event of the year (TN > TN_90p_15w and TN > {thr}°C)', 'nights'),
            f'NNHW{suffix}': (f'Number of night heatwave events in the year (TN > TN_90p_15w and TN > {thr}°C)', 'count'),
            f'TNHW{suffix}': (f'Average TN during night heatwave nights (TN > TN_90p_15w and TN > {thr}°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_day_coldwaves_abs(ds: xr.Dataset, ds_TX10p15w: xr.Dataset):
    """
    Input: t_max       [Grid, Day]       - daily maximum temperature at the grid level
           TX_10p_15w  [Grid, DayOfYear] - 10th percentile of t_max in a 15-day window
    Output: DDCW_10C, DDCW_15C, DDCW_20C [Grid, Year] - days in coldwave events with absolute t_max threshold
            LDCW_10C, LDCW_15C, LDCW_20C [Grid, Year] - days in longest coldwave with absolute t_max threshold
            NDCW_10C, NDCW_15C, NDCW_20C [Grid, Year] - number of coldwave events with absolute t_max threshold
            TDCW_10C, TDCW_15C, TDCW_20C [Grid, Year] - average TX during coldwave days with absolute threshold

    Extends day coldwave indicators with a combined relative+absolute threshold condition.
    A coldwave event requires at least 3 consecutive days where t_max < TX_10p_15w AND t_max < abs_thr.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_TX10p15w : xr.Dataset
        Dataset containing `TX_10p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TX_10p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DDCW_*C, LDCW_*C, NDCW_*C, TDCW_*C variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS = [10, 15, 20]
    all_vars = [f'{base}_{t}C' for t in THRESHOLDS for base in ['DDCW', 'LDCW', 'NDCW', 'TDCW']]
    logger.info("Calculating DDCW_*C, LDCW_*C, NDCW_*C, TDCW_*C (day coldwave indicators with absolute thresholds at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TX_10p_15w' not in ds_TX10p15w:
        raise KeyError(f"'TX_10p_15w' not found in dataset. Available variables: {list(ds_TX10p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_max']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tx10_table = ds_TX10p15w['TX_10p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tx10_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tx10_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmax for this block
        tx = ds_block['t_max'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tx10_arr = tx10_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue
                # Look up the daily 10th-percentile threshold for each day of this year at that grid point
                tx10_1d = tx10_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing

                for thr in THRESHOLDS:
                    suffix = f'_{thr}C'
                    # Creates binary variable: 1 where t_max falls below both the daily 10th percentile and the absolute threshold, 0 otherwise
                    cold = ((tx_1d < tx10_1d) & (tx_1d < thr)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of coldwaves
                    lengths = ends - starts
                    # A coldwave requires at least 3 consecutive cold days, so mask
                    cw_mask = lengths >= 3
                    # Assign 0 values if no coldwave is detected
                    if not cw_mask.any():
                        out[f'DDCW{suffix}'][0, i, j] = 0
                        out[f'LDCW{suffix}'][0, i, j] = 0
                        out[f'NDCW{suffix}'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the end date of the valid coldwaves only
                    cw_starts  = starts[cw_mask]
                    cw_lengths = lengths[cw_mask]
                    # Calculate the variables
                    out[f'DDCW{suffix}'][0, i, j] = cw_lengths.sum() # total days across all coldwave events
                    out[f'LDCW{suffix}'][0, i, j] = cw_lengths.max() # duration of the longest single event
                    out[f'NDCW{suffix}'][0, i, j] = cw_mask.sum() # count of distinct events
                    # Concatenate the actual t_max values from all coldwave periods to compute mean
                    cw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                    # Calculates mean
                    out[f'TDCW{suffix}'][0, i, j] = cw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx10_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for thr in THRESHOLDS:
        suffix = f'_{thr}C'
        attrs = {
            f'DDCW{suffix}': (f'Number of days in day coldwave events (TX < TX_10p_15w and TX < {thr}°C for >= 3 consecutive days)', 'days'),
            f'LDCW{suffix}': (f'Number of days in the longest day coldwave event of the year (TX < TX_10p_15w and TX < {thr}°C)', 'days'),
            f'NDCW{suffix}': (f'Number of day coldwave events in the year (TX < TX_10p_15w and TX < {thr}°C)', 'count'),
            f'TDCW{suffix}': (f'Average TX during day coldwave days (TX < TX_10p_15w and TX < {thr}°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_night_coldwaves_abs(ds: xr.Dataset, ds_TN10p15w: xr.Dataset):
    """
    Input: t_min       [Grid, Day]       - daily minimum temperature at the grid level
           TN_10p_15w  [Grid, DayOfYear] - 10th percentile of t_min in a 15-day window
    Output: DNCW_0C,  DNCW_5C,  DNCW_10C  [Grid, Year] - nights in coldwave events with absolute t_min threshold
            LNCW_0C,  LNCW_5C,  LNCW_10C  [Grid, Year] - nights in longest coldwave with absolute t_min threshold
            NNCW_0C,  NNCW_5C,  NNCW_10C  [Grid, Year] - number of coldwave events with absolute t_min threshold
            TNCW_0C,  TNCW_5C,  TNCW_10C  [Grid, Year] - average TN during coldwave nights with absolute threshold

    Extends night coldwave indicators with a combined relative+absolute threshold condition.
    A coldwave event requires at least 3 consecutive nights where t_min < TN_10p_15w AND t_min < abs_thr.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with
        a `valid_time` dimension (daily frequency).
    ds_TN10p15w : xr.Dataset
        Dataset containing `TN_10p_15w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TN_10p_15w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DNCW_*C, LNCW_*C, NNCW_*C, TNCW_*C variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS = [0, 5, 10]
    all_vars = [f'{base}_{t}C' for t in THRESHOLDS for base in ['DNCW', 'LNCW', 'NNCW', 'TNCW']]
    logger.info("Calculating DNCW_*C, LNCW_*C, NNCW_*C, TNCW_*C (night coldwave indicators with absolute thresholds at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TN_10p_15w' not in ds_TN10p15w:
        raise KeyError(f"'TN_10p_15w' not found in dataset. Available variables: {list(ds_TN10p15w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_min']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tn10_table = ds_TN10p15w['TN_10p_15w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tn10_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tn10_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmin for this block
        tn = ds_block['t_min'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tn10_arr = tn10_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue
                # Look up the daily 10th-percentile threshold for each day of this year at that grid point
                tn10_1d = tn10_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing

                for thr in THRESHOLDS:
                    suffix = f'_{thr}C'
                    # Creates binary variable: 1 where t_min falls below both the daily 10th percentile and the absolute threshold, 0 otherwise
                    cold = ((tn_1d < tn10_1d) & (tn_1d < thr)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of coldwaves
                    lengths = ends - starts
                    # A coldwave requires at least 3 consecutive cold nights, so mask
                    cw_mask = lengths >= 3
                    # Assign 0 values if no coldwave is detected
                    if not cw_mask.any():
                        out[f'DNCW{suffix}'][0, i, j] = 0
                        out[f'LNCW{suffix}'][0, i, j] = 0
                        out[f'NNCW{suffix}'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the end date of the valid coldwaves only
                    cw_starts  = starts[cw_mask]
                    cw_lengths = lengths[cw_mask]
                    # Calculate the variables
                    out[f'DNCW{suffix}'][0, i, j] = cw_lengths.sum() # total nights across all coldwave events
                    out[f'LNCW{suffix}'][0, i, j] = cw_lengths.max() # duration of the longest single event
                    out[f'NNCW{suffix}'][0, i, j] = cw_mask.sum() # count of distinct events
                    # Concatenate the actual t_min values from all coldwave periods to compute mean
                    cw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                    # Calculates mean
                    out[f'TNCW{suffix}'][0, i, j] = cw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn10_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for thr in THRESHOLDS:
        suffix = f'_{thr}C'
        attrs = {
            f'DNCW{suffix}': (f'Number of nights in night coldwave events (TN < TN_10p_15w and TN < {thr}°C for >= 3 consecutive nights)', 'nights'),
            f'LNCW{suffix}': (f'Number of nights in the longest night coldwave event of the year (TN < TN_10p_15w and TN < {thr}°C)', 'nights'),
            f'NNCW{suffix}': (f'Number of night coldwave events in the year (TN < TN_10p_15w and TN < {thr}°C)', 'count'),
            f'TNCW{suffix}': (f'Average TN during night coldwave nights (TN < TN_10p_15w and TN < {thr}°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_CSD_abs(ds: xr.Dataset, ds_TN10p5w: xr.Dataset):
    """
    Input: t_min      [Grid, Day]       - daily minimum temperature at the grid level
           TN_10p_5w  [Grid, DayOfYear] - 10th percentile of t_min in a 5-day window
    Output: CSD_15C, CSD_20C [Grid, Year] - days in cold spell events with absolute t_min threshold

    Extends CSD with a combined relative+absolute threshold condition.
    A cold spell requires at least 6 consecutive days where t_min < TN_10p_5w AND t_min < abs_thr.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with
        a `valid_time` dimension (daily frequency).
    ds_TN10p5w : xr.Dataset
        Dataset containing `TN_10p_5w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TN_10p_5w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with CSD_15C, CSD_20C variables and dimensions (year, latitude, longitude).
    """
    THRESHOLDS = [15, 20]
    all_vars = [f'CSD_{t}C' for t in THRESHOLDS]
    logger.info("Calculating CSD_*C (cold spell duration with absolute thresholds at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TN_10p_5w' not in ds_TN10p5w:
        raise KeyError(f"'TN_10p_5w' not found in dataset. Available variables: {list(ds_TN10p5w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_min']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tn10_table = ds_TN10p5w['TN_10p_5w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tn10_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tn10_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmin for this block
        tn = ds_block['t_min'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tn10_arr = tn10_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue
                # Look up the daily 10th-percentile threshold for each day of this year at that grid point
                tn10_1d = tn10_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing

                for thr in THRESHOLDS:
                    name = f'CSD_{thr}C'
                    # Creates binary variable: 1 where t_min falls below both the daily 10th percentile and the absolute threshold, 0 otherwise
                    cold = ((tn_1d < tn10_1d) & (tn_1d < thr)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each spell
                    lengths = ends - starts
                    # A cold spell requires at least 6 consecutive cold days, so mask
                    spell_mask = lengths >= 6
                    # Sum the total days across all valid cold spells (0 if none)
                    out[name][0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn10_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {f'CSD_{t}C': (f'Number of days in cold spell events (TN < TN_10p_5w and TN < {t}°C for >= 6 consecutive days)', 'days')
             for t in THRESHOLDS}
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_WSD_abs(ds: xr.Dataset, ds_TX90p5w: xr.Dataset):
    """
    Input: t_max      [Grid, Day]       - daily maximum temperature at the grid level
           TX_90p_5w  [Grid, DayOfYear] - 90th percentile of t_max in a 5-day window
    Output: WSD_15C, WSD_20C [Grid, Year] - days in warm spell events with absolute t_max threshold

    Extends WSD with a combined relative+absolute threshold condition.
    A warm spell requires at least 6 consecutive days where t_max > TX_90p_5w AND t_max > abs_thr.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_TX90p5w : xr.Dataset
        Dataset containing `TX_90p_5w` with dimensions (day_of_year, latitude, longitude).
        Typically temp_perc[['TX_90p_5w']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with WSD_15C, WSD_20C variables and dimensions (year, latitude, longitude).
    """
    THRESHOLDS = [15, 20]
    all_vars = [f'WSD_{t}C' for t in THRESHOLDS]
    logger.info("Calculating WSD_*C (warm spell duration with absolute thresholds at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 'TX_90p_5w' not in ds_TX90p5w:
        raise KeyError(f"'TX_90p_5w' not found in dataset. Available variables: {list(ds_TX90p5w.data_vars)}")

    # Remove Feb 29 and rechunk to one complete year per time block.
    # After removing Feb 29, every year has exactly 365 days, so chunk(valid_time=365) produces one block per year
    feb29 = (ds['valid_time'].dt.month == 2) & (ds['valid_time'].dt.day == 29)
    ds = ds[['t_max']].sel(valid_time=~feb29).chunk({'valid_time': 365})

    # Get an array of years present in the data
    unique_years = np.unique(ds['valid_time'].dt.year.values)
    # Get the number of years
    n_years = len(unique_years)
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly.
    # day_of_year must be a single chunk (-1) so that map_blocks passes all 365 thresholds to every block.
    tx90_table = ds_TX90p5w['TX_90p_5w'].chunk({'day_of_year': -1, 'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    # tx90_block (percentiles) and ds_block (daily values to be evaluated) should be aligned
    def process_block(ds_block, tx90_block):

        # Get the time steps in this block
        valid_time = ds_block['valid_time']

        # Normalize DOY to 1-365: leap-year dates after Feb 28 would be 1 day ahead
        # of the equivalent non-leap date, so subtract 1 to keep the threshold lookup consistent
        is_leap_and_late = valid_time.dt.is_leap_year & (valid_time.dt.month > 2)
        doy = (valid_time.dt.dayofyear - is_leap_and_late.astype(int)).values # shape (365,)

        # Extract tmax for this block
        tx = ds_block['t_max'].values # shape (n_days, n_lat, n_lon) - n_days is a multiple of 365
        # Extract the percentiles for this block
        tx90_arr = tx90_block.values # (365, n_lat, n_lon)
        # Get the number of latitude and longitudes coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NA (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue
                # Look up the daily 90th-percentile threshold for each day of this year at that grid point
                tx90_1d = tx90_arr[doy - 1, i, j]  # doy is 1-based, so subtract 1 for python 0-based indexing

                for thr in THRESHOLDS:
                    name = f'WSD_{thr}C'
                    # Creates binary variable: 1 where t_max exceeds both the daily 90th percentile and the absolute threshold, 0 otherwise
                    warm = ((tx_1d > tx90_1d) & (tx_1d > thr)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], warm, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a warm spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each spell
                    lengths = ends - starts
                    # A warm spell requires at least 6 consecutive warm days, so mask
                    spell_mask = lengths >= 6
                    # Sum the total days across all valid warm spells (0 if none)
                    out[name][0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (to be assigned to the output coordinates)
    year_first_dates = ds['valid_time'].values.reshape(n_years, 365)[:, 0]
    # Gets spatial chunks sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx90_table], template=template)

    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {f'WSD_{t}C': (f'Number of days in warm spell events (TX > TX_90p_5w and TX > {t}°C for >= 6 consecutive days)', 'days')
             for t in THRESHOLDS}
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_coldwarm_nightsdays_wy(ds: xr.Dataset, ds_T_wy: xr.Dataset):
    """
    Input: t_min  [Grid, Day] - daily minimum temperature at the grid level
           t_max  [Grid, Day] - daily maximum temperature at the grid level
           TN{p}  [Grid]      - p-th full-year percentile of t_min (no day_of_year dimension)
           TX{p}  [Grid]      - p-th full-year percentile of t_max (no day_of_year dimension)
    Output: CN10_wy, CN5_wy, CN2_5_wy, CN1_wy    [Grid, Year] - cold nights: t_min < TN{p}
            CD10_wy, CD5_wy, CD2_5_wy, CD1_wy    [Grid, Year] - cold days:   t_max < TX{p}
            WN90_wy, WN95_wy, WN97_5_wy, WN99_wy [Grid, Year] - warm nights: t_min > TN{p}
            WD90_wy, WD95_wy, WD97_5_wy, WD99_wy [Grid, Year] - warm days:   t_max > TX{p}

    Uses full-year unconditional percentiles (single value per grid cell, no day-of-year
    dimension). The 2D threshold is broadcast against the 3D daily data by xarray,
    so no Feb 29 removal or DOY normalisation is needed.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` and `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_T_wy : xr.Dataset
        Dataset containing full-year percentiles TN1, TN2_5, TN5, TN10,
        TN90, TN95, TN97_5, TN99, TX1, TX2_5, TX5, TX10, TX90, TX95, TX97_5, TX99
        with dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with CN*_wy, CD*_wy, WN*_wy, WD*_wy variables
        and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating CN*_wy, CD*_wy, WN*_wy, WD*_wy (cold/warm night and day counts with full-year percentile thresholds)")

    for var in ['t_min', 't_max']:
        if var not in ds:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds.data_vars)}")
    required_perc = ['TN1', 'TN2_5', 'TN5', 'TN10', 'TN90', 'TN95', 'TN97_5', 'TN99',
                     'TX1', 'TX2_5', 'TX5', 'TX10', 'TX90', 'TX95', 'TX97_5', 'TX99']
    for var in required_perc:
        if var not in ds_T_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_T_wy.data_vars)}")

    # Get an array of years present in the data
    years = np.unique(ds['valid_time'].dt.year.values)

    result = {}

    # The full-year threshold has dims (latitude, longitude) only; xarray broadcasts it against
    # the (valid_time, latitude, longitude) daily data automatically — no DOY lookup needed.
    for p in [10, 5, 2.5, 1]:
        p_str = str(p).replace('.', '_')
        name = f'CN{p_str}_wy'
        result[name] = (
            (ds['t_min'] < ds_T_wy[f'TN{p_str}']).where(ds['t_min'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of cold nights (t_min < TN{p_str} full-year percentile)', units='days')
            .rename(name)
        )

    for p in [10, 5, 2.5, 1]:
        p_str = str(p).replace('.', '_')
        name = f'CD{p_str}_wy'
        result[name] = (
            (ds['t_max'] < ds_T_wy[f'TX{p_str}']).where(ds['t_max'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of cold days (t_max < TX{p_str} full-year percentile)', units='days')
            .rename(name)
        )

    for p in [90, 95, 97.5, 99]:
        p_str = str(p).replace('.', '_')
        name = f'WN{p_str}_wy'
        result[name] = (
            (ds['t_min'] > ds_T_wy[f'TN{p_str}']).where(ds['t_min'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of warm nights (t_min > TN{p_str} full-year percentile)', units='days')
            .rename(name)
        )

    for p in [90, 95, 97.5, 99]:
        p_str = str(p).replace('.', '_')
        name = f'WD{p_str}_wy'
        result[name] = (
            (ds['t_max'] > ds_T_wy[f'TX{p_str}']).where(ds['t_max'].notnull())
            .resample(valid_time='YE').sum(min_count=1)
            .rename({'valid_time': 'year'}).assign_coords(year=years)
            .drop_attrs()
            .assign_attrs(long_name=f'Number of warm days (t_max > TX{p_str} full-year percentile)', units='days')
            .rename(name)
        )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset(result)


def calculate_day_heatwaves_wy(ds: xr.Dataset, ds_TX_wy: xr.Dataset):
    """
    Input: t_max   [Grid, Day] - daily maximum temperature at the grid level
           TX{p}   [Grid]      - p-th full-year percentile of t_max
    Output: DDHW95_wy, DDHW97_5_wy, DDHW99_wy [Grid, Year] - days in heatwave events
            LDHW95_wy, LDHW97_5_wy, LDHW99_wy [Grid, Year] - days in longest heatwave
            NDHW95_wy, NDHW97_5_wy, NDHW99_wy [Grid, Year] - number of heatwave events
            TDHW95_wy, TDHW97_5_wy, TDHW99_wy [Grid, Year] - average TX during heatwave days

    A heatwave event requires at least 3 consecutive days where t_max > TX{p}.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_max` with a `valid_time` dimension.
    ds_TX_wy : xr.Dataset
        Dataset containing TX95, TX97_5, TX99 with dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DDHW_*_wy, LDHW_*_wy, NDHW_*_wy, TDHW_*_wy variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS  = [95, 97.5, 99]
    THRESH_STRS = [str(p).replace('.', '_') for p in THRESHOLDS]
    all_vars = [f'{base}_{p_str}_wy' for p_str in THRESH_STRS for base in ['DDHW', 'LDHW', 'NDHW', 'TDHW']]
    logger.info("Calculating DDHW_*_wy, LDHW_*_wy, NDHW_*_wy, TDHW_*_wy (day heatwave indicators with full-year percentile thresholds at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in [f'TX{p_str}' for p_str in THRESH_STRS]:
        if var not in ds_TX_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_TX_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_max']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tx_wy = ds_TX_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tx_block):

        # Extract tmax for this block
        tx = ds_block['t_max'].values  # shape (365, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue

                for p_str in THRESH_STRS:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    threshold = tx_block[f'TX{p_str}'].values[i, j]
                    # Creates binary variable: 1 where t_max exceeds the full-year percentile threshold, 0 otherwise
                    hot = (tx_1d > threshold).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], hot, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a hot spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each heatwave
                    lengths = ends - starts
                    # A heatwave requires at least 3 consecutive hot days, so mask
                    hw_mask = lengths >= 3
                    # Assign 0 values if no heatwave is detected
                    if not hw_mask.any():
                        out[f'DDHW_{p_str}_wy'][0, i, j] = 0
                        out[f'LDHW_{p_str}_wy'][0, i, j] = 0
                        out[f'NDHW_{p_str}_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid heatwaves only
                    hw_starts  = starts[hw_mask]
                    hw_lengths = lengths[hw_mask]
                    # Calculate the variables
                    out[f'DDHW_{p_str}_wy'][0, i, j] = hw_lengths.sum()   # total days across all heatwave events
                    out[f'LDHW_{p_str}_wy'][0, i, j] = hw_lengths.max()   # duration of the longest single event
                    out[f'NDHW_{p_str}_wy'][0, i, j] = hw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_max values from all heatwave periods to compute mean
                    hw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                    # Calculates mean
                    out[f'TDHW_{p_str}_wy'][0, i, j] = hw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str in THRESH_STRS:
        attrs = {
            f'DDHW_{p_str}_wy': (f'Number of days in day heatwave events (TX > TX{p_str} full-year percentile, >= 3 consecutive days)', 'days'),
            f'LDHW_{p_str}_wy': (f'Number of days in the longest day heatwave event (TX > TX{p_str} full-year percentile)', 'days'),
            f'NDHW_{p_str}_wy': (f'Number of day heatwave events (TX > TX{p_str} full-year percentile)', 'count'),
            f'TDHW_{p_str}_wy': (f'Average TX during day heatwave days (TX > TX{p_str} full-year percentile)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_night_heatwaves_wy(ds: xr.Dataset, ds_TN_wy: xr.Dataset):
    """
    Input: t_min   [Grid, Day] - daily minimum temperature at the grid level
           TN{p}   [Grid]      - p-th full-year percentile of t_min
    Output: DNHW95_wy, DNHW97_5_wy, DNHW99_wy [Grid, Year] - nights in heatwave events
            LNHW95_wy, LNHW97_5_wy, LNHW99_wy [Grid, Year] - nights in longest heatwave
            NNHW95_wy, NNHW97_5_wy, NNHW99_wy [Grid, Year] - number of heatwave events
            TNHW95_wy, TNHW97_5_wy, TNHW99_wy [Grid, Year] - average TN during heatwave nights

    A heatwave event requires at least 3 consecutive nights where t_min > TN{p}.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` with a `valid_time` dimension.
    ds_TN_wy : xr.Dataset
        Dataset containing TN95, TN97_5, TN99 with dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DNHW_*_wy, LNHW_*_wy, NNHW_*_wy, TNHW_*_wy variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS  = [95, 97.5, 99]
    THRESH_STRS = [str(p).replace('.', '_') for p in THRESHOLDS]
    all_vars = [f'{base}_{p_str}_wy' for p_str in THRESH_STRS for base in ['DNHW', 'LNHW', 'NNHW', 'TNHW']]
    logger.info("Calculating DNHW_*_wy, LNHW_*_wy, NNHW_*_wy, TNHW_*_wy (night heatwave indicators with full-year percentile thresholds at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in [f'TN{p_str}' for p_str in THRESH_STRS]:
        if var not in ds_TN_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_TN_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_min']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tn_wy = ds_TN_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tn_block):

        # Extract tmin for this block
        tn = ds_block['t_min'].values  # shape (365, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue

                for p_str in THRESH_STRS:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    threshold = tn_block[f'TN{p_str}'].values[i, j]
                    # Creates binary variable: 1 where t_min exceeds the full-year percentile threshold, 0 otherwise
                    hot = (tn_1d > threshold).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], hot, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a hot spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each heatwave
                    lengths = ends - starts
                    # A heatwave requires at least 3 consecutive hot nights, so mask
                    hw_mask = lengths >= 3
                    # Assign 0 values if no heatwave is detected
                    if not hw_mask.any():
                        out[f'DNHW_{p_str}_wy'][0, i, j] = 0
                        out[f'LNHW_{p_str}_wy'][0, i, j] = 0
                        out[f'NNHW_{p_str}_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid heatwaves only
                    hw_starts  = starts[hw_mask]
                    hw_lengths = lengths[hw_mask]
                    # Calculate the variables
                    out[f'DNHW_{p_str}_wy'][0, i, j] = hw_lengths.sum()   # total nights across all heatwave events
                    out[f'LNHW_{p_str}_wy'][0, i, j] = hw_lengths.max()   # duration of the longest single event
                    out[f'NNHW_{p_str}_wy'][0, i, j] = hw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_min values from all heatwave periods to compute mean
                    hw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                    # Calculates mean
                    out[f'TNHW_{p_str}_wy'][0, i, j] = hw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str in THRESH_STRS:
        attrs = {
            f'DNHW_{p_str}_wy': (f'Number of nights in night heatwave events (TN > TN{p_str} full-year percentile, >= 3 consecutive nights)', 'nights'),
            f'LNHW_{p_str}_wy': (f'Number of nights in the longest night heatwave event (TN > TN{p_str} full-year percentile)', 'nights'),
            f'NNHW_{p_str}_wy': (f'Number of night heatwave events (TN > TN{p_str} full-year percentile)', 'count'),
            f'TNHW_{p_str}_wy': (f'Average TN during night heatwave nights (TN > TN{p_str} full-year percentile)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_day_coldwaves_wy(ds: xr.Dataset, ds_TX_wy: xr.Dataset):
    """
    Input: t_max   [Grid, Day] - daily maximum temperature at the grid level
           TX{p}   [Grid]      - p-th full-year percentile of t_max
    Output: DDCW5_wy, DDCW2_5_wy, DDCW1_wy [Grid, Year] - days in coldwave events
            LDCW5_wy, LDCW2_5_wy, LDCW1_wy [Grid, Year] - days in longest coldwave
            NDCW5_wy, NDCW2_5_wy, NDCW1_wy [Grid, Year] - number of coldwave events
            TDCW5_wy, TDCW2_5_wy, TDCW1_wy [Grid, Year] - average TX during coldwave days

    A coldwave event requires at least 3 consecutive days where t_max < TX{p}.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_max` with a `valid_time` dimension.
    ds_TX_wy : xr.Dataset
        Dataset containing TX1, TX2_5, TX5 with dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DDCW_*_wy, LDCW_*_wy, NDCW_*_wy, TDCW_*_wy variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS  = [5, 2.5, 1]
    THRESH_STRS = [str(p).replace('.', '_') for p in THRESHOLDS]
    all_vars = [f'{base}_{p_str}_wy' for p_str in THRESH_STRS for base in ['DDCW', 'LDCW', 'NDCW', 'TDCW']]
    logger.info("Calculating DDCW_*_wy, LDCW_*_wy, NDCW_*_wy, TDCW_*_wy (day coldwave indicators with full-year percentile thresholds at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in [f'TX{p_str}' for p_str in THRESH_STRS]:
        if var not in ds_TX_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_TX_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_max']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tx_wy = ds_TX_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tx_block):

        # Extract tmax for this block
        tx = ds_block['t_max'].values  # shape (365, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue

                for p_str in THRESH_STRS:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    threshold = tx_block[f'TX{p_str}'].values[i, j]
                    # Creates binary variable: 1 where t_max falls below the full-year percentile threshold, 0 otherwise
                    cold = (tx_1d < threshold).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each coldwave
                    lengths = ends - starts
                    # A coldwave requires at least 3 consecutive cold days, so mask
                    cw_mask = lengths >= 3
                    # Assign 0 values if no coldwave is detected
                    if not cw_mask.any():
                        out[f'DDCW_{p_str}_wy'][0, i, j] = 0
                        out[f'LDCW_{p_str}_wy'][0, i, j] = 0
                        out[f'NDCW_{p_str}_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid coldwaves only
                    cw_starts  = starts[cw_mask]
                    cw_lengths = lengths[cw_mask]
                    # Calculate the variables
                    out[f'DDCW_{p_str}_wy'][0, i, j] = cw_lengths.sum()   # total days across all coldwave events
                    out[f'LDCW_{p_str}_wy'][0, i, j] = cw_lengths.max()   # duration of the longest single event
                    out[f'NDCW_{p_str}_wy'][0, i, j] = cw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_max values from all coldwave periods to compute mean
                    cw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                    # Calculates mean
                    out[f'TDCW_{p_str}_wy'][0, i, j] = cw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str in THRESH_STRS:
        attrs = {
            f'DDCW_{p_str}_wy': (f'Number of days in day coldwave events (TX < TX{p_str} full-year percentile, >= 3 consecutive days)', 'days'),
            f'LDCW_{p_str}_wy': (f'Number of days in the longest day coldwave event (TX < TX{p_str} full-year percentile)', 'days'),
            f'NDCW_{p_str}_wy': (f'Number of day coldwave events (TX < TX{p_str} full-year percentile)', 'count'),
            f'TDCW_{p_str}_wy': (f'Average TX during day coldwave days (TX < TX{p_str} full-year percentile)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_night_coldwaves_wy(ds: xr.Dataset, ds_TN_wy: xr.Dataset):
    """
    Input: t_min   [Grid, Day] - daily minimum temperature at the grid level
           TN{p}   [Grid]      - p-th full-year percentile of t_min
    Output: DNCW5_wy, DNCW2_5_wy, DNCW1_wy [Grid, Year] - nights in coldwave events
            LNCW5_wy, LNCW2_5_wy, LNCW1_wy [Grid, Year] - nights in longest coldwave
            NNCW5_wy, NNCW2_5_wy, NNCW1_wy [Grid, Year] - number of coldwave events
            TNCW5_wy, TNCW2_5_wy, TNCW1_wy [Grid, Year] - average TN during coldwave nights

    A coldwave event requires at least 3 consecutive nights where t_min < TN{p}.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` with a `valid_time` dimension.
    ds_TN_wy : xr.Dataset
        Dataset containing TN1, TN2_5, TN5 with dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DNCW_*_wy, LNCW_*_wy, NNCW_*_wy, TNCW_*_wy variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS  = [5, 2.5, 1]
    THRESH_STRS = [str(p).replace('.', '_') for p in THRESHOLDS]
    all_vars = [f'{base}_{p_str}_wy' for p_str in THRESH_STRS for base in ['DNCW', 'LNCW', 'NNCW', 'TNCW']]
    logger.info("Calculating DNCW_*_wy, LNCW_*_wy, NNCW_*_wy, TNCW_*_wy (night coldwave indicators with full-year percentile thresholds at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in [f'TN{p_str}' for p_str in THRESH_STRS]:
        if var not in ds_TN_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_TN_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_min']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tn_wy = ds_TN_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tn_block):

        # Extract tmin for this block
        tn = ds_block['t_min'].values  # shape (365, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue

                for p_str in THRESH_STRS:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    threshold = tn_block[f'TN{p_str}'].values[i, j]
                    # Creates binary variable: 1 where t_min falls below the full-year percentile threshold, 0 otherwise
                    cold = (tn_1d < threshold).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each coldwave
                    lengths = ends - starts
                    # A coldwave requires at least 3 consecutive cold nights, so mask
                    cw_mask = lengths >= 3
                    # Assign 0 values if no coldwave is detected
                    if not cw_mask.any():
                        out[f'DNCW_{p_str}_wy'][0, i, j] = 0
                        out[f'LNCW_{p_str}_wy'][0, i, j] = 0
                        out[f'NNCW_{p_str}_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid coldwaves only
                    cw_starts  = starts[cw_mask]
                    cw_lengths = lengths[cw_mask]
                    # Calculate the variables
                    out[f'DNCW_{p_str}_wy'][0, i, j] = cw_lengths.sum()   # total nights across all coldwave events
                    out[f'LNCW_{p_str}_wy'][0, i, j] = cw_lengths.max()   # duration of the longest single event
                    out[f'NNCW_{p_str}_wy'][0, i, j] = cw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_min values from all coldwave periods to compute mean
                    cw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                    # Calculates mean
                    out[f'TNCW_{p_str}_wy'][0, i, j] = cw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str in THRESH_STRS:
        attrs = {
            f'DNCW_{p_str}_wy': (f'Number of nights in night coldwave events (TN < TN{p_str} full-year percentile, >= 3 consecutive nights)', 'nights'),
            f'LNCW_{p_str}_wy': (f'Number of nights in the longest night coldwave event (TN < TN{p_str} full-year percentile)', 'nights'),
            f'NNCW_{p_str}_wy': (f'Number of night coldwave events (TN < TN{p_str} full-year percentile)', 'count'),
            f'TNCW_{p_str}_wy': (f'Average TN during night coldwave nights (TN < TN{p_str} full-year percentile)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_WSD_wy(ds: xr.Dataset, ds_TX_wy: xr.Dataset):
    """
    Input: t_max   [Grid, Day] - daily maximum temperature at the grid level
           TX{p}   [Grid]      - p-th full-year percentile of t_max
    Output: WSD90_wy, WSD95_wy, WSD97_5_wy [Grid, Year] - days in warm spell events

    A warm spell requires at least 6 consecutive days where t_max > TX{p}.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_max` with a `valid_time` dimension.
    ds_TX_wy : xr.Dataset
        Dataset containing TX90, TX95, TX97_5 with dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with WSD90_wy, WSD95_wy, WSD97_5_wy variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS  = [90, 95, 97.5]
    THRESH_STRS = [str(p).replace('.', '_') for p in THRESHOLDS]
    all_vars = [f'WSD_{p_str}_wy' for p_str in THRESH_STRS]
    logger.info("Calculating WSD_*_wy (warm spell duration with full-year percentile thresholds at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in [f'TX{p_str}' for p_str in THRESH_STRS]:
        if var not in ds_TX_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_TX_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_max']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tx_wy = ds_TX_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tx_block):

        # Extract tmax for this block
        tx = ds_block['t_max'].values  # shape (365, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue

                for p_str in THRESH_STRS:
                    name = f'WSD_{p_str}_wy'
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    threshold = tx_block[f'TX{p_str}'].values[i, j]
                    # Creates binary variable: 1 where t_max exceeds the full-year percentile threshold, 0 otherwise
                    warm = (tx_1d > threshold).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], warm, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a warm spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each spell
                    lengths = ends - starts
                    # A warm spell requires at least 6 consecutive warm days, so mask
                    spell_mask = lengths >= 6
                    # Sum the total days across all valid warm spells (0 if none)
                    out[name][0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {f'WSD_{p_str}_wy': (f'Number of days in warm spell events (TX > TX{p_str} full-year percentile, >= 6 consecutive days)', 'days')
             for p_str in THRESH_STRS}
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_CSD_wy(ds: xr.Dataset, ds_TN_wy: xr.Dataset):
    """
    Input: t_min   [Grid, Day] - daily minimum temperature at the grid level
           TN{p}   [Grid]      - p-th full-year percentile of t_min
    Output: CSD10_wy, CSD5_wy, CSD2_5_wy [Grid, Year] - days in cold spell events

    A cold spell requires at least 6 consecutive days where t_min < TN{p}.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` with a `valid_time` dimension.
    ds_TN_wy : xr.Dataset
        Dataset containing TN2_5, TN5, TN10 with dimensions (latitude, longitude).

    Returns
    -------
    xr.Dataset
        Lazy dataset with CSD10_wy, CSD5_wy, CSD2_5_wy variables
        and dimensions (year, latitude, longitude).
    """
    THRESHOLDS  = [10, 5, 2.5]
    THRESH_STRS = [str(p).replace('.', '_') for p in THRESHOLDS]
    all_vars = [f'CSD_{p_str}_wy' for p_str in THRESH_STRS]
    logger.info("Calculating CSD_*_wy (cold spell duration with full-year percentile thresholds at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in [f'TN{p_str}' for p_str in THRESH_STRS]:
        if var not in ds_TN_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_TN_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_min']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tn_wy = ds_TN_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tn_block):

        # Extract tmin for this block
        tn = ds_block['t_min'].values  # shape (365, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue

                for p_str in THRESH_STRS:
                    name = f'CSD_{p_str}_wy'
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    threshold = tn_block[f'TN{p_str}'].values[i, j]
                    # Creates binary variable: 1 where t_min falls below the full-year percentile threshold, 0 otherwise
                    cold = (tn_1d < threshold).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends = np.where(d == -1)[0]
                    # Get the length of each spell
                    lengths = ends - starts
                    # A cold spell requires at least 6 consecutive cold days, so mask
                    spell_mask = lengths >= 6
                    # Sum the total days across all valid cold spells (0 if none)
                    out[name][0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude': ds_block['latitude'].values,
                  'longitude': ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    attrs = {f'CSD_{p_str}_wy': (f'Number of days in cold spell events (TN < TN{p_str} full-year percentile, >= 6 consecutive days)', 'days')
             for p_str in THRESH_STRS}
    for var, (long_name, unit) in attrs.items():
        result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


# ===========================================================================
# COMBINED FULL-YEAR PERCENTILE + ABSOLUTE THRESHOLD VARIABLES
# ===========================================================================


def calculate_coldwarm_abs_wy(ds: xr.Dataset, ds_T_wy: xr.Dataset):
    """
    Input: t_min    [Grid, Day] - daily minimum temperature at the grid level
           t_max    [Grid, Day] - daily maximum temperature at the grid level
           TN2_5    [Grid]      - 2.5th full-year percentile of t_min
           TN5      [Grid]      - 5th full-year percentile of t_min
           TN95     [Grid]      - 95th full-year percentile of t_min
           TN97_5   [Grid]      - 97.5th full-year percentile of t_min
           TX2_5    [Grid]      - 2.5th full-year percentile of t_max
           TX5      [Grid]      - 5th full-year percentile of t_max
           TX95     [Grid]      - 95th full-year percentile of t_max
           TX97_5   [Grid]      - 97.5th full-year percentile of t_max
    Output: CN5_0C_wy     [Grid, Year] - cold nights: t_min < TN5   and t_min < 0°C
            CN2_5_0C_wy   [Grid, Year] - cold nights: t_min < TN2_5 and t_min < 0°C
            CD5_10C_wy    [Grid, Year] - cold days:   t_max < TX5   and t_max < 10°C
            CD2_5_10C_wy  [Grid, Year] - cold days:   t_max < TX2_5 and t_max < 10°C
            WN95_20C_wy   [Grid, Year] - warm nights: t_min > TN95  and t_min > 20°C
            WN97_5_20C_wy [Grid, Year] - warm nights: t_min > TN97_5 and t_min > 20°C
            WD95_30C_wy   [Grid, Year] - warm days:   t_max > TX95  and t_max > 30°C
            WD97_5_30C_wy [Grid, Year] - warm days:   t_max > TX97_5 and t_max > 30°C

    Each variable requires both a full-year unconditional percentile condition and a fixed
    absolute threshold to be met simultaneously. The 2D percentile threshold broadcasts
    against the 3D daily data automatically — no Feb 29 removal or DOY normalisation needed.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` and `t_max` with
        a `valid_time` dimension (daily frequency).
    ds_T_wy : xr.Dataset
        Dataset containing TN2_5, TN5, TN95, TN97_5, TX2_5, TX5, TX95, TX97_5
        with dimensions (latitude, longitude). Typically temp_perc_wy (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with CN*_wy, CD*_wy, WN*_wy, WD*_wy variables
        and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating CN*_wy, CD*_wy, WN*_wy, WD*_wy (full-year percentile + absolute threshold cold/warm night and day counts)")

    for var in ['t_min', 't_max']:
        if var not in ds:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds.data_vars)}")
    for var in ['TN2_5', 'TN5', 'TN95', 'TN97_5', 'TX2_5', 'TX5', 'TX95', 'TX97_5']:
        if var not in ds_T_wy:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_T_wy.data_vars)}")

    # Get an array of years present in the data
    years = np.unique(ds['valid_time'].dt.year.values)

    result = {}

    # The full-year threshold has dims (latitude, longitude) only; xarray broadcasts it against
    # the (valid_time, latitude, longitude) daily data automatically — no DOY lookup needed.
    result['CN5_0C_wy'] = (
        ((ds['t_min'] < ds_T_wy['TN5']) & (ds['t_min'] < 0.0)).where(ds['t_min'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of cold nights (t_min < TN5 and t_min < 0°C)', units='days')
        .rename('CN5_0C_wy')
    )

    result['CN2_5_0C_wy'] = (
        ((ds['t_min'] < ds_T_wy['TN2_5']) & (ds['t_min'] < 0.0)).where(ds['t_min'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of cold nights (t_min < TN2_5 and t_min < 0°C)', units='days')
        .rename('CN2_5_0C_wy')
    )

    result['CD5_10C_wy'] = (
        ((ds['t_max'] < ds_T_wy['TX5']) & (ds['t_max'] < 10.0)).where(ds['t_max'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of cold days (t_max < TX5 and t_max < 10°C)', units='days')
        .rename('CD5_10C_wy')
    )

    result['CD2_5_10C_wy'] = (
        ((ds['t_max'] < ds_T_wy['TX2_5']) & (ds['t_max'] < 10.0)).where(ds['t_max'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of cold days (t_max < TX2_5 and t_max < 10°C)', units='days')
        .rename('CD2_5_10C_wy')
    )

    result['WN95_20C_wy'] = (
        ((ds['t_min'] > ds_T_wy['TN95']) & (ds['t_min'] > 20.0)).where(ds['t_min'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of warm nights (t_min > TN95 and t_min > 20°C)', units='days')
        .rename('WN95_20C_wy')
    )

    result['WN97_5_20C_wy'] = (
        ((ds['t_min'] > ds_T_wy['TN97_5']) & (ds['t_min'] > 20.0)).where(ds['t_min'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of warm nights (t_min > TN97_5 and t_min > 20°C)', units='days')
        .rename('WN97_5_20C_wy')
    )

    result['WD95_30C_wy'] = (
        ((ds['t_max'] > ds_T_wy['TX95']) & (ds['t_max'] > 30.0)).where(ds['t_max'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of warm days (t_max > TX95 and t_max > 30°C)', units='days')
        .rename('WD95_30C_wy')
    )

    result['WD97_5_30C_wy'] = (
        ((ds['t_max'] > ds_T_wy['TX97_5']) & (ds['t_max'] > 30.0)).where(ds['t_max'].notnull())
        .resample(valid_time='YE').sum(min_count=1)
        .rename({'valid_time': 'year'}).assign_coords(year=years)
        .drop_attrs()
        .assign_attrs(long_name='Number of warm days (t_max > TX97_5 and t_max > 30°C)', units='days')
        .rename('WD97_5_30C_wy')
    )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset(result)


def calculate_day_heatwaves_abs_wy(ds: xr.Dataset, ds_TX_wy: xr.Dataset):
    """
    Input: t_max   [Grid, Day] - daily maximum temperature at the grid level
           TX97_5  [Grid]      - 97.5th full-year percentile of t_max
           TX99    [Grid]      - 99th full-year percentile of t_max
    Output: DDHW97_5_30C_wy, LDHW97_5_30C_wy, NDHW97_5_30C_wy, TDHW97_5_30C_wy [Grid, Year]
            DDHW99_30C_wy,   LDHW99_30C_wy,   NDHW99_30C_wy,   TDHW99_30C_wy   [Grid, Year]

    A heatwave event requires at least 3 consecutive days where t_max > TX{p} AND t_max > 30°C.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_max` with a `valid_time` dimension.
    ds_TX_wy : xr.Dataset
        Dataset containing TX97_5 and TX99 with dimensions (latitude, longitude).
        Typically temp_perc_wy[['TX97_5', 'TX99']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DDHW/LDHW/NDHW/TDHW_97_5_30C_wy and _99_30C_wy variables
        and dimensions (year, latitude, longitude).
    """
    PERCENTILES   = [('97_5', 'TX97_5'), ('99', 'TX99')]
    ABS_THRESHOLD = 30.0
    all_vars = [f'{base}{p}_30C_wy' for p, _ in PERCENTILES for base in ['DDHW', 'LDHW', 'NDHW', 'TDHW']]
    logger.info("Calculating DDHW/LDHW/NDHW/TDHW_97_5_30C_wy and _99_30C_wy (day heatwave indicators with full-year percentile + t_max > 30°C at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    for _, pct_var in PERCENTILES:
        if pct_var not in ds_TX_wy:
            raise KeyError(f"'{pct_var}' not found in dataset. Available variables: {list(ds_TX_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_max']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tx_wy = ds_TX_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tx_wy_block):

        # Extract tmax for this block
        tx = ds_block['t_max'].values  # shape (365/366, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue

                for p_str, pct_var in PERCENTILES:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    pct_val = tx_wy_block[pct_var].values[i, j]
                    # Creates binary variable: 1 where t_max exceeds both the full-year percentile and the absolute threshold
                    hot = ((tx_1d > pct_val) & (tx_1d > ABS_THRESHOLD)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], hot, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a hot spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts  = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends    = np.where(d == -1)[0]
                    # Get the length of each heatwave
                    lengths = ends - starts
                    # A heatwave requires at least 3 consecutive hot days, so mask
                    hw_mask = lengths >= 3
                    # Assign 0 values if no heatwave is detected
                    if not hw_mask.any():
                        out[f'DDHW{p_str}_30C_wy'][0, i, j] = 0
                        out[f'LDHW{p_str}_30C_wy'][0, i, j] = 0
                        out[f'NDHW{p_str}_30C_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid heatwaves only
                    hw_starts  = starts[hw_mask]
                    hw_lengths = lengths[hw_mask]
                    # Calculate the variables
                    out[f'DDHW{p_str}_30C_wy'][0, i, j] = hw_lengths.sum()   # total days across all heatwave events
                    out[f'LDHW{p_str}_30C_wy'][0, i, j] = hw_lengths.max()   # duration of the longest single event
                    out[f'NDHW{p_str}_30C_wy'][0, i, j] = hw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_max values from all heatwave periods to compute mean
                    hw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                    # Calculates mean
                    out[f'TDHW{p_str}_30C_wy'][0, i, j] = hw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude':   ds_block['latitude'].values,
                  'longitude':  ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str, _ in PERCENTILES:
        attrs = {
            f'DDHW{p_str}_30C_wy': (f'Number of days in day heatwave events (TX > TX{p_str} and TX > 30°C, >= 3 consecutive days)', 'days'),
            f'LDHW{p_str}_30C_wy': (f'Number of days in the longest day heatwave event (TX > TX{p_str} and TX > 30°C)', 'days'),
            f'NDHW{p_str}_30C_wy': (f'Number of day heatwave events (TX > TX{p_str} and TX > 30°C)', 'count'),
            f'TDHW{p_str}_30C_wy': (f'Average TX during day heatwave days (TX > TX{p_str} and TX > 30°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_night_heatwaves_abs_wy(ds: xr.Dataset, ds_TN_wy: xr.Dataset):
    """
    Input: t_min   [Grid, Day] - daily minimum temperature at the grid level
           TN97_5  [Grid]      - 97.5th full-year percentile of t_min
           TN99    [Grid]      - 99th full-year percentile of t_min
    Output: DNHW97_5_20C_wy, LNHW97_5_20C_wy, NNHW97_5_20C_wy, TNHW97_5_20C_wy [Grid, Year]
            DNHW99_20C_wy,   LNHW99_20C_wy,   NNHW99_20C_wy,   TNHW99_20C_wy   [Grid, Year]

    A heatwave event requires at least 3 consecutive nights where t_min > TN{p} AND t_min > 20°C.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` with a `valid_time` dimension.
    ds_TN_wy : xr.Dataset
        Dataset containing TN97_5 and TN99 with dimensions (latitude, longitude).
        Typically temp_perc_wy[['TN97_5', 'TN99']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DNHW/LNHW/NNHW/TNHW_97_5_20C_wy and _99_20C_wy variables
        and dimensions (year, latitude, longitude).
    """
    PERCENTILES   = [('97_5', 'TN97_5'), ('99', 'TN99')]
    ABS_THRESHOLD = 20.0
    all_vars = [f'{base}{p}_20C_wy' for p, _ in PERCENTILES for base in ['DNHW', 'LNHW', 'NNHW', 'TNHW']]
    logger.info("Calculating DNHW/LNHW/NNHW/TNHW_97_5_20C_wy and _99_20C_wy (night heatwave indicators with full-year percentile + t_min > 20°C at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    for _, pct_var in PERCENTILES:
        if pct_var not in ds_TN_wy:
            raise KeyError(f"'{pct_var}' not found in dataset. Available variables: {list(ds_TN_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_min']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tn_wy = ds_TN_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tn_wy_block):

        # Extract tmin for this block
        tn = ds_block['t_min'].values  # shape (365/366, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue

                for p_str, pct_var in PERCENTILES:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    pct_val = tn_wy_block[pct_var].values[i, j]
                    # Creates binary variable: 1 where t_min exceeds both the full-year percentile and the absolute threshold
                    hot = ((tn_1d > pct_val) & (tn_1d > ABS_THRESHOLD)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], hot, [0]])
                    # Take the diff between consecutive nights, +1 marks the start of a hot spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts  = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends    = np.where(d == -1)[0]
                    # Get the length of each heatwave
                    lengths = ends - starts
                    # A heatwave requires at least 3 consecutive hot nights, so mask
                    hw_mask = lengths >= 3
                    # Assign 0 values if no heatwave is detected
                    if not hw_mask.any():
                        out[f'DNHW{p_str}_20C_wy'][0, i, j] = 0
                        out[f'LNHW{p_str}_20C_wy'][0, i, j] = 0
                        out[f'NNHW{p_str}_20C_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid heatwaves only
                    hw_starts  = starts[hw_mask]
                    hw_lengths = lengths[hw_mask]
                    # Calculate the variables
                    out[f'DNHW{p_str}_20C_wy'][0, i, j] = hw_lengths.sum()   # total nights across all heatwave events
                    out[f'LNHW{p_str}_20C_wy'][0, i, j] = hw_lengths.max()   # duration of the longest single event
                    out[f'NNHW{p_str}_20C_wy'][0, i, j] = hw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_min values from all heatwave periods to compute mean
                    hw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(hw_starts, hw_lengths)])
                    # Calculates mean
                    out[f'TNHW{p_str}_20C_wy'][0, i, j] = hw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude':   ds_block['latitude'].values,
                  'longitude':  ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str, _ in PERCENTILES:
        attrs = {
            f'DNHW{p_str}_20C_wy': (f'Number of nights in night heatwave events (TN > TN{p_str} and TN > 20°C, >= 3 consecutive nights)', 'nights'),
            f'LNHW{p_str}_20C_wy': (f'Number of nights in the longest night heatwave event (TN > TN{p_str} and TN > 20°C)', 'nights'),
            f'NNHW{p_str}_20C_wy': (f'Number of night heatwave events (TN > TN{p_str} and TN > 20°C)', 'count'),
            f'TNHW{p_str}_20C_wy': (f'Average TN during night heatwave nights (TN > TN{p_str} and TN > 20°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_day_coldwaves_abs_wy(ds: xr.Dataset, ds_TX_wy: xr.Dataset):
    """
    Input: t_max  [Grid, Day] - daily maximum temperature at the grid level
           TX2_5  [Grid]      - 2.5th full-year percentile of t_max
           TX1    [Grid]      - 1st full-year percentile of t_max
    Output: DDCW2_5_10C_wy, LDCW2_5_10C_wy, NDCW2_5_10C_wy, TDCW2_5_10C_wy [Grid, Year]
            DDCW1_10C_wy,   LDCW1_10C_wy,   NDCW1_10C_wy,   TDCW1_10C_wy   [Grid, Year]

    A coldwave event requires at least 3 consecutive days where t_max < TX{p} AND t_max < 10°C.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_max` with a `valid_time` dimension.
    ds_TX_wy : xr.Dataset
        Dataset containing TX2_5 and TX1 with dimensions (latitude, longitude).
        Typically temp_perc_wy[['TX2_5', 'TX1']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DDCW/LDCW/NDCW/TDCW_2_5_10C_wy and _1_10C_wy variables
        and dimensions (year, latitude, longitude).
    """
    PERCENTILES   = [('2_5', 'TX2_5'), ('1', 'TX1')]
    ABS_THRESHOLD = 10.0
    all_vars = [f'{base}{p}_10C_wy' for p, _ in PERCENTILES for base in ['DDCW', 'LDCW', 'NDCW', 'TDCW']]
    logger.info("Calculating DDCW/LDCW/NDCW/TDCW_2_5_10C_wy and _1_10C_wy (day coldwave indicators with full-year percentile + t_max < 10°C at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    for _, pct_var in PERCENTILES:
        if pct_var not in ds_TX_wy:
            raise KeyError(f"'{pct_var}' not found in dataset. Available variables: {list(ds_TX_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_max']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tx_wy = ds_TX_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tx_wy_block):

        # Extract tmax for this block
        tx = ds_block['t_max'].values  # shape (365/366, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue

                for p_str, pct_var in PERCENTILES:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    pct_val = tx_wy_block[pct_var].values[i, j]
                    # Creates binary variable: 1 where t_max falls below both the full-year percentile and the absolute threshold
                    cold = ((tx_1d < pct_val) & (tx_1d < ABS_THRESHOLD)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts  = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends    = np.where(d == -1)[0]
                    # Get the length of each coldwave
                    lengths = ends - starts
                    # A coldwave requires at least 3 consecutive cold days, so mask
                    cw_mask = lengths >= 3
                    # Assign 0 values if no coldwave is detected
                    if not cw_mask.any():
                        out[f'DDCW{p_str}_10C_wy'][0, i, j] = 0
                        out[f'LDCW{p_str}_10C_wy'][0, i, j] = 0
                        out[f'NDCW{p_str}_10C_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid coldwaves only
                    cw_starts  = starts[cw_mask]
                    cw_lengths = lengths[cw_mask]
                    # Calculate the variables
                    out[f'DDCW{p_str}_10C_wy'][0, i, j] = cw_lengths.sum()   # total days across all coldwave events
                    out[f'LDCW{p_str}_10C_wy'][0, i, j] = cw_lengths.max()   # duration of the longest single event
                    out[f'NDCW{p_str}_10C_wy'][0, i, j] = cw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_max values from all coldwave periods to compute mean
                    cw_tx = np.concatenate([tx_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                    # Calculates mean
                    out[f'TDCW{p_str}_10C_wy'][0, i, j] = cw_tx.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude':   ds_block['latitude'].values,
                  'longitude':  ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str, _ in PERCENTILES:
        attrs = {
            f'DDCW{p_str}_10C_wy': (f'Number of days in day coldwave events (TX < TX{p_str} and TX < 10°C, >= 3 consecutive days)', 'days'),
            f'LDCW{p_str}_10C_wy': (f'Number of days in the longest day coldwave event (TX < TX{p_str} and TX < 10°C)', 'days'),
            f'NDCW{p_str}_10C_wy': (f'Number of day coldwave events (TX < TX{p_str} and TX < 10°C)', 'count'),
            f'TDCW{p_str}_10C_wy': (f'Average TX during day coldwave days (TX < TX{p_str} and TX < 10°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_night_coldwaves_abs_wy(ds: xr.Dataset, ds_TN_wy: xr.Dataset):
    """
    Input: t_min  [Grid, Day] - daily minimum temperature at the grid level
           TN2_5  [Grid]      - 2.5th full-year percentile of t_min
           TN1    [Grid]      - 1st full-year percentile of t_min
    Output: DNCW2_5_0C_wy, LNCW2_5_0C_wy, NNCW2_5_0C_wy, TNCW2_5_0C_wy [Grid, Year]
            DNCW1_0C_wy,   LNCW1_0C_wy,   NNCW1_0C_wy,   TNCW1_0C_wy   [Grid, Year]

    A coldwave event requires at least 3 consecutive nights where t_min < TN{p} AND t_min < 0°C.
    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` with a `valid_time` dimension.
    ds_TN_wy : xr.Dataset
        Dataset containing TN2_5 and TN1 with dimensions (latitude, longitude).
        Typically temp_perc_wy[['TN2_5', 'TN1']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with DNCW/LNCW/NNCW/TNCW_2_5_0C_wy and _1_0C_wy variables
        and dimensions (year, latitude, longitude).
    """
    PERCENTILES   = [('2_5', 'TN2_5'), ('1', 'TN1')]
    ABS_THRESHOLD = 0.0
    all_vars = [f'{base}{p}_0C_wy' for p, _ in PERCENTILES for base in ['DNCW', 'LNCW', 'NNCW', 'TNCW']]
    logger.info("Calculating DNCW/LNCW/NNCW/TNCW_2_5_0C_wy and _1_0C_wy (night coldwave indicators with full-year percentile + t_min < 0°C at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    for _, pct_var in PERCENTILES:
        if pct_var not in ds_TN_wy:
            raise KeyError(f"'{pct_var}' not found in dataset. Available variables: {list(ds_TN_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_min']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tn_wy = ds_TN_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tn_wy_block):

        # Extract tmin for this block
        tn = ds_block['t_min'].values  # shape (365/366, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all threshold/variable combinations, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue

                for p_str, pct_var in PERCENTILES:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    pct_val = tn_wy_block[pct_var].values[i, j]
                    # Creates binary variable: 1 where t_min falls below both the full-year percentile and the absolute threshold
                    cold = ((tn_1d < pct_val) & (tn_1d < ABS_THRESHOLD)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive nights, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts  = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends    = np.where(d == -1)[0]
                    # Get the length of each coldwave
                    lengths = ends - starts
                    # A coldwave requires at least 3 consecutive cold nights, so mask
                    cw_mask = lengths >= 3
                    # Assign 0 values if no coldwave is detected
                    if not cw_mask.any():
                        out[f'DNCW{p_str}_0C_wy'][0, i, j] = 0
                        out[f'LNCW{p_str}_0C_wy'][0, i, j] = 0
                        out[f'NNCW{p_str}_0C_wy'][0, i, j] = 0
                        continue
                    # Use the mask to get the start and the length of the valid coldwaves only
                    cw_starts  = starts[cw_mask]
                    cw_lengths = lengths[cw_mask]
                    # Calculate the variables
                    out[f'DNCW{p_str}_0C_wy'][0, i, j] = cw_lengths.sum()   # total nights across all coldwave events
                    out[f'LNCW{p_str}_0C_wy'][0, i, j] = cw_lengths.max()   # duration of the longest single event
                    out[f'NNCW{p_str}_0C_wy'][0, i, j] = cw_mask.sum()      # count of distinct events
                    # Concatenate the actual t_min values from all coldwave periods to compute mean
                    cw_tn = np.concatenate([tn_1d[s:s + l] for s, l in zip(cw_starts, cw_lengths)])
                    # Calculates mean
                    out[f'TNCW{p_str}_0C_wy'][0, i, j] = cw_tn.mean()

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude':   ds_block['latitude'].values,
                  'longitude':  ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str, _ in PERCENTILES:
        attrs = {
            f'DNCW{p_str}_0C_wy': (f'Number of nights in night coldwave events (TN < TN{p_str} and TN < 0°C, >= 3 consecutive nights)', 'nights'),
            f'LNCW{p_str}_0C_wy': (f'Number of nights in the longest night coldwave event (TN < TN{p_str} and TN < 0°C)', 'nights'),
            f'NNCW{p_str}_0C_wy': (f'Number of night coldwave events (TN < TN{p_str} and TN < 0°C)', 'count'),
            f'TNCW{p_str}_0C_wy': (f'Average TN during night coldwave nights (TN < TN{p_str} and TN < 0°C)', 'Celsius'),
        }
        for var, (long_name, unit) in attrs.items():
            result[var].attrs.update(long_name=long_name, units=unit)

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_WSD_abs_wy(ds: xr.Dataset, ds_TX_wy: xr.Dataset):
    """
    Input: t_max  [Grid, Day] - daily maximum temperature at the grid level
           TX95   [Grid]      - 95th full-year percentile of t_max
           TX90   [Grid]      - 90th full-year percentile of t_max
    Output: WSD95_20C_wy [Grid, Year] - days in warm spell events (TX > TX95 and TX > 20°C, >= 6 consecutive)
            WSD90_20C_wy [Grid, Year] - days in warm spell events (TX > TX90 and TX > 20°C, >= 6 consecutive)

    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_max` with a `valid_time` dimension.
    ds_TX_wy : xr.Dataset
        Dataset containing TX95 and TX90 with dimensions (latitude, longitude).
        Typically temp_perc_wy[['TX95', 'TX90']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with WSD95_20C_wy and WSD90_20C_wy variables
        and dimensions (year, latitude, longitude).
    """
    PERCENTILES   = [('95', 'TX95'), ('90', 'TX90')]
    ABS_THRESHOLD = 20.0
    all_vars = [f'WSD{p}_20C_wy' for p, _ in PERCENTILES]
    logger.info("Calculating WSD95_20C_wy, WSD90_20C_wy (warm spell duration with full-year percentile + t_max > 20°C at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    for _, pct_var in PERCENTILES:
        if pct_var not in ds_TX_wy:
            raise KeyError(f"'{pct_var}' not found in dataset. Available variables: {list(ds_TX_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_max']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tx_wy = ds_TX_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tx_wy_block):

        # Extract tmax for this block
        tx = ds_block['t_max'].values  # shape (365/366, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tx.shape[1], tx.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmax data throughout the year for that location
                tx_1d = tx[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tx_1d)):
                    continue

                for p_str, pct_var in PERCENTILES:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    pct_val = tx_wy_block[pct_var].values[i, j]
                    # Creates binary variable: 1 where t_max exceeds both the full-year percentile and the absolute threshold
                    warm = ((tx_1d > pct_val) & (tx_1d > ABS_THRESHOLD)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], warm, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a warm spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts  = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends    = np.where(d == -1)[0]
                    # Get the length of each spell
                    lengths = ends - starts
                    # A warm spell requires at least 6 consecutive warm days, so mask
                    spell_mask = lengths >= 6
                    # Sum the total days across all valid warm spells (0 if none)
                    out[f'WSD{p_str}_20C_wy'][0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude':   ds_block['latitude'].values,
                  'longitude':  ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tx_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str, _ in PERCENTILES:
        result[f'WSD{p_str}_20C_wy'].attrs.update(
            long_name=f'Number of days in warm spell events (TX > TX{p_str} and TX > 20°C, >= 6 consecutive days)',
            units='days'
        )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


def calculate_CSD_abs_wy(ds: xr.Dataset, ds_TN_wy: xr.Dataset):
    """
    Input: t_min  [Grid, Day] - daily minimum temperature at the grid level
           TN5    [Grid]      - 5th full-year percentile of t_min
           TN10   [Grid]      - 10th full-year percentile of t_min
    Output: CSD5_10C_wy  [Grid, Year] - days in cold spell events (TN < TN5  and TN < 10°C, >= 6 consecutive)
            CSD10_10C_wy [Grid, Year] - days in cold spell events (TN < TN10 and TN < 10°C, >= 6 consecutive)

    Streak detection is applied per grid cell and per year via map_blocks.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `t_min` with a `valid_time` dimension.
    ds_TN_wy : xr.Dataset
        Dataset containing TN5 and TN10 with dimensions (latitude, longitude).
        Typically temp_perc_wy[['TN5', 'TN10']] (computed).

    Returns
    -------
    xr.Dataset
        Lazy dataset with CSD5_10C_wy and CSD10_10C_wy variables
        and dimensions (year, latitude, longitude).
    """
    PERCENTILES   = [('5', 'TN5'), ('10', 'TN10')]
    ABS_THRESHOLD = 10.0
    all_vars = [f'CSD{p}_10C_wy' for p, _ in PERCENTILES]
    logger.info("Calculating CSD5_10C_wy, CSD10_10C_wy (cold spell duration with full-year percentile + t_min < 10°C at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")
    for _, pct_var in PERCENTILES:
        if pct_var not in ds_TN_wy:
            raise KeyError(f"'{pct_var}' not found in dataset. Available variables: {list(ds_TN_wy.data_vars)}")

    # Keep Feb 29: the full-year thresholds are not day-of-year specific, so leap days are valid data.
    # Rechunk with actual per-year day counts so each block is exactly one year.
    years_all = ds['valid_time'].dt.year.values
    # Get an array of years present in the data and the number of days in each year
    unique_years, year_lengths = np.unique(years_all, return_counts=True)
    # Get the number of years
    n_years = len(unique_years)
    ds = ds[['t_min']].chunk({'valid_time': year_lengths.tolist()})
    # Rechunk the threshold table to match the spatial chunking of ds, so map_blocks aligns blocks correctly
    tn_wy = ds_TN_wy.chunk({'latitude': ds.chunksizes['latitude'], 'longitude': ds.chunksizes['longitude']})

    def process_block(ds_block, tn_wy_block):

        # Extract tmin for this block
        tn = ds_block['t_min'].values  # shape (365/366, n_lat, n_lon)
        # Get the number of latitude and longitude coordinates
        n_lat, n_lon = tn.shape[1], tn.shape[2]

        # Output arrays for all thresholds, initialised to NaN
        out = {name: np.full((1, n_lat, n_lon), np.nan, dtype=np.float32) for name in all_vars}

        # Loop through each location
        for i in range(n_lat):
            for j in range(n_lon):
                # Get all the tmin data throughout the year for that location
                tn_1d = tn[:, i, j]
                # If everything is NaN (ocean cells) continue
                if np.all(np.isnan(tn_1d)):
                    continue

                for p_str, pct_var in PERCENTILES:
                    # Full-year threshold is a scalar per grid cell — no DOY indexing needed
                    pct_val = tn_wy_block[pct_var].values[i, j]
                    # Creates binary variable: 1 where t_min falls below both the full-year percentile and the absolute threshold
                    cold = ((tn_1d < pct_val) & (tn_1d < ABS_THRESHOLD)).astype(np.int8)
                    # Pad with zeros so diff detects runs that start on day 0 or end on the last day
                    padded = np.concatenate([[0], cold, [0]])
                    # Take the diff between consecutive days, +1 marks the start of a cold spell; -1 marks the end
                    d = np.diff(padded)
                    # Get all the start dates (value = +1)
                    starts  = np.where(d == 1)[0]
                    # Get all the end dates (value = -1)
                    ends    = np.where(d == -1)[0]
                    # Get the length of each spell
                    lengths = ends - starts
                    # A cold spell requires at least 6 consecutive cold days, so mask
                    spell_mask = lengths >= 6
                    # Sum the total days across all valid cold spells (0 if none)
                    out[f'CSD{p_str}_10C_wy'][0, i, j] = lengths[spell_mask].sum() if spell_mask.any() else 0

        # Wrap results back into a Dataset with the first timestamp of the block as the time coordinate
        coords = {'valid_time': ds_block['valid_time'].values[[0]],
                  'latitude':   ds_block['latitude'].values,
                  'longitude':  ds_block['longitude'].values}
        dims = ['valid_time', 'latitude', 'longitude']
        return xr.Dataset({name: xr.DataArray(out[name], dims=dims, coords=coords) for name in all_vars})

    # Get day-1 of the year coordinates (Jan 1 of each year, accounting for variable year lengths with Feb 29)
    year_first_dates = ds['valid_time'].values[np.r_[0, year_lengths[:-1].cumsum()]]
    # Gets spatial chunk sizes in main dataframe (to be given to the output df)
    lat_chunks = ds.chunksizes['latitude']
    lon_chunks = ds.chunksizes['longitude']
    # Creates the empty (so far) output dataset with the correct structure
    template = xr.Dataset({
        var: xr.DataArray(
            da.full(
                (n_years, len(ds['latitude']), len(ds['longitude'])),
                np.nan, chunks=(1, lat_chunks, lon_chunks), dtype=np.float32,
                name=f'template-{var}',
            ),
            dims=['valid_time', 'latitude', 'longitude'],
            coords={'valid_time': year_first_dates, 'latitude': ds['latitude'], 'longitude': ds['longitude']},
        )
        for var in all_vars
    })

    # Runs (lazily) the computations over each chunk
    result = xr.map_blocks(process_block, ds, args=[tn_wy], template=template)
    # Corrects and assigns the correct dimension
    result = result.rename({'valid_time': 'year'}).assign_coords(year=unique_years)

    # Assigns attributes to each variable
    for p_str, _ in PERCENTILES:
        result[f'CSD{p_str}_10C_wy'].attrs.update(
            long_name=f'Number of days in cold spell events (TN < TN{p_str} and TN < 10°C, >= 6 consecutive days)',
            units='days'
        )

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")
    return result


# Load pre-computed percentiles and intermediate daily/monthly outputs
temp_perc = xr.open_dataset(OUT_PERC  / "temp_percentiles.nc", chunks="auto")
temp_perc_wy = xr.open_dataset(OUT_PERC / "temp_percentiles_fullyear.nc", chunks="auto")
wet_days_perc = xr.open_dataset(OUT_PERC  / "precip_percentiles.nc")
daily_ds = xr.open_mfdataset(sorted(OUT_DAILY.glob("daily_vars_*.nc")))
daily_ds = daily_ds.sel(valid_time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
DTR_d = daily_ds[['DTR_d']]
PW_d = daily_ds[['PW_d']]
P_jm = xr.open_mfdataset(sorted(OUT_MONTHLY.glob("monthly_vars_*.nc")))[['P_jm']]
P_jm = P_jm.sel(valid_time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

# Calculating yearly variables
TM = calculate_TM(ds)
TX = calculate_TX(ds)
TN = calculate_TN(ds)
TNN_TXX = calculate_TNN_TXX(ds)
DTR = calculate_DTR(DTR_d)
TVAR = calculate_TVAR(ds, TM)
coldwarm = calculate_coldwarm_nightsdays(ds, temp_perc)
day_hw = calculate_day_heatwaves(ds, temp_perc[['TX_90p_15w']])
night_hw = calculate_night_heatwaves(ds, temp_perc[['TN_90p_15w']])
day_cw = calculate_day_coldwaves(ds, temp_perc[['TX_10p_15w']])
night_cw = calculate_night_coldwaves(ds, temp_perc[['TN_10p_15w']])
CSD = calculate_CSD(ds, temp_perc[['TN_10p_5w']])
WSD = calculate_WSD(ds, temp_perc[['TX_90p_5w']])
coldwarm_abs = calculate_coldwarm_nightsdays_abs(ds, temp_perc)
day_hw_abs = calculate_day_heatwaves_abs(ds, temp_perc[['TX_90p_15w']])
night_hw_abs = calculate_night_heatwaves_abs(ds, temp_perc[['TN_90p_15w']])
day_cw_abs = calculate_day_coldwaves_abs(ds, temp_perc[['TX_10p_15w']])
night_cw_abs = calculate_night_coldwaves_abs(ds, temp_perc[['TN_10p_15w']])
CSD_abs = calculate_CSD_abs(ds, temp_perc[['TN_10p_5w']])
WSD_abs = calculate_WSD_abs(ds, temp_perc[['TX_90p_5w']])
TNnumber_TXnumber = calculate_TNnumber_TXnumber(ds)
TM_bins = calculate_TM_bins(ds)
PA = calculate_PA(ds)
PWT = calculate_PWT(PW_d, ds)
W = calculate_W(PW_d, ds)
PWA = calculate_PWA(PWT, W)
PVAR = calculate_PVAR(ds, PA)
PWVAR = calculate_PWVAR(PW_d, PWA, W)
P95WT_P99WT = calculate_P95WT_P99WT(ds, wet_days_perc)
consec_counts = calculate_consecutive_precip_counts(ds, wet_days_perc)
consec_totals = calculate_consecutive_precip_totals(ds, wet_days_perc)
PX1_PX5 = calculate_PX1_PX5(ds)
PXM_PNM = calculate_PXM_PNM(P_jm)
P_bins = calculate_P_bins(ds)
coldwarm_wy = calculate_coldwarm_nightsdays_wy(ds, temp_perc_wy)
day_hw_wy = calculate_day_heatwaves_wy(ds, temp_perc_wy[['TX95', 'TX97_5', 'TX99']])
night_hw_wy = calculate_night_heatwaves_wy(ds, temp_perc_wy[['TN95', 'TN97_5', 'TN99']])
day_cw_wy = calculate_day_coldwaves_wy(ds, temp_perc_wy[['TX1', 'TX2_5', 'TX5']])
night_cw_wy = calculate_night_coldwaves_wy(ds, temp_perc_wy[['TN1', 'TN2_5', 'TN5']])
WSD_wy = calculate_WSD_wy(ds, temp_perc_wy[['TX90', 'TX95', 'TX97_5']])
CSD_wy = calculate_CSD_wy(ds, temp_perc_wy[['TN2_5', 'TN5', 'TN10']])
coldwarm_abs_wy = calculate_coldwarm_abs_wy(ds, temp_perc_wy)
day_hw_abs_wy   = calculate_day_heatwaves_abs_wy(ds, temp_perc_wy[['TX97_5', 'TX99']])
night_hw_abs_wy = calculate_night_heatwaves_abs_wy(ds, temp_perc_wy[['TN97_5', 'TN99']])
day_cw_abs_wy   = calculate_day_coldwaves_abs_wy(ds, temp_perc_wy[['TX2_5', 'TX1']])
night_cw_abs_wy = calculate_night_coldwaves_abs_wy(ds, temp_perc_wy[['TN2_5', 'TN1']])
WSD_abs_wy      = calculate_WSD_abs_wy(ds, temp_perc_wy[['TX95', 'TX90']])
CSD_abs_wy      = calculate_CSD_abs_wy(ds, temp_perc_wy[['TN5', 'TN10']])

# Merge all lazy graphs into one dataset
yearly_ds = xr.merge([TM, TX, TN, TNN_TXX, TVAR, DTR, coldwarm,
                      day_hw, night_hw, day_cw, night_cw, CSD, WSD, TNnumber_TXnumber, TM_bins,
                      coldwarm_abs, day_hw_abs, night_hw_abs, day_cw_abs, night_cw_abs, CSD_abs, WSD_abs,
                      PA, PWT, W, PWA, PVAR, PWVAR,
                      P95WT_P99WT, consec_counts, consec_totals,
                      PX1_PX5, PXM_PNM, P_bins,
                      coldwarm_wy, day_hw_wy, night_hw_wy, day_cw_wy, night_cw_wy, WSD_wy, CSD_wy,
                      coldwarm_abs_wy, day_hw_abs_wy, night_hw_abs_wy, day_cw_abs_wy, night_cw_abs_wy, WSD_abs_wy, CSD_abs_wy])

logger.info("Computing and writing yearly variables to disk...")
encoding = {var: {'zlib': True, 'complevel': 4} for var in yearly_ds.data_vars}
for yr in yearly_ds['year'].values:
    logger.info(f"Processing {yr}...")
    with ProgressBar():
        yr_ds = yearly_ds.sel(year=yr).compute()
    yr_ds = yr_ds.astype(np.float32)
    fname = OUT_YEARLY / f"yearly_vars_{yr}.nc"
    yr_ds.to_netcdf(fname, encoding=encoding)
    logger.info(f"Saved {fname.name}")
    del yr_ds

del TM, TX, TN, TNN_TXX, TVAR, DTR, coldwarm
del day_hw, night_hw, day_cw, night_cw, CSD, WSD, TNnumber_TXnumber, TM_bins
del coldwarm_abs, day_hw_abs, night_hw_abs, day_cw_abs, night_cw_abs, CSD_abs, WSD_abs
del PA, PWT, W, PWA, PVAR, PWVAR
del P95WT_P99WT, consec_counts, consec_totals, PX1_PX5, PXM_PNM, P_bins
del coldwarm_wy, day_hw_wy, night_hw_wy, day_cw_wy, night_cw_wy, WSD_wy, CSD_wy
del coldwarm_abs_wy, day_hw_abs_wy, night_hw_abs_wy, day_cw_abs_wy, night_cw_abs_wy, WSD_abs_wy, CSD_abs_wy
del yearly_ds

logger.info("Everything done.")