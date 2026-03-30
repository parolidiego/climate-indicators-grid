# %%

# ---------------------------------------------------------------------------
# PACKAGES
# ---------------------------------------------------------------------------

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar


# %%

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("climate_variables_percentiles.log", mode='w'),
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

YEARS           = range(1950, 2024)   # (start: inclusuve, end: exclusive -> ends 31/12 year -1) ==> not used for percentiles
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
# PERCENTILE THRESHOLDS
# ===========================================================================

def compute_rolling_percentiles(ds: xr.Dataset, variable: str, percentile: int,
                                year_start: int, year_end: int, window_size: int):
    """
    Input: ds[variable] [Grid, Day] - daily variable at the grid level
    Output: {TX|TN}_{percentile}p_{window_size}w [Grid, DayOfYear] -
            p-th percentile of variable within a rolling window centered on each day of year,
            computed over the reference period

    For each day of year (1-366), collects all values from days within a centered
    window_size-day window across all years in the reference period, then computes
    the percentile.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing `variable` with a `valid_time` dimension.
    variable : str
        Variable name in ds (e.g. 't_min', 't_max').
    percentile : int
        Percentile to compute (e.g. 10 or 90).
    year_start : int
        First year of the reference period (inclusive).
    year_end : int
        Last year of the reference period (inclusive).
    window_size : int
        Number of days in the centered rolling window (e.g. 5 means +-2 days around target day).

    Returns
    -------
    xr.Dataset
        Lazy Dataset with variable `{TX|TN}_p{percentile}_w{window_size}`
        and dimensions (day_of_year, latitude, longitude).
    """

    # Setting output variable name
    VARIABLE_ALIASES = {'t_min': 'TN', 't_max': 'TX'}
    var_alias = VARIABLE_ALIASES.get(variable, variable)
    var_name = f'{var_alias}_{percentile}p_{window_size}w'
    logger.info(f"Calculating {var_name} ({percentile}th percentile of {variable}, "
                f"{window_size}-day window, reference period {year_start}-{year_end})")

    if variable not in ds:
        raise KeyError(f"'{variable}' not found in dataset. Available variables: {list(ds.data_vars)}")

    # Filter for variable data only and filter to reference period for percentile calculation
    var_data = ds[variable].sel(valid_time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
    logger.info(f"Filtered to reference period: {len(var_data.valid_time)} time steps")

    # Remove Feb 29: percentile thresholds are indexed by day-of-year 1-365, so all years
    # must be treated as 365-day years to avoid DOY misalignment between leap and non-leap years
    feb29 = (var_data.valid_time.dt.month == 2) & (var_data.valid_time.dt.day == 29)
    var_data = var_data.sel(valid_time=~feb29)
    logger.info(f"After removing Feb 29: {len(var_data.valid_time)} time steps")
    # For leap year dates after Feb 28, dt.dayofyear would be 1 higher than the equivalent in non-leap year date
    # Subtracting 1 for those dates gives a consistent DOY 1-365 across all years.
    # This new DOY version is called normalized doy (norm_doy)
    is_leap_and_late = var_data.valid_time.dt.is_leap_year & (var_data.valid_time.dt.month > 2)
    norm_doy = var_data.valid_time.dt.dayofyear - is_leap_and_late.astype(int)

    # Add normalized day-of-year as a coordinate for window selection.
    var_data = var_data.assign_coords(dayofyear=('valid_time', norm_doy.values))

    half_window = window_size // 2
    logger.info(f"Building calculations graph (window +-{half_window} days)...")

    day_results = []

    # Process each day of the year
    for day in range(1, 366):
        # Get the doy indexes for days within the window
        window_days = []
        for offset in range(-half_window, half_window + 1):
            d = day + offset
            if d < 1:
                d += 365
            elif d > 365:
                d -= 365
            window_days.append(d)

        # Filter the variable data getting all data points within the window
        window_data = var_data.isel(valid_time=var_data.dayofyear.isin(window_days))

        # Calculate percentile lazily
        perc = (window_data
                .quantile(percentile / 100.0, dim='valid_time', skipna=True)
                .drop_vars('quantile')
                .expand_dims(day_of_year=[day]))
        day_results.append(perc)

    # Assembles 365 lazy quantile graphs into one
    result = xr.concat(day_results, dim='day_of_year')

    # Attributes and renaming of final result dataframe
    result = result.rename(var_name)
    result = result.assign_attrs(
        long_name=(f'{percentile}th percentile of {variable} with {window_size}-day window '
                   f'over {year_start}-{year_end}'),
        units=ds[variable].attrs.get('units', ''),
        percentile=percentile,
        window_size=window_size,
        year_start=year_start,
        year_end=year_end,
    )
    result = result.to_dataset()

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return result

def calculate_PW_pj(ds: xr.Dataset, percentile: int, year_start: int = 1979, year_end: int = 2019):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
    Output: PW_{percentile}p [Grid] - p-th percentile of wet day precipitation over the reference period

    Filters to the reference period and to wet days only (daily precipitation >= 1mm), then
    computes the p-th percentile of the resulting distribution for each grid cell across all
    wet days in the reference period, regardless of day of year. Yields one number per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec` with a `valid_time` dimension (daily frequency).
    percentile : int
        Percentile to compute (e.g., 95 for the 95th percentile, 99 for the 99th).
    year_start : int
        First year of the reference period (inclusive). Default: 1979.
    year_end : int
        Last year of the reference period (inclusive). Default: 2019.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `PW_{percentile}p` and dimensions (latitude, longitude).
        One value per grid cell, no time dimension.
    """

    # Setting output variable name
    var_name = f'PW_{percentile}p'
    logger.info(f"Calculating {var_name} ({percentile}th percentile of wet day precipitation "
                f"over reference period {year_start}-{year_end})")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")

    wet_day_threshold = 0.001  # 1mm expressed in metres

    # Filter for variable data only and filter to reference period for percentile calculation
    ds_ref = ds['tot_prec'].sel(valid_time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))

    # Masks to distinguish ocean cells (always NaN) from land cells with no wet days
    land_mask = ds_ref.notnull().any(dim='valid_time')
    wet_day_mask = (ds_ref >= wet_day_threshold).any(dim='valid_time')
    no_wet_days_on_land = land_mask & ~wet_day_mask

    logger.info(f"Building calculations graph...")

    # Lazily calculates the percentile
    PW_pj = (
        ds_ref
        # Filters for wet days only
        .where(ds_ref >= wet_day_threshold)
        # Calculates percentile over those wet days
        .quantile(percentile / 100.0, dim='valid_time', skipna=True)
        .drop_vars('quantile')
        # Land cells with no wet days get inf threshold (never exceeded); ocean NaNs are preserved
        .where(~no_wet_days_on_land, other=np.inf)
        # Renaming and assigning attributes
        .rename(var_name)
        .assign_attrs(
            long_name=f'{percentile}th percentile of wet day precipitation over {year_start}-{year_end}',
            units='m',
            percentile=percentile,
            year_start=year_start,
            year_end=year_end,
            wet_day_threshold_m=wet_day_threshold
        )
        .to_dataset()
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return PW_pj


# N. CHUNKS must be at least 2x n. cores (be careful not to have too big of chunks tho)
# Rechunking for percentiles calculations
ds_perc = ds.chunk({"valid_time": -1, "latitude": 500, "longitude": 800})

# Calculating temperature percentiles
TN10p5w  = compute_rolling_percentiles(ds_perc, 't_min', 10, PERC_YEAR_START, PERC_YEAR_END, window_size=5)
TX10p5w  = compute_rolling_percentiles(ds_perc, 't_max', 10, PERC_YEAR_START, PERC_YEAR_END, window_size=5)
TN90p5w  = compute_rolling_percentiles(ds_perc, 't_min', 90, PERC_YEAR_START, PERC_YEAR_END, window_size=5)
TX90p5w  = compute_rolling_percentiles(ds_perc, 't_max', 90, PERC_YEAR_START, PERC_YEAR_END, window_size=5)
TN90p15w = compute_rolling_percentiles(ds_perc, 't_min', 90, PERC_YEAR_START, PERC_YEAR_END, window_size=15)
TX90p15w = compute_rolling_percentiles(ds_perc, 't_max', 90, PERC_YEAR_START, PERC_YEAR_END, window_size=15)
TN10p15w = compute_rolling_percentiles(ds_perc, 't_min', 10, PERC_YEAR_START, PERC_YEAR_END, window_size=15)
TX10p15w = compute_rolling_percentiles(ds_perc, 't_max', 10, PERC_YEAR_START, PERC_YEAR_END, window_size=15)

logger.info("Computing temperature percentiles...")
with ProgressBar():
    TN10p5w  = TN10p5w.compute()
    TX10p5w = TX10p5w.compute()
    TN90p5w = TN90p5w.compute()
    TX90p5w = TX90p5w.compute()
    TN90p15w = TN90p15w.compute()
    TX90p15w = TX90p15w.compute()
    TN10p15w = TN10p15w.compute()
    TX10p15w = TX10p15w.compute()

# Merge computed datasets
temp_perc = xr.merge([TN10p5w, TX10p5w, TN90p5w, TX90p5w,
                      TN90p15w, TX90p15w, TN10p15w, TX10p15w])

logger.info("Writing temperature percentiles to disk..")
encoding = {var: {'zlib': True, 'complevel': 8, 'dtype': 'float32'} for var in df.data_vars}
temp_perc.to_netcdf(OUT_PERC / "temp_percentiles.nc", encoding=encoding)
logger.info("Done.")


# Calculating wet days precipitation percentiles
PW_95p = calculate_PW_pj(ds_perc, percentile=95, year_start=PERC_YEAR_START, year_end=PERC_YEAR_END)
PW_99p = calculate_PW_pj(ds_perc, percentile=99, year_start=PERC_YEAR_START, year_end=PERC_YEAR_END)

logger.info("Computing wet days precipitation percentiles...")
with ProgressBar():
    PW_95p = PW_95p.compute()
    PW_99p = PW_99p.compute()

# Merge computed datasets
wet_days_perc = xr.merge([PW_95p, PW_99p])

logger.info("Writing precipitation percentiles to disk..")
encoding = {var: {'zlib': True, 'complevel': 4} for var in wet_days_perc.data_vars}
wet_days_perc.to_netcdf(OUT_PERC / "precip_percentiles.nc", encoding=encoding)
logger.info("Done.")

del TN10p5w, TX10p5w, TN90p5w, TX90p5w, TN90p15w, TX90p15w, TN10p15w, TX10p15w
del PW_95p, PW_99p
del temp_perc, wet_days_perc

logger.info("Everything done.")