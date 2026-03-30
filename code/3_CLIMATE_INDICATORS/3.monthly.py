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
        logging.FileHandler("climate_variables_monthly.log", mode='w'),
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
# MONTHLY VARIABLES
# ===========================================================================


def calculate_P_jm(ds: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
    Output: P_jm [Grid, Month] - total monthly precipitation at the grid level

    Sums daily precipitation values within each calendar month per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `P_jm` and dimensions (valid_time, latitude, longitude)
        where valid_time is at monthly frequency (month-end timestamps).
    """
    logger.info("Calculating P_jm (total monthly precipitation at grid cell level)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")

    # Summing precipitation for every month through .resample and .sum
    # min_count=1: return NaN (not 0) for months where all values are NaN (e.g. ocean)
    P_jm = (
        ds['tot_prec']
        .resample(valid_time='ME')
        .sum(dim='valid_time', skipna=True, min_count=1)
        .assign_attrs(long_name='Total monthly precipitation', units='m')
        .to_dataset(name='P_jm')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return P_jm


def calculate_TX_m(ds: xr.Dataset):
    """
    Input: t_max [Grid, Day] - maximum daily 2-metre temperature at the grid level
    Output: TX_m [Grid, Month] - monthly mean of daily maximum temperature at the grid level

    Averages daily t_max values within each calendar month per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_max` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `TX_m` and dimensions (valid_time, latitude, longitude)
        where valid_time is at monthly frequency (month-end timestamps).
    """
    logger.info("Calculating TX_m (monthly mean of daily maximum temperature at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")

    TX_m = (
        ds['t_max']
        .resample(valid_time='ME')
        .mean(dim='valid_time', skipna=True)
        .assign_attrs(long_name='Monthly mean of daily maximum temperature', units=ds['t_max'].attrs.get('units', 'K'))
        .to_dataset(name='TX_m')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return TX_m


def calculate_TN_m(ds: xr.Dataset):
    """
    Input: t_min [Grid, Day] - minimum daily 2-metre temperature at the grid level
    Output: TN_m [Grid, Month] - monthly mean of daily minimum temperature at the grid level

    Averages daily t_min values within each calendar month per grid cell.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `t_min` with
        a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `TN_m` and dimensions (valid_time, latitude, longitude)
        where valid_time is at monthly frequency (month-end timestamps).
    """
    logger.info("Calculating TN_m (monthly mean of daily minimum temperature at grid cell level)")

    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")

    TN_m = (
        ds['t_min']
        .resample(valid_time='ME')
        .mean(dim='valid_time', skipna=True)
        .assign_attrs(long_name='Monthly mean of daily minimum temperature', units=ds['t_min'].attrs.get('units', 'K'))
        .to_dataset(name='TN_m')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return TN_m


# Calculating monthly variables
P_jm  = calculate_P_jm(ds)
TX_m  = calculate_TX_m(ds)
TN_m  = calculate_TN_m(ds)

logger.info("Computing monthly variables...")
with ProgressBar():
    P_jm = P_jm.compute()
    TX_m = TX_m.compute()
    TN_m = TN_m.compute()

# Merge computed datasets
monthly_ds = xr.merge([P_jm, TX_m, TN_m])

monthly_ds = monthly_ds.astype(np.float32)

logger.info("Writing monthly variables to disk..")
encoding = {var: {'zlib': True, 'complevel': 4} for var in monthly_ds.data_vars}
for label, group in monthly_ds.resample(valid_time='YE'):
    ts = pd.Timestamp(label)
    fname = OUT_MONTHLY / f"monthly_vars_{ts.year}.nc"
    group.to_netcdf(fname, encoding=encoding)
    logger.info(f"Saved {fname.name}")

del P_jm, TX_m, TN_m
del monthly_ds

logger.info("Everything done.")