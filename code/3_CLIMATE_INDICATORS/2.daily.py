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
        logging.FileHandler("climate_variables_daily.log", mode='w'),
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
# DAILY VARIABLES
# ===========================================================================

def calculate_PW_d(ds: xr.Dataset):
    """
    Input: tot_prec [Grid, Day] - total daily precipitation at the grid level
    Output: PW_d [Grid, Day] - daily precipitation masked to wet days only (NaN on dry days)

    Filters precipitation data to retain only wet days.
    A day is considered wet if tot_prec >= 1mm (0.001m). Dry days are masked to NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variable `tot_prec` with a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy Dataset with variable `PW_d` and dimensions (valid_time, latitude, longitude).
    """
    logger.info("Calculating PW_d (wet day precipitation, threshold=1mm)")

    if 'tot_prec' not in ds:
        raise KeyError(f"'tot_prec' not found in dataset. Available variables: {list(ds.data_vars)}")

    wet_day_threshold = 0.001  # 1mm expressed in metres

    # Masking the data to create the variable wanted
    PW_d = (
        ds['tot_prec']
        .where(ds['tot_prec'] >= wet_day_threshold)
        .rename('PW_d')
        .to_dataset()
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return PW_d

def calculate_DTR_d(ds: xr.Dataset):
    """
    Input: t_max [Grid, Day] - maximum daily 2-metre temperature at the grid level
           t_min [Grid, Day] - minimum daily 2-metre temperature at the grid level
    Output: DTR_d [Grid, Day] - diurnal temperature range at the grid level

    Computed as: DTR_d = t_max - t_min for each grid cell and day.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily-loaded ERA5 dataset containing the variables `t_max` and `t_min`
        with a `valid_time` dimension (daily frequency).

    Returns
    -------
    xr.Dataset
        Lazy dataset with variable `DTR_d` and dimensions (valid_time, latitude, longitude).
    """
    logger.info("Calculating DTR_d (diurnal temperature range at grid cell level)")

    if 't_max' not in ds:
        raise KeyError(f"'t_max' not found in dataset. Available variables: {list(ds.data_vars)}")
    if 't_min' not in ds:
        raise KeyError(f"'t_min' not found in dataset. Available variables: {list(ds.data_vars)}")

    # Subtracting to create the variable wanted
    DTR_d = (
        (ds['t_max'] - ds['t_min'])
        .assign_attrs(long_name='Temperature range', units='Celsius')
        .to_dataset(name='DTR_d')
    )

    logger.info(f"Calculations graph is ready. Call .compute() to run the actual calculations.")

    return DTR_d


# Calculating daily variables
PW_d = calculate_PW_d(ds)
DTR_d = calculate_DTR_d(ds)

logger.info("Computing and saving daily variables directly...")
daily_ds = xr.merge([
    ds[['t_mean', 't_min', 't_max', 'tot_prec']].rename({
        't_mean': 'TM_d', 't_min': 'TN_d', 't_max': 'TX_d', 'tot_prec': 'P_d'
    }),
    PW_d, DTR_d
])

encoding = {var: {'zlib': True, 'complevel': 4} for var in daily_ds.data_vars}
for label, group in daily_ds.resample(valid_time='ME'):
    ts = pd.Timestamp(label)
    fname = OUT_DAILY / f"daily_vars_{ts.year}_{ts.month:02d}.nc"
    with ProgressBar():
        group.to_netcdf(fname, encoding=encoding)
    logger.info(f"Saved {fname.name}")


del PW_d, DTR_d
del daily_ds

logger.info("Everything done.")