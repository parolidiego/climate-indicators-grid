# %%

# ---------------------------------------------------------------------------
# PACKAGES
# ---------------------------------------------------------------------------

import logging
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar

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
        logging.FileHandler("climate_variables_4b_temp.log", mode='w'),
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

YEARS      = range(1950, 2024)
YEAR_START = YEARS.start
YEAR_END   = YEARS.stop - 1

OUT_PERC   = OUTPUT_DIR / "percentiles"
OUT_YEARLY = OUTPUT_DIR / "yearly"

dask.config.set(scheduler='threads')


# %%

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

logger.info("Opening ERA5 files (lazy)...")

files = sorted(INPUT_DIR.glob("era5_daily_*.nc"))
ds = xr.open_mfdataset(files, combine='by_coords', engine="netcdf4")
ds = ds.sel(valid_time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

logger.info(f"Graph opened: {ds}")

logger.info("Opening full-year temperature percentiles (lazy)...")
temp_perc_wy = xr.open_dataset(OUT_PERC / "temp_percentiles_fullyear.nc", chunks="auto")


# %%

# ===========================================================================
# PURELY ABSOLUTE-THRESHOLD VARIABLES (no percentile condition)
# ===========================================================================


def calculate_TNm5_TNm10_TX30(ds: xr.Dataset):
    """
    Input: t_min [Grid, Day] - daily minimum 2-metre temperature at the grid level
           t_max [Grid, Day] - daily maximum 2-metre temperature at the grid level
    Output: TNm5  [Grid, Year] - number of days with t_min < -5°C
            TNm10 [Grid, Year] - number of days with t_min < -10°C
            TX30  [Grid, Year] - number of days with t_max > 30°C

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
        Lazy dataset with variables TNm5, TNm10, TX30 and dimensions (year, latitude, longitude).
    """
    logger.info("Calculating TNm5, TNm10, TX30 (fixed temperature threshold exceedance counts at grid cell level)")

    for var in ['t_min', 't_max']:
        if var not in ds:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds.data_vars)}")

    years = np.unique(ds['valid_time'].dt.year.values)

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

    logger.info("Calculations graph is ready. Call .compute() to run the actual calculations.")

    return xr.Dataset({'TNm5': TNm5, 'TNm10': TNm10, 'TX30': TX30})


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


# %%

# ---------------------------------------------------------------------------
# BUILD LAZY GRAPH
# ---------------------------------------------------------------------------

fixed_thr       = calculate_TNm5_TNm10_TX30(ds)
coldwarm_abs_wy = calculate_coldwarm_abs_wy(ds, temp_perc_wy)
day_hw_abs_wy   = calculate_day_heatwaves_abs_wy(ds, temp_perc_wy[['TX97_5', 'TX99']])
night_hw_abs_wy = calculate_night_heatwaves_abs_wy(ds, temp_perc_wy[['TN97_5', 'TN99']])
day_cw_abs_wy   = calculate_day_coldwaves_abs_wy(ds, temp_perc_wy[['TX2_5', 'TX1']])
night_cw_abs_wy = calculate_night_coldwaves_abs_wy(ds, temp_perc_wy[['TN2_5', 'TN1']])
WSD_abs_wy      = calculate_WSD_abs_wy(ds, temp_perc_wy[['TX95', 'TX90']])
CSD_abs_wy      = calculate_CSD_abs_wy(ds, temp_perc_wy[['TN5', 'TN10']])

new_vars = xr.merge([fixed_thr, coldwarm_abs_wy, day_hw_abs_wy, night_hw_abs_wy,
                     day_cw_abs_wy, night_cw_abs_wy, WSD_abs_wy, CSD_abs_wy])


# %%

# ---------------------------------------------------------------------------
# PATCH EXISTING YEARLY FILES
# ---------------------------------------------------------------------------

logger.info("Patching existing yearly files with new variables...")
encoding_opts = {'zlib': True, 'complevel': 4}

for yr in new_vars['year'].values:
    logger.info(f"Processing {yr}...")
    fname = OUT_YEARLY / f"yearly_vars_{yr}.nc"
    tmp   = fname.with_suffix('.tmp.nc')

    with ProgressBar():
        yr_new = new_vars.sel(year=yr).compute()
    yr_new = yr_new.astype(np.float32)

    with xr.open_dataset(fname) as existing:
        merged = xr.merge([existing, yr_new])
        encoding = {v: encoding_opts for v in merged.data_vars}
        merged.to_netcdf(tmp, encoding=encoding)

    tmp.replace(fname)
    logger.info(f"Patched {fname.name}")
    del yr_new

del fixed_thr, coldwarm_abs_wy, day_hw_abs_wy, night_hw_abs_wy
del day_cw_abs_wy, night_cw_abs_wy, WSD_abs_wy, CSD_abs_wy
del new_vars

logger.info("Everything done.")
