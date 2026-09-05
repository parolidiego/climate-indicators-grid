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
        logging.FileHandler("climate_variables_add_drought_vars.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# %%

# ---------------------------------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data/out"

YEARS      = range(1950, 2024)
YEAR_START = YEARS.start
YEAR_END   = YEARS.stop - 1

OUT_MONTHLY = OUTPUT_DIR / "monthly"
OUT_YEARLY  = OUTPUT_DIR / "yearly"
# Temporary directory for the intermediate masked monthly NetCDF files produced in Phase 1.
# Deleted at the end of the script once the yearly files have been updated.
TMP_DIR     = OUTPUT_DIR / "tmp_drought"
TMP_DIR.mkdir(parents=True, exist_ok=True)

dask.config.set(scheduler='threads')

# Drought index variables present in the monthly files (produced by 4.droughts_monthly.R).
INDICES = ['spi_3', 'spi_6', 'spi_12', 'spei_3', 'spei_6', 'spei_12']

# Severity tiers, also used in variable name construction (threshold, human-readable label).
# A drought event happens when for at least MIN_DURATION consecutive months the index is below the threshold.
THRESHOLDS = {
    'n': (-1.0, 'normal'),
    's': (-1.5, 'severe'),
    'x': (-2.0, 'extreme'),
}

# Minimum consecutive months below the threshold required for an event to be classified as a drought.
# Run detection operates on the full multi-year time series, so an event that starts in December
# and continues into January of the next year counts as a single continuous event.
MIN_DURATION = 3


# %%

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

logger.info("Opening monthly ERA5 files (lazy)...")

monthly_files = sorted(OUT_MONTHLY.glob("monthly_vars_[0-9][0-9][0-9][0-9].nc"))
if not monthly_files:
    raise FileNotFoundError(f"No monthly_vars_YYYY.nc files found in {OUT_MONTHLY}")

# Open all yearly monthly files as a single lazy dataset. No data is read from disk yet.
ds_monthly = xr.open_mfdataset(monthly_files, combine='by_coords', engine='netcdf4')
# Slice to the configured period
ds_monthly = ds_monthly.sel(valid_time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

logger.info(f"Graph opened: {ds_monthly}")


# %%

# ===========================================================================
# VARIABLE CALCULATION
# ===========================================================================


def drought_event_ts(x, threshold, min_duration):
    """
    Apply drought event filter to a 1-D monthly time series.

    Months belonging to qualifying drought events (consecutive runs of length
    >= min_duration with index <= threshold) retain their original index values;
    all other months are set to 0. Cells where the entire series is NaN
    (e.g. ocean) return NaN.

    Run detection operates on the full multi-year series so that events
    spanning a December-January boundary are treated as continuous.

    Parameters
    ----------
    x : np.ndarray, shape (n_months,)
        Full monthly time series for a single grid cell.
    threshold : float
        Index threshold below which a month is considered to be in drought.
    min_duration : int
        Minimum number of consecutive months required to qualify as a drought event.

    Returns
    -------
    np.ndarray, shape (n_months,)
        Monthly values retained for qualifying drought months, 0 elsewhere, NaN for all-NaN cells.
    """
    # Ocean cells have no SPI/SPEI values at all; return NaN to propagate the missing-data flag.
    if np.all(np.isnan(x)):
        return np.full_like(x, np.nan)

    # Output array: starts as all zeros; qualifying drought months will have their original value written in.
    out = np.zeros_like(x)

    # Boolean mask: True only where the index is valid (not NaN) AND at or below the drought threshold.
    # NaN months (e.g. first months of SPI-12 due to the accumulation window) are excluded.
    below = (~np.isnan(x)) & (x <= threshold)

    # If no month in the full series is below the threshold, the output is all zeros — return early.
    if not below.any():
        return out

    # Convert the boolean mask to integers (0/1) and compute the first difference to detect run boundaries.
    # Prepending and appending 0 ensures that runs starting at index 0 or ending at the last index
    # produce a detectable +1 / -1 transition at the boundary.
    # A value of +1 in changes marks where a consecutive below-threshold run starts;
    # a value of -1 marks where it ends.
    changes = np.diff(below.astype(np.int8), prepend=np.int8(0), append=np.int8(0))

    # np.where returns a tuple (one element per dimension); [0] extracts the 1-D index array.
    starts = np.where(changes ==  1)[0]   # indices where each run begins
    ends   = np.where(changes == -1)[0]   # indices where each run ends (exclusive, like a Python slice)

    # For each detected run, check whether it meets the minimum duration requirement.
    # If yes, copy the original index values into the output for those months;
    # months that are part of a run that is too short stay at 0.
    for s, e in zip(starts, ends):
        if (e - s) >= min_duration:
            out[s:e] = x[s:e]

    return out


def calculate_drought_metrics(ds_monthly: xr.Dataset):
    """
    Input: spi_3, spi_6, spi_12, spei_3, spei_6, spei_12 [Grid, Month]
           Monthly SPI/SPEI indices produced by 4.droughts_monthly.R.
    Output: {index}_sev_{tier}, {index}_dur_{tier}  [Grid, Year]
            Annual drought severity and duration for each index × severity tier combination.

    Drought severity = sum of absolute index values over qualifying drought months in a year.
    Drought duration = count of qualifying drought months in a year.

    A qualifying drought event is a consecutive run of at least MIN_DURATION months where
    the index <= threshold. Run detection is applied to the full multi-year series so events
    crossing a December-January boundary are captured correctly; annual metrics are then
    computed per calendar year.

    Output variable naming:
      {index}_sev_n / {index}_dur_n : normal drought  (index <= -1.0)
      {index}_sev_s / {index}_dur_s : severe drought  (index <= -1.5)
      {index}_sev_x / {index}_dur_x : extreme drought (index <= -2.0)
    where {index} is one of spi_3, spi_6, spi_12, spei_3, spei_6, spei_12.

    Two-phase approach to avoid recomputing drought_event_ts 74 times:

    Phase 1 — Masked monthly temp files:
      For each of the 18 (index × tier) combinations, apply drought_event_ts once across
      the full multi-year time series and write the masked monthly values to a compressed
      temp NetCDF file chunked as (12, 30, 60) — one year per time chunk. This is the
      expensive step (Python run-length detection per grid cell), but it runs only once.

    Phase 2 — Annual aggregation from temp files:
      For each year and each combination, read exactly 12 months from the corresponding
      temp file (one on-disk time chunk), then compute severity and duration directly
      as a simple sum — no groupby needed. Each yearly file is updated in a single pass.

    Parameters
    ----------
    ds_monthly : xr.Dataset
        Lazily-loaded monthly dataset containing spi_3, spi_6, spi_12,
        spei_3, spei_6, spei_12 with a valid_time dimension (monthly frequency).

    Returns
    -------
    dict[int, xr.Dataset]
        Maps each year to a Dataset containing the 36 new variables (latitude, longitude).
    """
    logger.info("Calculating drought severity and duration metrics")

    for var in INDICES:
        if var not in ds_monthly:
            raise KeyError(f"'{var}' not found in dataset. Available variables: {list(ds_monthly.data_vars)}")

    # Rechunk so each grid cell's full time series fits in a single dask task.
    # valid_time=-1 puts all 888 months in one chunk: necessary because drought_event_ts
    # receives the complete time series per cell and cannot be split along the time axis.
    # Spatial dimensions are chunked small (30×60) so each dask task stays manageable in memory
    # (~30 × 60 × 888 × 4 bytes ≈ 6 MB per task).
    ds_m = ds_monthly[INDICES].chunk({'valid_time': -1, 'latitude': 30, 'longitude': 60})

    # -----------------------------------------------------------------------
    # PHASE 1 — Compute masked monthly series and write to temp files
    #
    # Temp files use chunksizes=(12, 30, 60): one 12-month year per time chunk.
    # This means reading a single year later requires exactly one disk read per
    # spatial chunk, rather than loading the entire 888-month series.
    # -----------------------------------------------------------------------

    tmp_paths = {}

    for var in INDICES:
        for tier, (threshold, label) in THRESHOLDS.items():
            tmp_path = TMP_DIR / f"drought_masked_{var}_{tier}.nc"
            # Register the path unconditionally so Phase 2 can find it even if we skip below.
            tmp_paths[(var, tier)] = tmp_path

            # Skip if the file already exists — allows resuming after a crash without
            # recomputing combinations that were already written.
            if tmp_path.exists():
                logger.info(f"  Temp file already exists, skipping: {tmp_path.name}")
                continue

            logger.info(f"  Computing masked monthly series: {var}, tier={tier} (threshold<={threshold})")

            # Apply drought_event_ts to every grid cell along the full time series.
            # apply_ufunc with vectorize=True calls the function once per (lat, lon) cell,
            # passing all 888 months as a 1-D array — required so December-January events
            # are treated as a single continuous run rather than broken at the year boundary.
            # input_core_dims=[['valid_time']]: "pass the entire time axis to the function, don't loop over it"
            # output_core_dims=[['valid_time']]: "the function also returns a full time axis"
            # dask='parallelized': keep the computation lazy; dask will parallelize over spatial chunks
            da_masked = xr.apply_ufunc(
                drought_event_ts,
                ds_m[var],
                kwargs=dict(threshold=threshold, min_duration=MIN_DURATION),
                input_core_dims=[['valid_time']],
                output_core_dims=[['valid_time']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
            )
            # apply_ufunc places the output core dimension (valid_time) last, so we need to
            # transpose back to the original order (valid_time, latitude, longitude).
            da_masked = da_masked.transpose(*ds_m[var].dims)
            # apply_ufunc operates on raw numpy arrays and does not preserve coordinate metadata.
            # Reassign the datetime values so that .sel(valid_time=...) works correctly in Phase 2.
            da_masked = da_masked.assign_coords(valid_time=ds_m['valid_time'])

            # Trigger the actual dask computation and write to disk.
            # to_netcdf processes the data chunk by chunk, so the full 888-month array
            # is never loaded into memory all at once.
            # chunksizes=(12, 30, 60): store one year per time chunk on disk so that
            # Phase 2 can read a single year with a single disk access per spatial chunk.
            # zlib compression is effective here because most months are 0 (non-drought).
            with ProgressBar():
                da_masked.to_dataset(name='masked').to_netcdf(
                    tmp_path,
                    encoding={'masked': {
                        'zlib': True, 'complevel': 4, 'dtype': 'float32',
                        'chunksizes': (12, 30, 60),  # one year per time chunk
                    }},
                )
            logger.info(f"  Written: {tmp_path.name}")

    # -----------------------------------------------------------------------
    # PHASE 2 — Compute annual severity and duration per year from temp files
    #
    # Open all temp files lazily (outside the year loop — one open per combo).
    # For each year, .sel() reads exactly 12 months (one time chunk) per spatial
    # chunk and computes severity and duration as a direct sum.
    # min_count=1 ensures all-NaN cells (ocean) yield NaN rather than 0.
    # -----------------------------------------------------------------------

    logger.info("Opening temp files and computing annual metrics year by year...")

    # Open all 18 temp files lazily outside the year loop: file handles are reused
    # across years, avoiding the overhead of reopening the same file 74 times each.
    # chunks={'valid_time': 12} aligns dask chunks with the on-disk chunksizes from Phase 1,
    # so each .sel(valid_time=...) slice maps to exactly one disk read per spatial chunk.
    masked_das = {
        (var, tier): xr.open_dataset(
            tmp_paths[(var, tier)],
            chunks={'valid_time': 12, 'latitude': 30, 'longitude': 60},
        )['masked']
        for var in INDICES for tier in THRESHOLDS
    }

    # Will hold one xr.Dataset per year, each with all 36 new variables as 2-D (lat, lon) arrays.
    annual_results = {}

    for yr in YEARS:
        logger.info(f"  Aggregating year {yr}...")
        # Accumulates the 36 (var_name → DataArray) pairs for this year before assembling into a Dataset.
        yr_vars = {}

        for var in INDICES:
            for tier, (threshold, label) in THRESHOLDS.items():
                # Select the 12 months of this year from the temp file.
                # Because the file was stored with chunksizes=(12,...), this slice
                # maps to one on-disk chunk per spatial block — only those 12 months
                # are read from disk, not the full 888-month series.
                da_year = masked_das[(var, tier)].sel(
                    valid_time=slice(f'{yr}-01-01', f'{yr}-12-31')
                )

                # --- Severity ---
                # Sum the absolute values of the qualifying drought months.
                # Drought index values are negative, so |value| gives the magnitude.
                # Non-drought months are 0, contributing nothing to the sum.
                # skipna=True: ignore NaN months (e.g. the first months of SPI-12 are NaN
                #   because the 12-month accumulation window is not yet complete).
                # min_count=1: if ALL 12 months are NaN (ocean cell), return NaN instead of 0,
                #   because 0 would be indistinguishable from a land cell with no drought.
                sev = (
                    np.abs(da_year)
                    .sum(dim='valid_time', skipna=True, min_count=1)
                    .assign_attrs(
                        long_name=(
                            f'Annual drought severity '
                            f'({var.upper().replace("_", "-")}, {label}, index<={threshold}, '
                            f'min_duration={MIN_DURATION} months)'
                        ),
                        units='std',
                    )
                )

                # --- Duration ---
                # Count how many months in this year belong to a qualifying drought event.
                # Step 1: (da_year != 0) → True (1) for drought months, False (0) for non-drought months.
                #   Note: in numpy, NaN != 0 evaluates to True, so ocean months would be counted
                #   as drought months without the next step.
                # Step 2: .where(da_year.notnull()) → replace positions where the original value
                #   was NaN with NaN, undoing the spurious True from Step 1 for ocean months.
                # Step 3: .sum(..., min_count=1) → count the True values (drought months) per cell;
                #   ocean cells (all NaN after Step 2) get NaN via min_count=1.
                dur = (
                    (da_year != 0)
                    .where(da_year.notnull())
                    .sum(dim='valid_time', skipna=True, min_count=1)
                    .assign_attrs(
                        long_name=(
                            f'Annual drought duration '
                            f'({var.upper().replace("_", "-")}, {label}, index<={threshold}, '
                            f'min_duration={MIN_DURATION} months)'
                        ),
                        units='months',
                    )
                )

                yr_vars[f'{var}_sev_{tier}'] = sev
                yr_vars[f'{var}_dur_{tier}'] = dur

        # Bundle all 36 lazy DataArrays into a Dataset, trigger computation, and cast to float32.
        # .compute() reads the 12-month slices from all 18 temp files and evaluates the sums;
        # all 36 variables are computed in a single dask scheduler pass.
        # .assign_coords(year=yr) attaches the integer year as a scalar coordinate so the
        # Dataset can be merged with the existing yearly files that use the same convention.
        with ProgressBar():
            annual_results[yr] = (
                xr.Dataset(yr_vars)
                .compute()
                .astype(np.float32)
                .assign_coords(year=yr)
            )

    return annual_results


# %%

# ---------------------------------------------------------------------------
# COMPUTE AND WRITE
# ---------------------------------------------------------------------------

annual_results = calculate_drought_metrics(ds_monthly)

logger.info("Writing new variables into existing yearly files...")

for yr in YEARS:
    logger.info(f"Processing {yr}...")

    new_yr    = annual_results[yr]
    fname     = OUT_YEARLY / f"yearly_vars_{yr}.nc"
    # Write to a .tmp.nc file first so the original is never left in a partially-written
    # state if the script crashes mid-write; the rename at the end is atomic.
    tmp_fname = fname.with_suffix('.tmp.nc')

    # Open the existing yearly file, merge the 36 new variables with those already present,
    # and load everything into memory before writing so the source file can be safely replaced.
    with xr.open_dataset(fname, decode_timedelta=False) as existing:
        merged = xr.merge([existing, new_yr]).load()

    encoding = {var: {'zlib': True, 'complevel': 4} for var in merged.data_vars}
    merged.to_netcdf(tmp_fname, encoding=encoding)
    # Atomically replace the original file with the updated version.
    tmp_fname.replace(fname)

    logger.info(f"Updated {fname.name}")
    del new_yr, merged

# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------

logger.info("Removing temp files...")
for tmp_path in (TMP_DIR / f"drought_masked_{var}_{tier}.nc"
                 for var in INDICES for tier in THRESHOLDS):
    if tmp_path.exists():
        tmp_path.unlink()

# Remove the temp directory itself if it is now empty.
if TMP_DIR.exists() and not any(TMP_DIR.iterdir()):
    TMP_DIR.rmdir()

logger.info("Everything done.")