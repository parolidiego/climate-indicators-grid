import pandas as pd
import xarray as xr
from pathlib import Path
import logging
from datetime import datetime
from textwrap import dedent

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base paths
BASE_PATH = Path(__file__).parent.parent.parent
TEMP_PATH = BASE_PATH / "data/tmp/output_api/2m_temperature"
PREC_PATH = BASE_PATH / "data/tmp/output_api/total_precipitation"
OUTPUT_PATH = BASE_PATH / "data/in/era5_daily"

# Create output directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Create log file path
LOG_FILE = BASE_PATH / "aggregating.log"

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROCESSING FUNCTION
# ============================================================================

def process_monthly_data(year, month, skip_existing=False):
    """
    Process and merge ERA5 data for a single month.

    Parameters:
    -----------
    year : int
        Year to process (e.g., 1950)
    month : int
        Month to process (1-12)
    """
    logger.info(f"{'=' * 80}")
    logger.info(f"Processing {year}-{month:02d}")
    logger.info(f"{'=' * 80}")

    # Check if output file already exists
    output_file = OUTPUT_PATH / f"era5_daily_{year}_{month:02d}.nc"
    if skip_existing and output_file.exists():
        logger.info(f"     Output file already exists, skipping: {output_file.name}")
        return "skipped"

    try:
        # ====================================================================
        # LOAD CURRENT MONTH DATA
        # ====================================================================

        logger.info("Loading datasets...")

        temp_file = TEMP_PATH / f"era5_2m_temperature_{year}_{month:02d}.nc"
        prec_file = PREC_PATH / f"era5_total_precipitation_{year}_{month:02d}.nc"

        ds_temp = xr.open_dataset(temp_file)
        ds_prec = xr.open_dataset(prec_file)


        # ====================================================================
        # RENAME VARIABLES
        # ====================================================================

        logger.info("Renaming variables...")

        ds_prec = ds_prec.rename({'tp': 'tot_prec'})


        # ====================================================================
        # SHIFT PRECIPITATION TIME
        # ====================================================================

        logger.info("Shifting precipitation variable and filtering to current month...")

        original_prec_start = ds_prec.valid_time.values[0]
        original_prec_end = ds_prec.valid_time.values[-1]

        ds_prec['valid_time'] = ds_prec['valid_time'] - pd.Timedelta(days=1)

        # Filter to keep only current month after shifting
        ds_prec = ds_prec.sel(valid_time=ds_prec.valid_time.dt.month == month)

        logger.info(f"  Original: {original_prec_start} to {original_prec_end}")
        logger.info(f"  Shifted:  {ds_prec.valid_time.values[0]} to {ds_prec.valid_time.values[-1]}")


        # ====================================================================
        # LOAD NEXT MONTH'S FIRST DAY FOR PRECIPITATION
        # ====================================================================

        logger.info("Loading next month's precipitation for last day...")

        # Calculate next month/year
        next_month = month + 1
        next_year = year
        if next_month > 12:
            next_month = 1
            next_year = year + 1

        next_prec_file = PREC_PATH / f"era5_total_precipitation_{next_year}_{next_month:02d}.nc"

        if not next_prec_file.exists():
            logger.warning(f" Next month file not found: {next_prec_file.name}")
            logger.warning("  Last day of month will be missing from precipitation data")
            ds_prec_complete = ds_prec
        else:

            ds_next_prec = xr.open_dataset(next_prec_file)
            ds_next_prec = ds_next_prec.rename({'tp': 'tot_prec'})

            # Extract first timestep and shift it
            ds_last_day = ds_next_prec.isel(valid_time=0)
            ds_last_day['valid_time'] = ds_last_day['valid_time'] - pd.Timedelta(days=1)
            ds_last_day = ds_last_day.expand_dims('valid_time')

            logger.info(f"  Last day extracted: {ds_last_day.valid_time.values[0]}")

            # Concatenate
            ds_prec_complete = xr.concat([ds_prec, ds_last_day], dim='valid_time')

            ds_next_prec.close()

        # ====================================================================
        # MERGE ALL DATASETS
        # ====================================================================

        logger.info("Merging all variables...")

        ds_merged = xr.merge([
            ds_temp,
            ds_prec_complete
        ], join='outer', compat='no_conflicts')

        logger.info(f"  Time range: {ds_merged.valid_time.values[0]} to {ds_merged.valid_time.values[-1]}")
        logger.info(f"  Time steps: {len(ds_merged.valid_time)}")
        logger.info(f"  Variables: {list(ds_merged.data_vars)}")


        # ====================================================================
        # ADD METADATA
        # ====================================================================

        # Rewrite attributes
        ds_merged.attrs['description'] = f'ERA5-Land daily data for {year}-{month:02d}.'
        ds_merged.attrs['created'] = f'Dataset assembled on {datetime.now().date()}, using data obtained through CDS API from Copernicus Climate Change Service. For information contact Diego Paroli (diego.paroli@cmcc.it)'
        ds_merged.attrs["note"] = dedent("""
            Temperature and precipitation variables come from the following data table: ERA5-Land hourly data from 1950 to present.
            
            Temperature variable processing:
            Temperature has been aggregated from the hourly to the daily level, calculating three variables: TMAX, TMIN, and TMEAN.
            
            Precipitation variable processing:
            Originally, precipitation values at date D time 00:00 represented the 24-hour accumulation for the previous calendar day (D-1).
            This dataset has been adjusted so that values at 00:00 on date D represent the total precipitation accumulated during date D (from 00:00 to 23:59).
        """).strip()

        ds_merged["t_max"].attrs.clear()
        ds_merged["t_max"].attrs["long_name"] = '2 metre temperature - daily maximum'
        ds_merged["t_max"].attrs["units"] = 'Celsius'
        ds_merged["t_min"].attrs.clear()
        ds_merged["t_min"].attrs["long_name"] = '2 metre temperature - daily minimum'
        ds_merged["t_min"].attrs["units"] = 'Celsius'
        ds_merged["t_mean"].attrs.clear()
        ds_merged["t_mean"].attrs["long_name"] = '2 metre temperature - daily mean'
        ds_merged["t_mean"].attrs["units"] = 'Celsius'
        ds_merged["tot_prec"].attrs.clear()
        ds_merged["tot_prec"].attrs["long_name"] = 'Total daily precipitation'
        ds_merged["tot_prec"].attrs["units"] = 'Metre'


        # ====================================================================
        # SAVE MERGED DATASET
        # ====================================================================

        output_file = OUTPUT_PATH / f"era5_daily_{year}_{month:02d}.nc"

        logger.info(f"Saving to: {output_file}")

        # Set encoding for compression
        encoding = {}
        for var in ds_merged.data_vars:
            encoding[var] = {'zlib': True, 'complevel': 4}

        ds_merged.to_netcdf(output_file, encoding=encoding)

        logger.info("File saved successfully")


        # ====================================================================
        # CLEANUP
        # ====================================================================

        ds_temp.close()
        ds_prec.close()

        return True

    except Exception as e:
        logger.error(f"✗ Error processing {year}-{month:02d}: {str(e)}")
        return False


def process_date_range(start_year, start_month, end_year, end_month, skip_existing=True):
    """
    Process multiple months of ERA5 data.

    Parameters:
    -----------
    start_year : int
        Starting year (e.g., 1950)
    start_month : int
        Starting month (1-12)
    end_year : int
        Ending year (e.g., 1950)
    end_month : int
        Ending month (1-12)
    skip_existing : bool
        If True, skip months where output file already exists (default: True)
    """
    logger.info(f"\n{'#' * 80}")
    logger.info(f"# ERA5 DATA PROCESSING")
    logger.info(f"# Period: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    logger.info(f"{'#' * 80}\n")

    # Generate list of (year, month) tuples
    months_to_process = []
    current_year = start_year
    current_month = start_month

    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        months_to_process.append((current_year, current_month))
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    logger.info(f"Total months to process: {len(months_to_process)}\n")

    # Process each month
    successful = 0
    failed = 0
    skipped = 0

    for year, month in months_to_process:
        result = process_monthly_data(year, month, skip_existing=skip_existing)
        if result == "skipped":
            skipped += 1
        elif result:
            successful += 1
        else:
            failed += 1

    # Final summary
    logger.info(f"\n{'#' * 80}")
    logger.info(f"# PROCESSING COMPLETE")
    logger.info(f"# Successful: {successful}")
    logger.info(f"# Skipped: {skipped}")
    logger.info(f"# Failed: {failed}")
    logger.info(f"{'#' * 80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Process a single month
    # process_monthly_data(year=1950, month=1)

    # Process a range of months
    process_date_range(
        start_year=1950,
        start_month=1,
        end_year=2023,
        end_month=12,
        skip_existing=True
    )