import cdsapi
import os
from pathlib import Path
import logging
import time
from datetime import datetime

# Configure logging - only for this script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler - append mode
file_handler = logging.FileHandler('download_log_prec.log', mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Suppress cdsapi's verbose logging
logging.getLogger('cdsapi').setLevel(logging.WARNING)

# Add session separator to log file
logger.info("=" * 80)
logger.info(f"NEW SESSION STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 80)

var = "total_precipitation"
directory = str(Path(__file__).parent.parent.parent / "data" / "tmp" / "output_api" / var)

c = cdsapi.Client()

years = [
    '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957', '1958', '1959',
    '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
    '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979',
    '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',
    '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
    '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
    '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
    '2020', '2021', '2022', '2023', '2024'
]

months = [
    '01', '02', '03',
    '04', '05', '06',
    '07', '08', '09',
    '10', '11', '12'
]

os.makedirs(directory, exist_ok=True)
retries = 10
retry_delay = 5  # seconds to wait between retries

# Track failed downloads
failed_downloads = []
total_requests = len(years) * len(months)
completed_requests = 0

logger.info(f"Starting download of {total_requests} files")
logger.info(f"Variable: {var}")
logger.info(f"Output directory: {directory}")

for yr in years:
    for mn in months:
        file_id = f"{yr}-{mn}"
        output_file = f"{directory}/era5_{var}_{yr}_{mn}.nc"

        # Skip if file already exists
        if os.path.exists(output_file):
            logger.info(f"[{completed_requests + 1}/{total_requests}] Skipping {file_id} - file already exists")
            completed_requests += 1
            continue

        logger.info(f"[{completed_requests + 1}/{total_requests}] Processing {file_id}")

        for attempt in range(retries):
            try:
                dataset = "reanalysis-era5-land"
                request = {
                    "variable": var,
                    "year": yr,
                    "month": mn,
                    "day": [
                        "01", "02", "03",
                        "04", "05", "06",
                        "07", "08", "09",
                        "10", "11", "12",
                        "13", "14", "15",
                        "16", "17", "18",
                        "19", "20", "21",
                        "22", "23", "24",
                        "25", "26", "27",
                        "28", "29", "30",
                        "31"
                    ],
                    "time": [
                        "00:00"
                    ],
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                }

                c.retrieve(dataset, request).download(output_file)

                # Verify file was created and has content
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    logger.info(f"[OK] Successfully downloaded {file_id} (size: {os.path.getsize(output_file)} bytes)")
                    completed_requests += 1
                    break
                else:
                    raise Exception("Downloaded file is missing or empty")

            except KeyboardInterrupt:
                logger.warning("Download interrupted by user")
                raise

            except Exception as e:
                error_type = type(e).__name__
                logger.warning(f"[FAIL] Attempt {attempt + 1}/{retries} failed for {file_id}: {error_type} - {str(e)}")

                if attempt == retries - 1:
                    logger.error(f"[FATAL] {file_id} after {retries} attempts - {error_type}: {str(e)}")
                    failed_downloads.append({
                        'year': yr,
                        'month': mn,
                        'error': str(e),
                        'error_type': error_type
                    })
                    completed_requests += 1
                else:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

# Final summary
logger.info("=" * 60)
logger.info("DOWNLOAD SUMMARY")
logger.info("=" * 60)
logger.info(f"Total requests: {total_requests}")
logger.info(f"Successful: {total_requests - len(failed_downloads)}")
logger.info(f"Failed: {len(failed_downloads)}")

if failed_downloads:
    logger.error(f"\n{len(failed_downloads)} downloads failed:")
    for failure in failed_downloads:
        logger.error(f"  - {failure['year']}-{failure['month']}: {failure['error_type']}")

    # Write failed downloads to a file for easy retry
    retry_file = f"failed_downloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(retry_file, 'w') as f:
        f.write("# Failed downloads - Year,Month,Error\n")
        for failure in failed_downloads:
            f.write(f"{failure['year']},{failure['month']},{failure['error_type']}: {failure['error']}\n")
    logger.info(f"\nFailed downloads saved to: {retry_file}")
else:
    logger.info("All downloads completed successfully!")