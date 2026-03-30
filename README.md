# ERA5 Climate Indicators Pipeline

This repository contains a end-to-end pipeline that 

1) Downloads hourly temperature and accumulated precipitation data from ERA5-Land reanalysis (globally, at 0.1° x 0.1° resolution, from 1950 to 2023) through the Copernicus Climate Data Store (CDS) API 

2) Process it into harmonized daily observations (Tmax, Tmin, Tmean, total precipitation) for each grid cell 

3) Computes 48+ yearly climate indicators at the grid cell level (0.1° x 0.1°) covering temperature, heatwaves, coldwaves, and precipitation statistics  

## Reference

The set of climate indicators calculated in this pipeline is inspired by:

> Akyapı, Berkay, Matthieu Bellon, and Emanuele Massetti. 2025. "Estimating Macrofiscal Effects of Climate Shocks from Billions of Geospatial Weather Observations." American Economic Journal: Macroeconomics 17 (3): 114–59. https://doi.org/10.1257/mac.20230042

Akyapı et al. construct a rich set of climate variables from daily ERA5 0.25° x 0.25° data trying to capture the center and tails of the temperature and precipitation distributions, including heatwaves, coldwaves, droughts, and intense precipitation events. They finally use LASSO to identify which variables best explain country-level GDP.

**Key difference:** In their paper, grid-cell level weather data is aggregated up to the **country level** to be matched against national macroeconomic data. This pipeline instead retains the full **0.1° × 0.1° grid-cell resolution**, computing a big subset of their indicators at each individual grid point. 

## Repository structure

```
era5/
|
├── code/
│   ├── 1_API_ERA5/              # Step 1 – download raw ERA5 temp. & precip. variables via CDS API
│   │   ├── api_calls_prec.py    
│   │   └── api_calls_temp.py    
│   ├── 2_AGGREGATING/           # Step 2 – join variables into harmonized files containg daily grid level data points
│   │   └── aggregating.py
│   ├── 3_CLIMATE_INDICATORS/    # Steps 3–6 – compute climate indicators
│   │   ├── 1.percentiles.py     
│   │   ├── 2.daily.py           
│   │   ├── 3.monthly.py         
│   │   └── 4.yearly.py          
│   └── MISCELLANEOUS/
│       └── check_yearly.ipynb   # Notebook visualizing yearly indicators output
│       └── test_notebook.ipynb  # Notebook used for debugging when writing the main scripts
|
├── data/
│   ├── tmp/output_api/          # Stores raw API downloads
|   ├── in/era5_daily/           # Stores aggregated daily data for Tmax, Tmin, Tmean and Totprec
│   └── out/                     # Stores all climate indicators calculated splitted into subfolders by time frequency
│       ├── percentiles/         
│       ├── daily/               
│       ├── monthly/             
│       └── yearly/              
|
└── era5.yml                     # Conda environment file
```

---

## Environment setup

### 1. Create the environment

```bash
conda env create -f era5.yml
```

### 2. Activate it

```bash
conda activate era5
```

---

## CDS API setup

The download scripts use the [Copernicus Climate Data Store API](https://cds.climate.copernicus.eu/how-to-api). You need to set it up once before running any download.

Instructions can be found in the [CDS API documentation](https://cds.climate.copernicus.eu/how-to-api), but here’s a quick summary for Windows users:

### 1. Create an account

Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) and confirm your email.

### 2. Get your API key

Log in, click your username in the top-right corner, and go to your profile page. Copy your `url` and `key` from the "API key" section in your [profile](https://cds.climate.copernicus.eu/profile).

### 3. Create the credentials file

Create a file called `.cdsapirc` in your home directory (ex. `C:\Users\<YourUsername>\.cdsapirc`) and add the following content, replacing the token with your own:

```
url: https://cds.climate.copernicus.eu/api
key: <YOUR-PERSONAL-ACCESS-TOKEN>
```

### 4. Accept the dataset Terms of Use

Before any download will work, you must manually accept the terms for each dataset on the CDS website. The dataset used in this project is the following:

- [ERA5-Land hourly data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)

Go to the `Download` tab, scroll down and accept the **"Terms of use"** while logged in.

---

## Running the pipeline

Run the scripts in the following order. Each step depends on the output of the previous one.

---

### Step 1 — Download ERA5 data from CDS

**Scripts:** `code/1_API_ERA5/api_calls_prec.py` and `api_calls_temp.py`

Downloads raw ERA5-Land Hourly data (1950–2024) with a 0.1° x 0.1° resolution from the CDS API, one file per year-month.

- `api_calls_prec.py` — downloads **total precipitation** at 00:00 UTC (which refers to total precipitation accumulated on day `d-1`). Output: one NetCDF per month in `data/tmp/output_api/total_precipitation` with one observation per day for each grid cell.
- `api_calls_temp.py` — downloads **2-metre temperature** (hourly), then converts from Kelvin to Celsius and resamples to daily Tmax / Tmin / Tmean. Output: one NetCDF per month in `data/tmp/output_api/2m_temperature` with 3 data points (TX, TN, TM) per day for each grid cell.

These two scripts are actually independent of each other and can be run in parallel.

---

### Step 2 — Merge temperature and precipitation into daily files

**Script:** `code/2_AGGREGATING/aggregating.py`

Merges the temperature and precipitation data produced in Step 1 into single combined NetCDF files (one file per month named `era5_daily_{YEAR}_{MONTH}.nc` containing daily grid level observations for `TX`, `TN`, `TM`, `tot_prec`). Also corrects the ERA5 precipitation time-shift (ERA5 precipitation at 00:00 UTC on day `d` represents the accumulated precipitation from 00.00 to 23.59 on day `d-1`, so timestamps are shifted back by one day). Output goes to `data/in/era5_daily/`, one file per month.

---

### Step 3 — Compute reference percentiles

**Script:** `code/3_CLIMATE_INDICATORS/1.percentiles.py`

Computes percentile thresholds over the 1965–1994 reference period (can be changed as you wish) using a centred rolling window around each calendar day-of-year (for temperature), or simplying calculating a percentile value for the whole period (for precipitation). These thresholds are used by the subsequent scripts to classify extreme events.

- **Temperature** (5-day and 15-day windows): TN10p, TX10p, TN90p, TX90p → `data/out/percentiles/temp_percentiles.nc`
- **Precipitation** (wet days ≥ 1 mm): PW_95p, PW_99p → `data/out/percentiles/precip_percentiles.nc`

---

### Step 4 — Compute daily variables

**Script:** `code/3_CLIMATE_INDICATORS/2.daily.py`

Produces monthly files of daily-resolution variables (`daily_vars_{YEAR}_{MONTH}.nc`) in `data/out/daily/`:

| Variable | Description |
|----------|-------------|
| `TM_d`   | Daily mean temperature (°C) |
| `TX_d`   | Daily maximum temperature (°C) |
| `TN_d`   | Daily minimum temperature (°C) |
| `P_d`    | Daily total precipitation (m) |
| `PW_d`   | Precipitation on wet days only (NaN on dry days) |
| `DTR_d`  | Diurnal temperature range (TX − TN) |

---

### Step 5 — Compute monthly indicators

**Script:** `code/3_CLIMATE_INDICATORS/3.monthly.py`

Aggregates daily variables to monthly level. Outputs one file per year (`monthly_vars_{YEAR}.nc`) in `data/out/monthly/`:

| Variable | Description |
|----------|-------------|
| `P_jm`   | Total monthly precipitation (sum) |
| `TX_m`   | Monthly mean of daily Tmax |
| `TN_m`   | Monthly mean of daily Tmin |

---

### Step 6 — Compute yearly indicators

**Script:** `code/3_CLIMATE_INDICATORS/4.yearly.py`

The main output of the pipeline. Computes 48 yearly climate indicators per grid cell and saves them as `yearly_vars_{YEAR}.nc` in `data/out/yearly/`. Indicators include:

**Temperature**
`TM`, `TX`, `TN`, `TNN`, `TXX`, `TVAR`, `DTR`

**Cold/warm day and night counts**
`CN10`, `CD10`, `WN90`, `WD90`

**Heatwaves** (≥ 3 consecutive days above the 90th percentile)
`DDHW`, `LDHW`, `NDHW`, `TDHW` (daytime) · `DNHW`, `LNHW`, `NNHW`, `TNHW` (nighttime)

**Coldwaves** (≥ 3 consecutive days below the 10th percentile)
`DDCW`, `LDCW`, `NDCW`, `TDCW` (daytime) · `DNCW`, `LNCW`, `NNCW`, `TNCW` (nighttime)

**Spell durations**
`CSD` (cold spell), `WSD` (warm spell)

**Precipitation**
`PA`, `PWT`, `W`, `PWA`, `PVAR`, `PWVAR`, `P95WT`, `P99WT`, `CDD`, `CWD`, `C95WD`, `C99WD`, `PCWD`, `PC95WD`, `PC99WD`, `PX1`, `PX5`, `PXM`, `PNM`

---

### Optional — Visualize yearly output to check for soundness

**Notebook:** `code/MISCELLANEOUS/check_yearly.ipynb`

Loads all yearly output files, averages across years, and produces two plots saved alongside the notebook:

- `yearly_overview.png` — all 48 variables on a global map (~1° resolution)
- `yearly_europe.png` — all 48 variables zoomed into Europe (~0.2° resolution)

Also prints a sanity-check table with global min / max / mean and NaN fraction per variable.