# ERA5 Climate Indicators Pipeline

This repository contains an end-to-end pipeline that:

1) Downloads hourly temperature and accumulated precipitation data from ERA5-Land reanalysis (globally, at 0.1° x 0.1° resolution, from 1950 to 2023) through the Copernicus Climate Data Store (CDS) API 

2) Process it into harmonized daily observations (Tmax, Tmin, Tmean, total precipitation) for each grid cell 

3) Computes 350+ yearly climate indicators at the grid cell level (0.1° x 0.1°) covering temperature, heatwaves, coldwaves, precipitation statistics and droughts

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
│   ├── 3_CLIMATE_INDICATORS/    # Steps 3–8 – compute climate indicators (run in the order of the file prefixes)
│   │   ├── 1.percentiles.py     
│   │   ├── 2.daily.py           
│   │   ├── 3.monthly.py         
│   │   ├── 4.droughts_monthly.R # Monthly SPI / SPEI indices (R)
│   │   ├── 5.yearly.py          # Main script: all yearly indicators
│   │   └── 6.droughts_yearly.py # Annual drought severity / duration
│   └── MISCELLANEOUS/
│       ├── check_yearly.ipynb              # Notebook visualizing yearly indicators output
│       ├── plot_temp_percentiles.ipynb     # Notebook visualizing the percentile thresholds
│       └── temp_precip_tx_distributions.py # Distribution plots of selected indicators
|
├── data/
│   ├── tmp/output_api/          # Stores raw API downloads
|   ├── in/era5_daily/           # Stores aggregated daily data for Tmax, Tmin, Tmean and Totprec
│   └── out/                     # Stores all climate indicators calculated splitted into subfolders by time frequency
│       ├── percentiles/         
│       ├── daily/               
│       ├── monthly/             
│       ├── yearly/              
│       └── tmp_drought/         # Transient scratch space for Steps 6 and 8, deleted when they finish
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

The environment contains both the Python and the R stack, so the R drought step needs no separate setup.

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

**Important:** Steps 7 and 8 both write to `data/out/yearly/`, but they do it differently. Step 7 rewrites each yearly file from scratch, while Step 8 opens the existing file and merges its variables into it. Every rerun of Step 7 therefore erases the drought variables, and **Step 8 must always be rerun after Step 7**.

---

### Step 1 — Download ERA5 data from CDS

**Scripts:** `code/1_API_ERA5/api_calls_prec.py` and `api_calls_temp.py`

Downloads raw ERA5-Land Hourly data (1950–2023) with a 0.1° x 0.1° resolution from the CDS API, one file per year-month.

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

Computes percentile thresholds over the 1965–1994 reference period (can be changed as you wish). These thresholds are used by the subsequent scripts to classify extreme events. Three files are produced:

- **Temperature, day-of-year percentiles** — computed on a centred rolling window (5-day and 15-day) around each calendar day, so the threshold varies through the year: TN10p, TX10p, TN90p, TX90p → `data/out/percentiles/temp_percentiles.nc`
- **Temperature, whole-year percentiles** — a single threshold per grid cell computed over the whole reference period, with no day-of-year structure: TX90, TX95, TX97_5, TX99, TN1, TN2_5, TN5, TN10 → `data/out/percentiles/temp_percentiles_fullyear.nc`
- **Precipitation** (wet days ≥ 1 mm): PW_95p, PW_99p → `data/out/percentiles/precip_percentiles.nc`

The two temperature files are what distinguishes the day-of-year indicator families from the whole-year (`_wy`) families in Step 7.

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

`P_jm` feeds the yearly precipitation indicators; all three feed the drought indices in the next step.

---

### Step 6 — Compute monthly drought indices (R)

**Script:** `code/3_CLIMATE_INDICATORS/4.droughts_monthly.R`

The only R step in the pipeline. Computes standardized drought indices from the monthly files and **appends them in place** to `monthly_vars_{YEAR}.nc`:

- **SPI** (Standardized Precipitation Index) — from `P_jm`, fitted with a Gamma distribution.
- **SPEI** (Standardized Precipitation-Evapotranspiration Index) — from the water balance `P_jm − PET`, fitted with a log-Logistic distribution. PET is computed with the **Hargreaves** formula from `TX_m`, `TN_m` and extraterrestrial radiation derived from latitude and day of year.

Both are computed at 3, 6 and 12-month accumulation scales (`spi_3`, `spi_6`, `spi_12`, `spei_3`, `spei_6`, `spei_12`), calibrated on 1965–1994 so the indices express anomalies relative to a fixed baseline. Shorter scales capture fast-onset agricultural drought, longer scales slower hydrological drought.

---

### Step 7 — Compute yearly indicators

**Script:** `code/3_CLIMATE_INDICATORS/5.yearly.py`

The main output of the pipeline. Computes ~320 yearly climate indicators per grid cell and saves them as `yearly_vars_{YEAR}.nc` in `data/out/yearly/`. Rather than listing every variable, the families are described below; the exact variable names, thresholds and definitions are documented in the docstring of each `calculate_*` function in the script.

**Temperature level and spread**
`TM`, `TX`, `TN` (annual means), `TNN` / `TXX` (coldest night / hottest day), `TVAR` (temperature variance), `DTR` (diurnal temperature range).

**Cold/warm nights and days** — counts of days beyond a day-of-year percentile threshold
`CN10`, `CD10`, `WN90`, `WD90`.

**Heatwaves and coldwaves** — runs of ≥ 3 consecutive days beyond a day-of-year percentile threshold, computed separately for daytime (Tmax) and nighttime (Tmin). Each of the four combinations yields the same four metrics: total days in events, length of the longest event, number of events, and cumulative intensity — e.g. `DDHW`, `LDHW`, `NDHW`, `TDHW` for day heatwaves, with the corresponding `DNHW` / `LNHW` / `NNHW` / `TNHW` for night heatwaves and the `*CW` equivalents for coldwaves.

**Spell durations** — runs of ≥ 6 consecutive days beyond a day-of-year percentile threshold
`CSD` (cold spell), `WSD` (warm spell).

**Absolute-threshold variants** (suffix `_XXC`, e.g. `WD90_35C`, `DDHW_30C`, `CSD_15C`)
The same cold/warm counts, heatwaves, coldwaves and spells as above, but requiring the day to cross **both** the percentile threshold **and** an absolute temperature threshold. This filters out events that are locally anomalous but not physically extreme (a "heatwave" at 12°C in a cold climate, for instance).

**Whole-year percentile variants** (suffix `_wy`)
The same families again, but thresholds come from the whole-year percentiles (`TX90`…`TX99`, `TN1`…`TN10`) instead of the day-of-year rolling percentiles — so an event is measured against the full annual distribution rather than against what is normal for that calendar day. Combined absolute + whole-year percentile versions also exist (e.g. `WSD95_20C_wy`, `TDHW99_30C_wy`).

**Fixed-threshold day counts**
Warm side: `TX30`, `TX35`, `TX40`. Cold side: `TN0`, `TNm5`, `TNm10` (on Tmin) and `TX10`, `TX5`, `TX0`, `TXm5`, `TXm10` (on Tmax).

**Temperature distribution bins** — day counts per temperature interval, left-inclusive and right-exclusive, in three overlapping resolutions of the same distribution: 3°C-wide (19 bins), 5°C-wide (14 bins) and 1°C-wide (52 bins), each with open-ended bins at both tails.

**Precipitation**
Levels and variability (`PA`, `PWT`, `W`, `PWA`, `PVAR`, `PWVAR`); wet-day extremes (`P95WT`, `P99WT`); consecutive dry/wet spells and their totals (`CDD`, `CWD`, `C95WD`, `C99WD`, `PCWD`, `PC95WD`, `PC99WD`); maxima and minima (`PX1`, `PX5`, `PXM`, `PNM`); and daily precipitation bins (`P_1_to_10`, `P_10_to_20`, `P_above_20`).

---

### Step 8 — Compute yearly drought indicators

**Script:** `code/3_CLIMATE_INDICATORS/6.droughts_yearly.py`

Reads the monthly SPI/SPEI indices from Step 6 and **merges 36 annual drought variables into the existing** `yearly_vars_{YEAR}.nc` files.

A drought event is a run of at least **3 consecutive months** with the index at or below a threshold, detected over the full multi-year series so that events crossing a December–January boundary count as one continuous event. Three severity tiers are used: normal (`≤ -1.0`, suffix `_n`), severe (`≤ -1.5`, suffix `_s`) and extreme (`≤ -2.0`, suffix `_x`).

For each of the 6 indices × 3 tiers, two variables are produced:

| Variable | Description |
|----------|-------------|
| `{index}_sev_{tier}` | Annual drought severity — sum of the absolute index values over qualifying drought months |
| `{index}_dur_{tier}` | Annual drought duration — count of qualifying drought months |

The script writes intermediate masked monthly files to `data/out/tmp_drought/` and skips any that already exist, so an interrupted run can be resumed cheaply. They are deleted when the script completes.

---

### Optional — Visualize output to check for soundness

**Notebook:** `code/MISCELLANEOUS/check_yearly.ipynb`

Loads all yearly output files, averages across years, and produces global and Europe-focused maps of every variable in the dataset (saved as PNGs alongside the notebook), plus comparison plots between the percentile-based, absolute-threshold and whole-year variants of the same families. Also prints a sanity-check table with global min / max / mean and NaN fraction per variable.

**Notebook:** `code/MISCELLANEOUS/plot_temp_percentiles.ipynb`

Maps the reference percentile thresholds produced in Step 3.

**Script:** `code/MISCELLANEOUS/temp_precip_tx_distributions.py`

Distribution plots of selected temperature and precipitation indicators from the regression-ready dataset.
