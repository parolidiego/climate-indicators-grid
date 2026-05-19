# ---------------------------------------------------------------------------
# PACKAGES
# ---------------------------------------------------------------------------

library(terra)
library(ncdf4)
library(lubridate)
library(SPEI)
library(here)



# ---------------------------------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------------------------------

project_root <- here()

input_dir <- file.path(project_root, "data/out/monthly")
tmp_dir   <- file.path(project_root, "data/out/tmp_drought")
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

YEAR_START <- 1950
YEAR_END <- 2023

# Accumulation windows (in months) for which SPI and SPEI are computed.
# Shorter scales capture fast-onset agricultural drought; longer scales reflect slower hydrological / groundwater drought.
SCALES <- c(3, 6, 12)

# Calibration period: the distribution (Gamma for SPI, log-Logistic for SPEI) is fitted on data from these years only.
# All years are then scored relative to this baseline, so the index expresses anomalies with respect to a fixed reference
# rather than the full record.
CAL_START <- c(1965, 1)
CAL_END <- c(1994, 12)

# Representative day for the middle of each calendar month (Jan=15, Feb=46, …).
MID_MONTH_J <- c(15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349)

# Solar constant
Gsc <- 0.0820

N_CORES <- floor(parallel::detectCores() * 0.9) - 1



# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

cat("Loading ERA5 monthly files...\n")

# One file per year: monthly_vars_YYYY.nc, each with 12 layers (one per month).
# Stacking all years gives nlyr = n_years * 12 for each variable.
files <- sort(list.files(input_dir, pattern = "monthly_vars_[0-9]{4}\\.nc$",
                         full.names = TRUE))
if (length(files) == 0) stop("No monthly_vars_YYYY.nc files found in ", input_dir)

# Read via ncdf4 then assign extent and CRS from the coordinate variables in the file.
load_var <- function(files, varname) {
  layers <- lapply(files, function(f) {
    nc  <- ncdf4::nc_open(f)
    lon <- ncdf4::ncvar_get(nc, "longitude")
    lat <- ncdf4::ncvar_get(nc, "latitude")
    arr <- ncdf4::ncvar_get(nc, varname)
    ncdf4::nc_close(nc)
    arr <- aperm(arr, c(2, 1, 3)) # reorder from (lat, lon, time) to (lon, lat, time) for terra
    r   <- terra::rast(arr) # convert to terra raster
    res_lon <- abs(lon[2] - lon[1])
    res_lat <- abs(lat[2] - lat[1])
    terra::ext(r) <- c(min(lon) - res_lon/2, max(lon) + res_lon/2,
                       min(lat) - res_lat/2, max(lat) + res_lat/2)
    terra::crs(r) <- "EPSG:4326"
    r
  })
  terra::rast(layers)
}

p_stack  <- load_var(files, "P_jm")
tn_stack <- load_var(files, "TN_m")
tx_stack <- load_var(files, "TX_m")

# Builds a end-of-month time axis vector, matching the valid_time convention in the monthly NetCDF files.
full_dates <- ceiling_date(
  seq(as.Date(paste0(YEAR_START, "-01-01")),
      as.Date(paste0(YEAR_END,   "-12-01")),
      by = "month"),
  unit = "month"
) - 1
# Assigns this time dimension to the raster stacks
time(p_stack) <- full_dates
time(tn_stack) <- full_dates
time(tx_stack) <- full_dates

cat(sprintf("Loaded %d months (%d-%d), grid: %d rows x %d cols\n",
            length(full_dates), YEAR_START, YEAR_END,
            nrow(p_stack), ncol(p_stack)))





# ===========================================================================
# HELPER: COMPUTE PET (HARGREAVES MONTHLY)
# ===========================================================================

# Input:  TN_m [Grid, Month] - monthly mean of daily minimum 2-metre temperature [degC]
#         TX_m [Grid, Month] - monthly mean of daily maximum 2-metre temperature [degC]
# Output: PET  [Grid, Month] - monthly Hargreaves potential evapotranspiration [mm/month]
#
# Calculates PET using Hargreaves formula
# Three steps:
#   1. Ra stack  — extraterrestrial radiation [mm/day] computed per time step from latitude
#                  and day of year. Written to disk immediately to free RAM.
#   2. PET daily — Hargreaves formula applied pixel-by-pixel over the full time dimension:
#                  PET [mm/day] = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)
#                  TN, TX, and Ra are concatenated into one stack so app() can access all
#                  three variables at once as v[1:n], v[(n+1):(2n)], v[(2n+1):(3n)].
#   3. PET monthly — daily PET multiplied by the actual number of days in each month
#                    to give mm/month; uses days_in_month() so leap-year February is correct.
compute_pet <- function(tn, tx, tmp_prefix) {

  # Full Date vector for each time step
  dates      <- time(tx)
  # Calendar month index (1–12) for each time step, used to look up MID_MONTH_J
  months_vec <- month(dates)
  # Number of days in each month used to scale from daily to monthly PET
  days_vec <- as.numeric(days_in_month(dates))
  # Total number of time steps; used inside app() to split the concatenated vector v
  nlyr_each <- nlyr(tx)
  # Raster where each cell holds its latitude in radians, Ra will be written onto this raster
  lat_rad <- init(tx[[1]], "y") * pi / 180

  # Step 1: Ra stack
  cat("  Building Ra stack (extraterrestrial radiation)...\n")

  # For each time step, compute a spatial raster of Ra [mm/day] using latitude and day of year
  Ra_list <- lapply(seq_along(dates), function(i) {
    J       <- MID_MONTH_J[months_vec[i]]
    dr      <- 1 + 0.033 * cos(2 * pi * J / 365)
    sol_dec <- 0.409 * sin(2 * pi * J / 365 - 1.39)
    ws      <- acos(clamp(-tan(lat_rad) * tan(sol_dec), lower = -1, upper = 1))
    Ra      <- (24*60/pi) * Gsc * dr *
                 (ws * sin(lat_rad) * sin(sol_dec) +
                  cos(lat_rad) * cos(sol_dec) * sin(ws))
    Ra / 2.45
  })
  # Stack the list of per-timestep rasters into a single multi-layer raster
  Ra_stack <- terra::rast(Ra_list)
  # Assign the correct time dimension
  time(Ra_stack) <- dates
  # Write to disk and reload as file-backed raster to free RAM before app()
  ra_tmp <- paste0(tmp_prefix, "_Ra.tif")
  writeRaster(Ra_stack, ra_tmp, overwrite = TRUE)
  rm(Ra_stack, Ra_list); gc()
  Ra_stack <- terra::rast(ra_tmp)

  # Step 2: daily PET
  cat("  Computing PET [mm/day]...\n")
  # Get the number of months
  n <- nlyr_each
  # Concatenate TN, TX, Ra into one stack: v[1:n]=TN, v[(n+1):(2n)]=TX, v[(2n+1):(3n)]=Ra
  stack_all <- c(tn, tx, Ra_stack)
  # Build the file path where to store PET data
  pet_day_tmp <- paste0(tmp_prefix, "_PET_day.tif")

  # Apply Hargreaves formula to calculate PET along the time dimension, cell by cell
  PET_day <- app(
    stack_all,
    fun = function(v) {
      tmin  <- v[1:n]
      tmax  <- v[(n+1):(2*n)]
      Ra    <- v[(2*n+1):(3*n)]
      tmean <- (tmin + tmax) / 2
      0.0023 * Ra * (tmean + 17.8) * sqrt(pmax(tmax - tmin, 0)) # temperature range clipped at 0 to avoid sqrt of negative values
    },
    filename  = pet_day_tmp,
    overwrite = TRUE
  )
  rm(Ra_stack, stack_all); gc()
  file.remove(ra_tmp)

  # Step 3: monthly totals
  cat("  Scaling to mm/month...\n")
  # Building path of temp file where monthly data will be written
  pet_mm_tmp <- paste0(tmp_prefix, "_PET_mm.tif")
  # Each daily value gets multiple by the number of days in the month to get to the monthly PET value
  PET_mm <- app(
    PET_day,
    fun      = function(v) v * days_vec,
    filename = pet_mm_tmp,
    overwrite = TRUE
  )
  rm(PET_day); gc()
  file.remove(pet_day_tmp)

  # Format the raster in a proper way
  time(PET_mm)  <- dates
  names(PET_mm) <- format(dates, "%Y_%m")

  list(rast = PET_mm, tmp = pet_mm_tmp)
}



# ===========================================================================
# HELPER: COMPUTE SPI / SPEI AND SAVE TO TEMP TIF
# ===========================================================================

# Input:  input_stack [Grid, Month] - monthly precipitation [mm] for SPI, or
#                                     monthly water balance WB = P - PET [mm] for SPEI
# Output: index [Grid, Month] - standardised drought index (std units), saved to tmp_path
#
# Two steps:
#   1. Distribution fitting — for each cell, the full time series is wrapped as a ts object
#      and passed to SPEI::spi or SPEI::spei. The distribution (Gamma for SPI, log-Logistic
#      for SPEI) is fitted separately for each of the 12 calendar months, but only using data
#      from the calibration period (CAL_START-CAL_END). All years are then scored against
#      this fixed baseline, so the index expresses anomalies relative to that reference period.
#      Fitting uses ub-pwm (unbiased probability-weighted moments), the SPEI package default.
#   2. Output — the standardised index values are written directly to a temp .tif
#      via app(filename=), keeping the full multi-year stack on disk. Only small per-year
#      slices are loaded into RAM later during the NetCDF append step.
compute_index <- function(input_stack, sc, index_type, tmp_path) {
  cat(sprintf("  Computing %s-%d...\n", index_type, sc))

  # Ensures time consistency
  dates <- time(input_stack)
  # Capture config in local variables so they are accessible inside the app() closure
  .cal_start <- CAL_START
  .cal_end <- CAL_END
  .sc <- sc
  .type <- index_type
  .start <- YEAR_START

  # Apply distribution fitting and probability transform cell by cell along the time dimension
  idx_r <- app(input_stack, fun = function(x) {
    # Skip ocean / no-data cells
    if (all(is.na(x))) return(rep(NA_real_, length(x)))
    tryCatch({
      # Wrap as monthly ts so the SPEI package knows values 1,13,25,… are Januaries, 2,14,26,… are Februaries, etc.
      ts_obj <- ts(x, start = c(.start, 1), frequency = 12)
      # Set the correct function to be called depending on index type
      fn <- if (.type == "SPEI") SPEI::spei else SPEI::spi

      # Fit the distribution on the calibration period only, then score all data points against it;
      # Returns a list with $fitted holding the standardised index values for the full series
      res <- fn(
        ts_obj,
        scale        = .sc,
        ref.start    = .cal_start,
        ref.end      = .cal_end,
        distribution = if (.type == "SPEI") "log-Logistic" else "Gamma",
        fit          = "ub-pwm",
        na.rm        = TRUE,
        verbose      = FALSE
      )
      out <- as.numeric(res$fitted)
      # Values of exactly 0 or 1 map to -Inf/+Inf, so must be set to NA manually
      out[is.infinite(out) | is.nan(out)] <- NA
      out
    }, error = function(e) rep(NA_real_, length(x)))
  },
  cores     = N_CORES,
  filename  = tmp_path,
  overwrite = TRUE)

  # Ensuer time consistency
  time(idx_r) <- dates
  list(rast = idx_r, tmp = tmp_path)
}



# ===========================================================================
# COMPUTE PET, WATER BALANCE, AND ALL INDICES
# ===========================================================================

cat("Computing PET (Hargreaves)...\n")
# Compute monthly PET [mm/month] from TN and TX using the Hargreaves formula
pet_result <- compute_pet(tn_stack, tx_stack,
                          tmp_prefix = file.path(tmp_dir, "era5"))
# Extract the raster from the result list
pet <- pet_result$rast
# TN and TX are no longer needed after PET is computed
rm(tn_stack, tx_stack)

cat("Converting precipitation from m to mm...\n")
rr_mm_tmp <- file.path(tmp_dir, "rr_mm.tif")
# ERA5 stores P in metres; multiply by 1000 to get mm required by SPI and WB
rr_mm <- p_stack * 1000
# Write to disk and reload as file-backed raster to free RAM
writeRaster(rr_mm, rr_mm_tmp, overwrite = TRUE)
rm(p_stack, rr_mm)
rr_mm <- terra::rast(rr_mm_tmp)
# writeRaster does not preserve time metadata; reassign manually
time(rr_mm) <- full_dates

cat("Computing water balance (P - PET)...\n")
wb_tmp <- file.path(tmp_dir, "wb.tif")
# WB = P - PET: the climatic water balance used as input to SPEI
wb <- rr_mm - pet
# Write to disk and reload as file-backed raster to free RAM
writeRaster(wb, wb_tmp, overwrite = TRUE)
# PET is no longer needed after WB is computed
rm(wb, pet)
wb <- terra::rast(wb_tmp)
# writeRaster does not preserve time metadata; reassign manually
time(wb) <- full_dates

cat(strrep("=", 60), "\n")
cat("Computing SPI and SPEI from ERA5 monthly data\n")
cat(strrep("=", 60), "\n")
# Named list holding one list(rast, tmp) per index key (e.g. "spi_3", "spei_12")
index_results <- list()
for (sc in SCALES) {
  for (type in c("SPI", "SPEI")) {
    # Build the key used as variable name in the output NetCDF (e.g. "spi_3")
    key <- paste0(tolower(type), "_", sc)
    # Build the temp .tif path
    tmp_path <- file.path(tmp_dir, paste0(key, ".tif"))
    # SPI is fitted on precipitation; SPEI is fitted on the climatic water balance
    input    <- if (type == "SPI") rr_mm else wb
    # Compute the drought index and store its results in the list
    index_results[[key]] <- compute_index(input, sc, type, tmp_path)
    cat(sprintf("  Done: %s\n", key))
  }
}



# ===========================================================================
# APPEND INDICES TO MONTHLY FILES
# ===========================================================================

# The monthly NetCDF files store variables as (valid_time, latitude, longitude)
# in CDL order. ncdf4 uses Fortran (reversed) dimension ordering, so the same
# variable is accessed as (longitude, latitude, valid_time) = (3600, 1801, 12) in R.
# terra's as.array() returns (latitude, longitude, valid_time) = (1801, 3600, 12),
# so aperm(arr, c(2, 1, 3)) is required to match ncdf4's expected layout before writing.

cat(strrep("=", 60), "\n")
cat("Appending indices to monthly files\n")
cat(strrep("=", 60), "\n")

for (yr in YEAR_START:YEAR_END) {
  nc_file <- file.path(input_dir, sprintf("monthly_vars_%d.nc", yr))
  # Indices into full_dates that correspond to this calendar year (12 layers)
  yr_idx  <- which(year(full_dates) == yr)

  nc <- ncdf4::nc_open(nc_file, write = TRUE)
  lon_dim  <- nc$dim[["longitude"]]
  lat_dim  <- nc$dim[["latitude"]]
  time_dim <- nc$dim[["valid_time"]]

  for (key in names(index_results)) {
    # Reconstruct the human-readable long name from the key (e.g. "spi_3" -> "SPI-3")
    parts <- strsplit(key, "_")[[1]]
    longname <- paste0(toupper(parts[1]), "-", parts[2])

    # Define the variable in the file if this is a fresh run for this year.
    # Dimension order list(lon, lat, time) maps to CDL var(time, lat, lon) because
    # ncdf4 reverses the list — matching the ordering of the existing P_jm, TN_m, TX_m.
    if (!(key %in% names(nc$var))) {
      new_var <- ncdf4::ncvar_def(
        name        = key,
        units       = "std",
        dim         = list(lon_dim, lat_dim, time_dim),
        missval     = NaN,
        longname    = longname,
        prec        = "float",
        compression = 4
      )
      nc <- ncdf4::ncvar_add(nc, new_var)
    }

    # Extract this year's 12 layers from the full stack (loads only those layers
    # from the temp tif, not the entire multi-year raster) and reorder axes for ncdf4.
    arr <- as.array(index_results[[key]]$rast[[yr_idx]])  # (lat, lon, 12)
    arr <- aperm(arr, c(2, 1, 3)) # (lon, lat, 12) for ncdf4
    ncdf4::ncvar_put(nc, key, arr)
  }

  ncdf4::nc_close(nc)
  cat(sprintf("  Updated: monthly_vars_%d.nc\n", yr))
}



# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------

# Remove the temp files completly
all_tmps <- c(
  pet_result$tmp,
  rr_mm_tmp,
  wb_tmp,
  sapply(index_results, `[[`, "tmp")
)
file.remove(all_tmps[file.exists(all_tmps)])

leftover <- list.files(tmp_dir, pattern = "\\.tif$", full.names = TRUE)
if (length(leftover) > 0) file.remove(leftover)

if (dir.exists(tmp_dir) && length(list.files(tmp_dir)) == 0) unlink(tmp_dir, recursive = FALSE)

cat("Everything done.\n")