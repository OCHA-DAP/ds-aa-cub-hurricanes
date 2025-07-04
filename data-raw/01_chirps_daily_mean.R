#' this code taken from `ds-adhoc-cuba`

library(rgee)
library(tidyverse)
library(tidyrgee)
library(targets)
library(here)
library(sf)
library(cumulus)


ee_Initialize(project = "ee-zackarno")


cat("loading Cuba boundary from FAO GAUL\n")
fc_gaul <- ee$FeatureCollection("FAO/GAUL/2015/level0")
fc_aoi <- fc_gaul$filter(ee$Filter$eq("ADM0_NAME", "Cuba"))


cat("reading and manipulate image collection\n")
ic <- ee$ImageCollection("UCSB-CHG/CHIRPS/DAILY")
tic <- as_tidyee(ic)


ic_aoi <- ic$filterBounds(fc_aoi)


yrs_unique <- tic$vrt$year |> unique()

# a little trick to get 3 years at a time downloading. More than this was 
# timing out, but less takes more time.

yr_grps <- split(
  yrs_unique,
  ceiling(seq_along(yrs_unique) / 3)
)

# Check blob for what's already been downloaded. If already present, 
# it will get skipped.
pc <- cumulus::blob_containers(
  stage= "dev"
)$projects

blobs_present <- AzureStor::list_blobs(
  pc,
  prefix = "ds-aa-cub-hurricanes/processed/chirps/"
)


yr_grps_remain <- yr_grps |> 
  discard(
    ~paste0("ds-aa-cub-hurricanes/processed/chirps/chirps_daily_", max(.x), ".csv") %in% blobs_present$name
  )


# started at 1:56 pm
df_rainfall_adm <- yr_grps_remain %>%
  map(
    \(yrs){
      cat("downloading data for year ", yrs, "\n")
      tic_temp <- tic %>%
        filter(
          year %in% yrs
        )

      df_tidy <- ee_extract_tidy(
        x = tic_temp,
        y = fc_aoi,
        scale = 5566,
        stat = "mean",
        via = "drive"
      )

      write_csv(
        x = df_tidy,
        file = file.path(
          "data",
          paste0("chirps_daily_", max(yrs), ".csv")
        )
      )
      
      cumulus::blob_write(
        df_tidy,
        container = "projects",
        name = paste0("ds-aa-cub-hurricanes/processed/chirps/chirps_daily_", max(yrs), ".csv")
        )
    }
  ) %>%
  list_rbind()


prefix_date <- format(Sys.Date(), "%Y%m%d")
cat("write zonal sats to csv\n")
cumulus::blob_write(
  x = df_rainfall_adm,
  container = "projects",
  name = paste0("ds-aa-cub-hurricanes/processed/chirps/",prefix_date,"_chirps_daily_historical_cuba.csv"),
  stage = "dev"

)
cat("finished")
