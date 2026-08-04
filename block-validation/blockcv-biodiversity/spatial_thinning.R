library(sf)
library(terra)

shp_file <- "~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/Butterfly_Moth.shp"

pts <- st_read(shp_file, quiet = TRUE) |>
  st_transform(102022)  # Africa Albers Equal Area

# Create 50 km grid
grid <- st_make_grid(
  pts,
  cellsize = 150000,
  square = TRUE
)

grid_sf <- st_sf(id = 1:length(grid), geometry = grid)

# Assign points to grid cells
join <- st_join(pts, grid_sf)

# Keep one point per cell
thinned <- join |>
  group_by(id) |>
  slice(1) |>
  ungroup()

# Back to WGS84
thinned <- st_transform(thinned, 4326)

coords <- st_coordinates(thinned)

result <- data.frame(
  longitude = coords[,1],
  latitude = coords[,2]
)

write.csv(
  result,
  "~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/Butterfly_Moth_thinned_150km.csv",
  row.names = FALSE
)

cat("Original points:", nrow(pts), "\n")
cat("Thinned points:", nrow(result), "\n")



# //////////////////////////////////////////////////


library(dplyr)

# File paths
file1 <- "~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/Butterfly_Moth_thinned_50km.csv"

file2 <- "~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/Butterfly_Moth_combined_thinned_5km.csv"

# Read files
df1 <- read.csv(file1)
df2 <- read.csv(file2)

# Standardize column names
names(df1) <- tolower(names(df1))
names(df2) <- tolower(names(df2))

# Keep only longitude and latitude
df1 <- df1[, c("longitude", "latitude")]
df2 <- df2[, c("longitude", "latitude")]

# Combine and remove duplicate coordinates
combined <- bind_rows(df1, df2) %>%
  distinct(longitude, latitude, .keep_all = TRUE)

# Output file
output_file <- "~/Development/geospatial-analysis/Biodiversity-Project/block-validation/blockcv-biodiversity/data/Butterfly_Moth_combined_final.csv"

# Save
write.csv(combined, output_file, row.names = FALSE)

cat("File 1 records:", nrow(df1), "\n")
cat("File 2 records:", nrow(df2), "\n")
cat("Combined unique records:", nrow(combined), "\n")
cat("Saved to:", output_file, "\n")