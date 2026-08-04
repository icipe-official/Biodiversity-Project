FROM rocker/geospatial:4.3.1

# Install system C/C++ dependencies for spatial libraries
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Install specific required R packages
RUN R -e "install.packages(c('optparse', 'blockCV', 'viridis', 'readr'), repos='https://cloud.r-project.org/')"

WORKDIR /app

# Copy project files to container container
COPY config.R main.R ./
COPY R/ ./R/
COPY src/ ./src/

ENTRYPOINT ["Rscript", "main.R"]