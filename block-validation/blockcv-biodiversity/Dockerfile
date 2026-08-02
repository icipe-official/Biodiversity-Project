FROM rocker/geospatial:4.4.1
RUN R -e "install.packages(c( \
    'blockCV', \
    'caret', \
    'ecospat', \
    'gbm', \
    'kernlab', \
    'mgcv', \
    'ncf', \
    'patchwork', \
    'pROC', \
    'PRROC', \
    'randomForest', \
    'rnaturalearth', \
    'rnaturalearthdata', \
    'sf', \
    'spdep', \
    'terra', \
    'tidyverse', \
    'tmap', \
    'viridis', \
    'xgboost' \
    ), repos='https://cloud.r-project.org')"

WORKDIR /workspace

COPY blockCV_validation.R .

CMD ["Rscript", "blockCV_validation.R"]