FROM python:3.10-slim

# Install system C/C++ dependencies for spatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files to container
COPY config.py main.py ./
COPY utils/ ./utils/
COPY src/ ./src/

ENTRYPOINT ["python", "main.py"]