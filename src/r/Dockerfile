FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install R packages first so Docker can cache this layer
COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic.R .

# The data is NOT copied into the image. Mount src/data to /app/data at run time.
ENV DATA_DIR=/app/data

CMD ["Rscript", "titanic.R"]
