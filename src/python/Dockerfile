FROM python:3.11-slim

WORKDIR /app

# Install pinned Python packages first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic.py .

# The data is NOT copied into the image. Mount src/data to /app/data at run time.
ENV DATA_DIR=/app/data

CMD ["python", "titanic.py"]
