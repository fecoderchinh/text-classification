FROM python:3.9

# Maintainer info
LABEL maintainer="fecoder.chinh@gmail.com"

# Make working directories
WORKDIR  /tr9h_ml_classification

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY ./static/* ./tr9h_ml_classification/static
COPY . .

# Install application dependencies from the requirements file
RUN pip install -r setup.txt

EXPOSE 3000

# Run the python application
CMD ["python", "main.py"]