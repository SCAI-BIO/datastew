# Define the base image
FROM python:3.9

# Create a folder and copy the folder structure from local
RUN mkdir -p /index

COPY /index /index

# Install API requirements
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

EXPOSE 80

# API entry point
CMD ["uvicorn", "index.api.routes:app", "--host", "0.0.0.0", "--port", "80"]