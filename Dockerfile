# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/ 
COPY . /usr/src/app/
# Copy the current directory contents into the container at /usr/src/app


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

USER 10001
# Run script.py when the container launches
CMD ["python", "./script.py"]
