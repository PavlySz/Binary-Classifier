# Using Ubuntu 18.0 image
FROM ubuntu:18.04

# Check for updates
RUN apt-get update && \
  apt-get install -y software-properties-common

# Get essential depenedencies for Python3
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv

# Update pip
RUN python3.6 -m pip install pip --upgrade

# Install required packages
RUN pip install --no-cache-dir numpy pandas matplotlib flask scipy scikit-learn

# Set the working directory of the docker image
WORKDIR /app
COPY . /app

# export a Docker port
EXPOSE 5000

# Run the command when the Docker image is ran
ENTRYPOINT ["python3"]

CMD ["flask_website.py"]
