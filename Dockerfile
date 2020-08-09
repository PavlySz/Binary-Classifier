# FROM alpine:latest

# # RUN apk add python3 py3-pip \
# #     && pip3 install --upgrade pip
# RUN apk --no-cache --update-cache add gcc gfortran python3 python3-dev py3-pip build-base wget freetype-dev libpng-dev openblas-dev
# RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
# RUN pip3 install pandas

# WORKDIR /app

# COPY . /app

# RUN apk add g++ 
# RUN pip3 --no-cache-dir install -r requirements.txt

# EXPOSE 5000

# ENTRYPOINT [ "python3" ]
# CMD [ "flask_website.py" ]

### Dockerfile
FROM python:3.6.7-alpine3.6

# Set the working directory of the docker image
WORKDIR /app
COPY . /app

# Install native libraries, required for numpy
RUN apk --no-cache add musl-dev linux-headers g++

# Upgrade pip
RUN pip install --upgrade pip

# packages that we need
RUN pip install numpy && \
    pip install pandas && \
    pip install flask && \
    pip install scikit-learn

# export a Docker port
EXPOSE 5000

# Run the command when the Docker image is ran
ENTRYPOINT ["python3"]

CMD ["flask_website.py"]