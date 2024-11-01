#!/bin/bash

# Define the image name and tag
IMAGE_NAME="localdev"
TAG="latest"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) --build-arg USERNAME=$USER  .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image '${IMAGE_NAME}:${TAG}' built successfully."
else
    echo "Failed to build Docker image '${IMAGE_NAME}:${TAG}'."
    exit 1
fi

