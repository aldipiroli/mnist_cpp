#!/usr/bin/env bash

if [ -z ${TAG} ]; then
  echo "No tag given. Defaulting to latest"
  TAG="latest"
fi

IMAGE=localdev

DOCKER_ARGS=(
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -e DOCKER_MACHINE_NAME="${IMAGE}:${TAG}"
  --network=host
  --ulimit core=99999999999:99999999999
  --ulimit nofile=1024
  --privileged
  --rm
  -e DISPLAY=$DISPLAY
  -e QT_X11_NO_MITSHM=1
  --ipc=host
)

DOCKER_ARGS+=(
  -v /home/$USER:/home/$USER)

docker run --gpus all ${DOCKER_ARGS[@]} -it ${IMAGE}:${TAG}
