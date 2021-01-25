#!/usr/bin/env bash

DOCKER_NAME="objectnav_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

echo ${CUDA_VISIBLE_DEVICES}
docker run -v $(pwd)/habitat-challenge-data:/habitat-challenge-data -v /coc/dataset/habitat-sim-datasets/mp3d/:/coc/dataset/habitat-sim-datasets/mp3d/ -v $(pwd)/logs:/logs \
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2020.local.rgbd.yaml" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    ${DOCKER_NAME}\

