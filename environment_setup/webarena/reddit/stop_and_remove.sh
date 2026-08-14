#!/bin/bash

CONTAINER_NAME="${1:-forum}"

# Stop and remove existing containers
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
