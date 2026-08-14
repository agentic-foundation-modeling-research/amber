#!/bin/bash

# tile server URL (use default openstreetmap server)
OSM_TILE_SERVER_URL="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
# geocoding server URL (use default openstreetmap server)
OSM_GEOCODING_SERVER_URL="https://nominatim.openstreetmap.org/"
# routing server URLs (use default openstreetmap server)
OSM_ROUTING_SERVER_URL="https://routing.openstreetmap.de"
OSM_CAR_SUFFIX="/routed-car"
OSM_BIKE_SUFFIX="/routed-bike"
OSM_FOOT_SUFFIX="/routed-foot"
# original WebArena config (CMU server with different ports for each vehicule type)
# OSM_ROUTING_SERVER_URL="http://metis.lti.cs.cmu.edu"
# OSM_CAR_SUFFIX=":5000"
# OSM_BIKE_SUFFIX=":5001"
# OSM_FOOT_SUFFIX=":5002"

# stop if any error occur
set -e

DIR_NAME=$(dirname "$0")

tar -xvzf $DIR_NAME/../archive/openstreetmap-website.tar.gz -C $DIR_NAME/

# openstreetmap website set up
# override the docker compose file to work with custom port
cp $DIR_NAME/openstreetmap-templates/docker-compose.yml $DIR_NAME/openstreetmap-website/docker-compose.yml
# copy template files to be set up
cp $DIR_NAME/openstreetmap-templates/leaflet.osm.js $DIR_NAME/openstreetmap-website/vendor/assets/leaflet/leaflet.osm.js
cp $DIR_NAME/openstreetmap-templates/fossgis_osrm.js $DIR_NAME/openstreetmap-website/app/assets/javascripts/index/directions/fossgis_osrm.js

# sed works differently on Mac (BSD) and Linux (GNU),
# so we need to check the version of sed to determine the correct syntax for in-place editing
if [[ -z "$OSTYPE" ]]; then
  echo "Error: OSTYPE is not set. Please run this script in a proper shell environment."
  exit 1
fi
if [[ "$OSTYPE" == "darwin"* ]]; then
  SED_INPLACE=(-i '')  # MacOS
else
  SED_INPLACE=(-i)  # Linux
fi

# set up tile server URL
sed "${SED_INPLACE[@]}" "s|url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'|url: '${OSM_TILE_SERVER_URL}'|g" $DIR_NAME/openstreetmap-website/vendor/assets/leaflet/leaflet.osm.js
# set up geocoding server URL
sed "${SED_INPLACE[@]}" "s|nominatim_url:.*|nominatim_url: \"$OSM_GEOCODING_SERVER_URL\"|g" $DIR_NAME/openstreetmap-website/config/settings.yml
# set up routing server URLs
sed "${SED_INPLACE[@]}" "s|fossgis_osrm_url:.*|fossgis_osrm_url: \"$OSM_ROUTING_SERVER_URL\"|g" $DIR_NAME/openstreetmap-website/config/settings.yml
sed "${SED_INPLACE[@]}" "s|__OSMCarSuffix__|${OSM_CAR_SUFFIX}|g" $DIR_NAME/openstreetmap-website/app/assets/javascripts/index/directions/fossgis_osrm.js
sed "${SED_INPLACE[@]}" "s|__OSMBikeSuffix__|${OSM_BIKE_SUFFIX}|g" $DIR_NAME/openstreetmap-website/app/assets/javascripts/index/directions/fossgis_osrm.js
sed "${SED_INPLACE[@]}" "s|__OSMFootSuffix__|${OSM_FOOT_SUFFIX}|g" $DIR_NAME/openstreetmap-website/app/assets/javascripts/index/directions/fossgis_osrm.js
