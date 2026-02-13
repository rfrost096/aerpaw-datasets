#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

echo "Generating graphs for dataset 22..."
generate-kml --workarea "$DATASET_22_HOME/Logs" --input-log "lte.csv" -o "$DATASET_22_HOME/KMLs/rsrp.kml"