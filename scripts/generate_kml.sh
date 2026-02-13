#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

echo "Processing logs at: $DATASET_22_HOME/$DATASET_22_LOGS"
generate-kml --workarea "$DATASET_22_HOME/$DATASET_22_LOGS" --input-log "$DATASET_22_FILE" -o "$DATASET_22_HOME/$DATASET_22_KMLS/rsrp.kml"