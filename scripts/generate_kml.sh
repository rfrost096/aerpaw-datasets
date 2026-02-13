

#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

if [ -f "$SCRIPT_DIR/../config.env" ]; then
    set -a
    source "$SCRIPT_DIR/../config.env"
    set +a
else
    echo "Error: config.env not found."
    exit 1
fi

echo "Processing logs at: $DATASET_22_HOME/$DATASET_22_LOGS"
generate-kml --workarea "$DATASET_22_HOME/$DATASET_22_LOGS" --input-log "$DATASET_22_FILE" -o "$DATASET_22_HOME/$DATASET_22_KMLS/rsrp.kml"