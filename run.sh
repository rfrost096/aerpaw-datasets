

#!/bin/bash

if [ -f config.env ]; then
    set -a
    source config.env
    set +a
else
    echo "Error: config.env not found."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Processing logs at: $DATASET_22_HOME/$DATASET_22_LOGS"
python $SCRIPT_DIR/$KML_SUBMODULE/generate_kpi_kml.py --workarea "$DATASET_22_HOME/$DATASET_22_LOGS" --input-log "$DATASET_22_FILE" -o "$DATASET_22_HOME/$DATASET_22_KMLS/rsrp.kml"