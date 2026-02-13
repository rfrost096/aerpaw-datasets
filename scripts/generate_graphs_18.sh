#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

echo "Generating graphs for dataset 18..."
generate-graphs -d 18 -f  yaw45/inputf1_rsrp_with_header.csv,yaw315/inputf1_rsrp_with_header.csv -k -g "LTE RSRP"