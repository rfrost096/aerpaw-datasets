#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

echo "Analyzing features for dataset 22..."
analyze-features -d 22 -f lte.csv -g mutual --fields rsrp,rsrq,rssi,asu,pci,ta,ci,longitude,latitude,altitude