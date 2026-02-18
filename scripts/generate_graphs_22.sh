#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

echo "Generating graphs for dataset 22..."
# generate-graphs -d 22 -f lte.csv -k -g rsrp,rsrq
# generate-graphs -d 22 -f lte.csv -g pci
generate-graphs -d 22 -f lte.csv -t -g rsrp --time-col companion_abs_time -a --relative
generate-graphs -d 22 -f lte.csv -t -g rsrp --time-col companion_abs_time --relative