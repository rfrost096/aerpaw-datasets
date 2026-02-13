#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

echo "Generating graphs for dataset 24..."
generate-graphs -d 24 -f pawprints_4G_LTE_altitude30m_flight1.csv,pawprints_4G_LTE_altitude50m.csv -k -g rsrp,rsrq
generate-graphs -d 24 -f pawprints_4G_LTE_altitude30m_flight1.csv,pawprints_4G_LTE_altitude50m.csv -g pci