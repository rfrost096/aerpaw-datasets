#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

graph-label \
    --flight-ids "Dataset_24_PawPrints_4G_30m_Flight_1" \
    --add-spherical \
