#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
datamodel-codegen --input $SCRIPT_DIR/config_file.yaml --input-file-type yaml --class-name Config --output $SCRIPT_DIR/config_class.py