#!/usr/bin/env bash

set -ux

function main() {
    config_path="yamls/hydra-yamls"
    config_name="SDXL-base-256_common-canvas.yaml"

    command="composer run.py --config-path ${config_path} --config-name ${config_name}"

    # Retry the command until it succeeds
    while true; do
        $command
        status=$?
        
        if [ $status -eq 0 ]; then
            echo "Command executed successfully"
            break
        else
            echo "Command failed with status $status, retrying..."
        fi
    done
}

main
