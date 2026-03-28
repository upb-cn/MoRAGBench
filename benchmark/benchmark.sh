#!/bin/bash

mkdir -p ./results
mkdir -p ./completed_configs

for file in ./configs/*.json; do
    name="$(basename "$file" .json)"

    cd ../client
    python main.py --config ../benchmark/$file --output_path ../benchmark/results/$name
    cd -
    mv $file ./completed_configs
done