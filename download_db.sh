#!/bin/bash
set -e

MIMIC_ID="1FU66Em_VNYWWGLDbrMuKLdAsORZVR0d3"
EICU_ID="1_fsa36CeryFZqLDNXBMs11Zvaa8IgKbm"

echo "Downloading mimic_iv_star.sqlite..."
gdown "$MIMIC_ID" -O src/envs/mimic_iv_star/mimic_iv_star.sqlite

echo "Downloading eicu_star.sqlite..."
gdown "$EICU_ID" -O src/envs/eicu_star/eicu_star.sqlite

echo "Done."
