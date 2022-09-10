#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=30
step=1e-2

# for RNN2 only, otherwise doesnt matter
middleDim=30

model="