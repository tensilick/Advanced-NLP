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

model="RNN" #either RNN, RNN2, RNN3, RNTN, or DCNN

#missing 5 on purpose, already ran
wvecDimBatch=("5" "15" "25" "30" "35" "45")

for wvecDim in "${wvecDimBatch[@]}"
do
########################################################
# Probably a good idea to let items below here be
##################