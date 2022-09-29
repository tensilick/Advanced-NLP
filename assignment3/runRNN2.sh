
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

model="RNN2" #either RNN, RNN2, RNN3, RNTN, or DCNN

#missing 5 on purpose, already ran
middleDimBatch=("5" "15" "25" "30" "35" "45")
wvecDim=30

for middleDim in "${middleDimBatch[@]}"