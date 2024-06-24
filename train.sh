#!/bin/bash
VAL_TEST_DATA=True
export VAL_TEST_DATA

ExpName=Final
export ExpName

WEIGHT_DECALY=0.0003
export WEIGHT_DECALY

epochs=80
export epochs

bash trainPEM07.sh;
bash trainPEM08.sh;
bash trainPEM03.sh;
bash trainPEM04.sh;
# /mistgpu/shutdown.sh;
