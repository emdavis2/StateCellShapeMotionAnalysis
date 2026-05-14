#!/bin/bash

for i in 32 37 39 41 45 46 47 49 50 51 52 53 54 55 56 58 59 60
do
sbatch submit_driftcorrection.slurm $i
sleep 1
done
