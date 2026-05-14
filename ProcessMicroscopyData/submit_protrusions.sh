#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem=50g
#SBATCH -p general
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emae@med.unc.edu
#SBATCH -o %j.out
#SBATCH -e err.%j

python ./GetProtrusions.py