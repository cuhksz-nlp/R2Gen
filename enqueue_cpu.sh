#!/bin/sh
#SBATCH -p cpu
#SBATCH -c2
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmandal@techfak.uni-bielefeld.de
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH -o logs/stdout_%j_%t

srun $HOME/projects/R2Gen/run_iu_xray.sh

