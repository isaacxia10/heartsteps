#!/bin/bash
#SBATCH -J remove
#SBATCH -n 1 # Number of cores requested
#SBATCH -t 100000 # Runtime in minutes
#SBATCH -p murphy # Partition to submit to
#SBATCH --mem=3000 # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o remove_%A.out # Standard out goes to this file
#SBATCH -e remove_%A.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL
#SBATCH --mail-user=isaacxia@college.harvard.edu
module load python/3.6.0-fasrc01
rm -rf /n/murphy_lab/lab/isaac_bigtest/*01112/
#rm -rf /n/murphy_lab/lab/isaac_bigtest_std/*
#rm -rf /n/murphy_lab/lab/isaac_iter/
#rm -rf /n/murphy_lab/lab/isaac_test/
#rm -rf /n/murphy_lab/lab/isaac_results/
#python hs_test_runner.py 0 results_111/ ~/HS_data/ yamls/ 1 1 0
# python hs_plot_runner.py 0
