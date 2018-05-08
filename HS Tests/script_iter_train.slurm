#!/bin/bash
#SBATCH -J iter_train
#SBATCH -n 1 # Number of cores requested
#SBATCH -t 10000 # Runtime in minutes
#SBATCH -p murphy # Partition to submit to
#SBATCH --mem=8000 # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o output/train_%A.out # Standard out goes to this file
#SBATCH -e output/train_%A.err # Standard err goes to this filehostname
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=isaacxia@college.harvard.edu
module load python/3.6.0-fasrc01
python iter_test.py $num_vals $yaml_loc $out_dir $fig_dir $result_dir
