#!/bin/bash
#SBATCH -J gen_yamls
#SBATCH -n 1 # Number of cores requested
#SBATCH -t 10000 # Runtime in minutes
#SBATCH -p murphy # Partition to submit to
#SBATCH --mem=500 # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o output/gen_yamls_%A.out # Standard out goes to this file
#SBATCH -e output/gen_yamls_%A.err # Standard err goes to this filehostname
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=isaacxia@college.harvard.edu
module load python/3.6.0-fasrc01
echo python $script $args
python $script $args
