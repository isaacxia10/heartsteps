#!/bin/bash
#SBATCH -J iter_exec
#SBATCH -n 1 # Number of cores requested
#SBATCH -t 10000 # Runtime in minutes
#SBATCH -p murphy # Partition to submit to
#SBATCH --mem=3500 # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o output/iter_exec_%A_%a.out # Standard out goes to this file
#SBATCH -e output/iter_exec_%A_%a.err # Standard err goes to this filehostname
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=isaacxia@college.harvard.edu
module load python/3.6.0-fasrc01
echo test_exec.py ${SLURM_ARRAY_TASK_MAX} ${SLURM_ARRAY_TASK_ID} $num_vals $tot_runs $yaml_loc $out_dir $fig_dir $data_dir hs_test_runner.py
python test_exec.py ${SLURM_ARRAY_TASK_MAX} ${SLURM_ARRAY_TASK_ID} $num_vals $tot_runs $yaml_loc $out_dir $fig_dir $data_dir hs_test_runner.py
# python hs_test_runner.py 0 results_train11112/ ~/HS_data/ yamls_train11112/
#echo "$ac$fc\_hi$pc"
#python hs_test_runner.py 0 results_test/ ~/HS_data/ yamls0000/ 0 0 0 0
# python hs_plot_runner.py 0
