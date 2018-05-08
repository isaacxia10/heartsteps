# HeartSteps v2: Bandit Testing Platform

Welcome!  These files should get you ready to run the testing platform

## Getting started
0. Ensure you have HS data  in a directory on the server.
1. Generate parameter files using `script_gen_yamls.slurm`.  This script calls `gen_yamls.py`, which creates the yaml directory and places in a custom `master.yaml` file inside.  You can find an example of such a file as `master.yaml`.  
2. Initiate a test.  A test consists of a *master* job and an array of *worker* jobs.  
	The overall structure of a test of is a tuning parameter sweep of the Bandit algorithm, optimizing over one parameter at a time; the *worker* files run the actual test simulations, while the *master* job collects the actual results and optimizes over them.
	1. The *master* job is initiated by *script_iter_train.slurm*, which calls `iter_test.py`.
	Specifically, it creates parameter files from the `master.yaml` template, which the *worker* jobs use to run tests.  Once a round of a parameter sweep has been conducted, it collects the results, chooses the optimal value of the parameter, and starts a new round.
	At the end, it two simulations itself; one for the optimal set of tuning parameters from the training batch, and one using the same optimal set on the testing batch.
	2.  The array of *worker* jobs is initiated an array job batch using `script_iter_exec.slurm`; each job spawns its own instance of `test_exec.py`.  These jobs wait for a parameter file (e.g. `params_simNum0.yaml`); once it detects the one corresponding to its array id, it runs a simulation of the bandit algorithm by first calling `hs_test_runner.py`, and then creating plots on the results using `hs_plot_runner`.


`hs_test_funcs.py` are functions used by `hs_test_runner.py` for simulations, and `hs_plot_funcs.py` are functions used by `hs_plot_runner.py` to create Quality Metric plots.

## Example

We introduce how to run the algorithm for a specific variant.  In this case, we run the variant of the bandit with action centering, feedback controlling, probability clipping, and the Small bandit model, for split batch 0.

We will need these directories:
* HS v1 data directory (`data_dir`).  This should have your data files (`suggest-analysis-kristjan.csv` and `suggest-kristjan.csv`) before the testing.
* Parameter directory (`yaml_loc`).  This will store your parameters, as well as the data of results and plots for all intermediate optimal parameters.  This need not exist before running step 1. above, and should be free from non-master yaml files before, as `test_exec.py` depends on files in this directory to know when to proceed.
* Result output directory (`result_dir`).  This will contain the results and plots from the training simulation and testing simulation using optimal tuning parameters.
* Result data output directory (`out_dir`).  This stores all data for results, and will be fairly big; I put this on the lab storage.  This should be empty before the test.
* Plot output directory (`fig_dir`).  This stores all quality metric figures, and will be somewhat big.  I put this on the lab directory.  This *must* be empty before the test, as `iter_test.py` relies on files in this directory to know when to proceed.


Example run:

1. Generate yaml files `sbatch --export=script=gen_yamls.py,args="train 1 1 1 1 0" script_gen_yamls.slurm
`
	* We run the script `gen_yamls.py`, with argument that it is a `train` batch, `<ac_flag> <fc_flag> <pc_flag> <small_flag> <batch_num>`.  The flags are `1` for True and `0` for False, and `batch_num` should be one of 0,1,2 if we perform `k = 3` fold splitting.  
2. Run testing files.
	1. Master: `sbatch --dependency=afterok:43470497 --export=num_vals=160,yaml_loc="yamls_train11110/",out_dir="/n/murphy_lab/lab/isaac_bigtest/results_train11110/",fig_dir="/n/murphy_lab/lab/isaac_bigtest/resultsfig_train11110/",result_dir="results_train11110/" script_iter_train.slurm`
		* An example of job dependency; in this case, this job wait for job `43470497` to finish with an ok signal to start.
		* We use `num_vals = 160` to indicate we will have 160 values, of each parameter to be run; the range is set in `iter_test.py`, and the values are evenly spaced out.
		* `yaml_loc`,`result_dir` are all directories within the folder where this script is being run.
		* `out_dir`,`fig_dir` are directories in the lab storage.
		* Note that the suffix of `_train11110/` are used in directory folder names, but are not strictly necessary.
	2. Worker:`sbatch --dependency=afterok:43470497 --array=0-79 --export=num_vals=160,tot_runs=24,yaml_loc="yamls_train11110/",out_dir="/n/murphy_lab/lab/isaac_bigtest/results_train11110/",fig_dir="/n/murphy_lab/lab/isaac_bigtest/resultsfig_train11110/",data_loc="~/HS_data" script_iter_exec.slurm`
		* We use array ids from 0 to 79, inclusive, for 80 *worker* jobs.
		* `tot_runs` is set to 24, as there are 6 total tuning parameters, and within `iter_train.py`, we set there to be 4 run-throughs of the sweep.  Thus, this *worker* job will execute its loop 24 times before terminating.