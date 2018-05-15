import os
import sys
import datetime
import numpy as np
import yaml
import pandas as pd
import os.path
import time
import errno

# python iter_test.py num_vals yaml_loc out_dir
num_vals = int(sys.argv[1])
yaml_loc = sys.argv[2]
out_dir = sys.argv[3]
fig_dir = sys.argv[4]
result_dir = sys.argv[5]

# num_vals = 10
# yaml_loc = "yamls_train11102/" #train ac_flag fc_flag pc_flag small_flag batch_num
# out_dir = "results_train11102/"
# fig_dir = "resultsfig_train11102/"

if not os.path.exists(out_dir):
  try:
    os.makedirs(out_dir,exist_ok=True)
  except OSError as exc:
    if exc.errno == erno.EEXIST and os.path.isdir(out_dir):
      pass
    else:
      raise

if not os.path.exists(result_dir):
  try:
    os.makedirs(result_dir,exist_ok=True)
  except OSError as exc:
    if exc.errno == erno.EEXIST and os.path.isdir(result_dir):
      pass
    else:
      raise


min_param_vals = {"N_c_mult": 0.05,
              "T_c":3,
              "sig2_mult":0.10,
              "gamma":0.,
              "lamb":0.1,
              "prior_cov_mult":0.10
}

max_param_vals = {"N_c_mult": 0.9,
              "T_c":70,
              "sig2_mult":2.5,
              "gamma":1.,
              "lamb":10.,
              "prior_cov_mult":3.
}

num_run_through = 4


params = ["N_c_mult", "T_c", "sig2_mult","gamma","lamb","prior_cov_mult"]

master_param_val_yaml = yaml_loc + "master.yaml"


with open(master_param_val_yaml) as f:
    master_param_vals =  yaml.load(f)


sum_stats = ["Mean","Std","Min","25%","Median","75%","Max"]

sim_param_vals = pd.DataFrame(np.empty((num_vals * len(params) * num_run_through+2, len(params)+len(sum_stats))), columns=params + sum_stats)
sim_param_vals.loc[:,:] = np.nan

# Start counter for sim_num
sim_num = 0

for run_through in range(num_run_through):
    for param in params:
        print("Started " + str(run_through) + " " + param + " " + str(datetime.datetime.now()), flush = True)
        
        start_sim_num = sim_num

        param_space = np.linspace(min_param_vals[param], max_param_vals[param],num=num_vals)
        if param == "T_c":
            param_space = param_space.round()
        for param_val,val_num in zip(param_space,range(num_vals)):
            param_vals = master_param_vals.copy()
            param_vals[param] = param_val.tolist()
            param_vals["sim_num"] = sim_num

            ## Write param_vals to yaml ##
            with open(yaml_loc + "params" + str(val_num) + ".yaml","w") as f:
                yaml.dump(param_vals, f)
            
            # Save sim_params
            sim_param_vals.iloc[sim_num,:len(params)] = [param_vals[param] for param in params]
            sim_num += 1

        
    
        # Update master_param_vals based on optimum #
        for param_val,sim_num_done in zip(param_space, range(sim_num-num_vals,sim_num)):
            regret_file = out_dir + "regret_simNum" + str(sim_num_done) + ".npy"
            QM_file = fig_dir + "QM16_simNum" + str(sim_num_done) + ".png"
            
            # WAITING CODE ##
            wait_counter = 0
            while not os.path.exists(QM_file):
                time.sleep(30)
                if wait_counter == 1 or wait_counter >= 20:
                  print("Waiting for " + str(QM_file) + ", " + str(datetime.datetime.now()), flush = True)
                if wait_counter == 1:
                	# If waited for longer than a minute, save sim_param_vals.csv so user can recreate .yaml if worker script still working
                  print(sim_param_vals.loc[sim_num_done])
                  sim_param_vals.to_csv(yaml_loc + "sim_param_vals.csv")
                wait_counter += 1
                
                
            if not os.path.isfile(regret_file):
                raise ValueError("%s isn't a file!" % regret_file)

            regret = np.load(regret_file)
            sim_param_vals.loc[sim_num_done,"Mean"] = np.nanmean(regret,axis=1).mean()
            sim_param_vals.loc[sim_num_done,"Std"] = np.nanmean(regret,axis=1).std()
            sim_param_vals.loc[sim_num_done,"Min"] = np.percentile(np.nanmean(regret,axis=1), 0)
            sim_param_vals.loc[sim_num_done,"5%"] = np.percentile(np.nanmean(regret,axis=1), 5)
            sim_param_vals.loc[sim_num_done,"25%"] = np.percentile(np.nanmean(regret,axis=1), 25)
            sim_param_vals.loc[sim_num_done,"Median"] = np.percentile(np.nanmean(regret,axis=1), 50)
            sim_param_vals.loc[sim_num_done,"75%"] = np.percentile(np.nanmean(regret,axis=1), 75)
            sim_param_vals.loc[sim_num_done,"95%"] = np.percentile(np.nanmean(regret,axis=1), 95)
            sim_param_vals.loc[sim_num_done,"Max"] = np.percentile(np.nanmean(regret,axis=1), 100)

        sim_num_min = sim_param_vals.loc[start_sim_num:sim_num,"Mean"].idxmin()
        master_param_vals[param] = sim_param_vals.loc[sim_num_min,param].tolist()
        print(param + " opt val: " + str(master_param_vals[param]) + "\n", flush = True)

        # Copy results of optimal param to yaml folder
        command = "cp " + out_dir + "*simNum" + str(sim_num_min) + ".npy " + yaml_loc
        os.system(command)
        command = "cp " + fig_dir + "*simNum" + str(sim_num_min) + ".png " + yaml_loc
        os.system(command)

        # If last run, save optimal value as penultimate 
        if (run_through == num_run_through-1) and (param == "prior_cov_mult"):
            sim_num += 1
            sim_param_vals.loc[sim_num] = sim_param_vals.loc[sim_num_min]

        sim_param_vals.to_csv(yaml_loc + "sim_param_vals.csv")
        
sim_num += 1

# Obtain optimal training simulation paramters

sim_num_min = sim_param_vals.iloc[-163:-3]["Mean"].idxmin()

master_param_vals["N_c_mult"] = sim_param_vals.loc[sim_num_min].tolist()[0]
master_param_vals["T_c"] = sim_param_vals.loc[sim_num_min].tolist()[1]
master_param_vals["sig2_mult"] = sim_param_vals.loc[sim_num_min].tolist()[2]
master_param_vals["gamma"] = sim_param_vals.loc[sim_num_min].tolist()[3]
master_param_vals["lamb"] = sim_param_vals.loc[sim_num_min].tolist()[4]
master_param_vals["prior_cov_mult"] = sim_param_vals.loc[sim_num_min].tolist()[5]


final_sim_num = 20180330
master_param_vals["sim_num"] = final_sim_num

with open(yaml_loc + "params" + str(final_sim_num) + ".yaml", "w") as f:
    yaml.dump(master_param_vals, f)

print(str(master_param_vals))

### SETUP AND RUN TEST ###

# Setup var and prior from best train
command = "python hs_test_runner.py " + str(final_sim_num) + " " + out_dir + " ~/HS_data/ " + yaml_loc + " final"
print(command)
os.system(command)

command = "python hs_plot_runner.py " + str(final_sim_num) + " " + fig_dir + " " + out_dir
os.system(command)

# Copy resulting files

os.system("cp " + out_dir + "*simNum" + str(final_sim_num) + ".npy " + yaml_loc)
os.system("cp " + fig_dir + "*simNum" + str(final_sim_num) + ".png " + yaml_loc)
os.system("cp " + out_dir + "*simNum" + str(final_sim_num) + ".npy " + result_dir)
os.system("cp " + fig_dir + "*simNum" + str(final_sim_num) + ".png " + result_dir)


# Read in optimal parameters

with open(yaml_loc + "params" + str(final_sim_num) + ".yaml") as f:
    test_param_vals =  yaml.load(f)

# Run Test on params

test_sim_num  = final_sim_num + 1
test_param_vals["sim_num"] = test_sim_num

with open(yaml_loc + "params" + str(test_sim_num) + ".yaml", "w") as f:
    yaml.dump(test_param_vals, f)

print("Started Test")

print(str(test_param_vals))

command = "python hs_test_runner.py " + str(test_sim_num) + " " + out_dir + " " + "~/HS_data/" + " " + yaml_loc
os.system(command)

# Plot for Test
command = "python hs_plot_runner.py " + str(test_sim_num) + " " + fig_dir + " " + out_dir
os.system(command)

os.system("cp " + out_dir + "*simNum" + str(test_sim_num) + ".npy " + yaml_loc)
os.system("cp " + fig_dir + "*simNum" + str(test_sim_num) + ".png " + yaml_loc)
os.system("cp " + out_dir + "*simNum" + str(test_sim_num) + ".npy " + result_dir)
os.system("cp " + fig_dir + "*simNum" + str(test_sim_num) + ".png " + result_dir)

regret_file = out_dir + "regret_simNum" + str(test_sim_num) + ".npy"

if not os.path.isfile(regret_file):
    raise ValueError("%s isn't a file!" % regret_file)

# Save test result as final number
sim_num += 1

regret = np.load(regret_file)
sim_param_vals.loc[sim_num,"Mean"] = np.nanmean(np.nanmean(regret,axis=1))
sim_param_vals.loc[sim_num,"Std"] = np.nanstd(np.nanmean(regret,axis=1))
sim_param_vals.loc[sim_num,"Min"] = np.nanpercentile(np.nanmean(regret,axis=1), 0)
sim_param_vals.loc[sim_num,"5%"] = np.nanpercentile(np.nanmean(regret,axis=1), 5)
sim_param_vals.loc[sim_num,"25%"] = np.nanpercentile(np.nanmean(regret,axis=1), 25)
sim_param_vals.loc[sim_num,"Median"] = np.nanpercentile(np.nanmean(regret,axis=1), 50)
sim_param_vals.loc[sim_num,"75%"] = np.nanpercentile(np.nanmean(regret,axis=1), 75)
sim_param_vals.loc[sim_num,"95%"] = np.nanpercentile(np.nanmean(regret,axis=1), 95)
sim_param_vals.loc[sim_num,"Max"] = np.nanpercentile(np.nanmean(regret,axis=1), 100)
for param in params:
  sim_param_vals.loc[sim_num,param] = test_param_vals[param]

sim_param_vals.to_csv(yaml_loc + "sim_param_vals.csv")
sim_param_vals.to_csv(result_dir + "sim_param_vals.csv")