# Batch runner

import os
import sys
import datetime
import time
import yaml


# assert len(sys.argv) >= 5


core_count = int(sys.argv[1]) + 1 # input is max array num, which is one less than total
core_num = int(sys.argv[2])
num_vals = int(sys.argv[3])
tot_runs = int(sys.argv[4])
yaml_loc = sys.argv[5]
out_dir = sys.argv[6]
fig_dir = sys.argv[7]
data_loc_pref = sys.argv[8]

'''
core_count = 5
num_vals = 6
core_num = 0
tot_runs = 1 # Num 
yaml_loc = "yamls_train11112/"
out_dir = "results_train11112/"
data_loc_pref = "~/HS_data/"
script_name = "hs_test_runner.py"
'''


for run in range(tot_runs):    
    for val_num in range(core_num,num_vals,core_count):
        yaml_file = yaml_loc + "params" + str(val_num) + ".yaml"
        
        ## WAITING CODE ##
        wait_counter = 0
        while not os.path.exists(yaml_file):
            time.sleep(30)
            if wait_counter == 1 or wait_counter >= 20:
                print("Waiting for " + str(yaml_file) + " " + str(datetime.datetime.now()), flush=True)
            wait_counter += 1

        # Gracefully catch
        if not os.path.isfile(yaml_file):
            raise ValueError("%s isn't a file!" % yaml_file)

        # Load in param values
        with open(yaml_loc + "/params" + str(val_num) + ".yaml") as f:
            params = yaml.load(f)

        locals().update(params)

        ####Execute Testing for Yaml####
        command = "python hs_test_runner.py " + str(val_num) + " " + out_dir + " "  + data_loc_pref + " " + yaml_loc
        # print("Run " + str(run) + ", val:" + str(val_num) + " out of " + str(num_vals) + "," + str(core_count) ,flush=True)
        os.system(command)

        #### Execute Plotting for Yaml####
        command = "python hs_plot_runner.py " + str(sim_num) + " " + fig_dir + " " + out_dir
        os.system(command)

        # Remove yaml so main runner can continue
        try:
            os.remove(yaml_file)
        except OSError:
            # Safe check
            pass
