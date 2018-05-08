import yaml
import os
import errno
import sys


base_out_dir = "yamls"

assert len(sys.argv) > 6
print(sys.argv)
train = sys.argv[1]
ac_flag = int(sys.argv[2])
fc_flag = int(sys.argv[3])
pc_flag = int(sys.argv[4])
small_flag = int(sys.argv[5])
batch_num = int(sys.argv[6])

k = 3
sim_seed = 12731221
split_seed = 461247612
# sim_seed = 562638126
# split_seed = 2937102758
if len(sys.argv) > 7:
    sim_seed = int(sys.argv[7])
if len(sys.argv) > 8:
    split_seed = int(sys.argv[8])
if len(sys.argv) > 9:
    k = int(sys.argv[9])

master_param_vals = {"N_c_mult": 0.5,
  "T_c":5,
  "sig2_mult":1.,
  "gamma":0.9,
  "lamb":1.,
  "prior_cov_mult":0.5,
  "sim_seed": 562638126,
  "split_seed": 2937102758,
  "batch_num": 0, # to change
  "train": True,
  "k": 3,
  "prior_mean": None,
  "resid_sig2": None,
  "sim_num": 0,
  "ac_flag":True,# to change
  "fc_flag":True,# to change
  "pc_flag":True,# to change
  "small_flag":True# to change
  }

master_param_vals["ac_flag"] = bool(ac_flag)
master_param_vals["fc_flag"] = bool(fc_flag)
master_param_vals["pc_flag"] = bool(pc_flag)
master_param_vals["small_flag"] = bool(small_flag)
master_param_vals["k"] = k
master_param_vals["sim_seed"] = sim_seed
master_param_vals["split_seed"] = split_seed
master_param_vals["batch_num"] = batch_num

if train == "train":
    master_param_vals["train"] = True
elif train == "test":
    master_param_vals["test"] = False
else:
    raise NameError("train is not 'train' or 'test'")


out_dir = base_out_dir + "_" + train + str(ac_flag) + str(fc_flag) + str(pc_flag) + str(small_flag) + str(batch_num) + "/"
print(out_dir, master_param_vals)
if not os.path.exists(out_dir):
    try:
        os.makedirs(out_dir,exist_ok = True)
    except OSError as exc:
        if exc.errno == erno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise



with open(out_dir + "master.yaml","w") as f:
    yaml.dump(master_param_vals,f)
