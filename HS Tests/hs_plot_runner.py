# Results Analysis
#Imports
import datetime
from hs_plot_funcs import *
import numpy as np
import datetime
import yaml
import os
import sys


'''
Read in:
sim_num
out_dir
data_loc_pref
'''

# python hs_test_plot_runner.py <sim_num> <fig_dir> <out_dir>

assert len(sys.argv) == 4
sim_num = int(sys.argv[1])
fig_dir = str(sys.argv[2])
out_dir = str(sys.argv[3])

if not os.path.exists(os.path.dirname(fig_dir)):
    try:
        os.makedirs(os.path.dirname(fig_dir), exist_ok = True)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST and os.path.isdir(fig_dir):
            pass
        else:
            raise

regret = None
prob = None
action = None
opt = None
fc_invoked = None
theta_mse = None
treatment_pred = None

regret = np.load(out_dir + "regret_simNum" + str(sim_num) + ".npy")
prob = np.load(out_dir + "prob_simNum" + str(sim_num) + ".npy")
action = np.load(out_dir + "action_simNum" + str(sim_num) + ".npy")
opt = np.load(out_dir + "opt_simNum" + str(sim_num) + ".npy")
fc_invoked = np.load(out_dir + "fc_invoked_simNum" + str(sim_num) + ".npy")
theta_mse = np.load(out_dir + "theta_mse_simNum" + str(sim_num) + ".npy")
treatment_pred = np.load(out_dir + "treatment_pred_simNum" + str(sim_num) + ".npy")

plot_QM1(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM1_simNum" + str(sim_num) + ".png")
plot_QM2(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM2_simNum" + str(sim_num) + ".png")
plot_QM3(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM3_simNum" + str(sim_num) + ".png")
plot_QM4(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM4_simNum" + str(sim_num) + ".png")
plot_QM5(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM5_simNum" + str(sim_num) + ".png")
plot_QM6(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM6_simNum" + str(sim_num) + ".png")
plot_QM7(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM7_simNum" + str(sim_num) + ".png")
plot_QM8(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM8_simNum" + str(sim_num) + ".png")
plot_QM9(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM9_simNum" + str(sim_num) + ".png")
plot_QM10(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM10_simNum" + str(sim_num) + ".png")
plot_QM11(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM11_simNum" + str(sim_num) + ".png")
plot_QM12(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM12_simNum" + str(sim_num) + ".png")
plot_QM13(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM13_simNum" + str(sim_num) + ".png")
plot_QM14(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM14_simNum" + str(sim_num) + ".png")
plot_QM15(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM15_simNum" + str(sim_num) + ".png")
plot_QM16(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred).savefig(fig_dir + "QM16_simNum" + str(sim_num) + ".png")