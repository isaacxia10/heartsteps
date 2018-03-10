# Results Analysis
#Imports
import datetime
from hs_plot_funcs import *
import numpy as np
import datetime
import yaml
import os
import sys
import psutil


'''
Read in:
sim_num
out_dir
data_loc_pref
'''

assert len(sys.argv) >= 2
sim_num = int(sys.argv[1])

if len(sys.argv) > 2:
    out_dir = str(sys.argv[2])
else:
    filename = "C:/Users/isaac/Dropbox/Harvard 17-18 Senior/Thesis/Results/ST_figs/"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    fig_dir = "C:/Users/isaac/Dropbox/Harvard 17-18 Senior/Thesis/Results/ST_figs/"
    


if len(sys.argv) > 3:
    fig_dir = str(sys.argv[3])
else:
    out_dir = "C:/Users/isaac/Dropbox/Harvard 17-18 Senior/Thesis/Results/ST/"
    


reward_exp = None
reward_0 = None
reward_1 = None
prob = None
action = None
fc_invoked = None

reward_exp = np.load(out_dir + "reward_exp_simNum" + str(sim_num) + ".npy")
reward_0 = np.load(out_dir + "reward_0_simNum" + str(sim_num) + ".npy")
reward_1 = np.load(out_dir + "reward_1_simNum" + str(sim_num) + ".npy")
prob = np.load(out_dir + "prob_simNum" + str(sim_num) + ".npy")
action = np.load(out_dir + "action_simNum" + str(sim_num) + ".npy")
fc_invoked = np.load(out_dir + "fc_invoked_simNum" + str(sim_num) + ".npy")


plot_QM1(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM1_simNum" + str(sim_num) + ".png")
plot_QM1b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM1b_simNum" + str(sim_num) + ".png")
plot_QM1c(reward_exp, reward_0, reward_1, prob, action, fc_invoked, percentage_to_show=0.02, num_show = 5, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM1c_simNum" + str(sim_num) + ".png")

plot_QM2(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM2_simNum" + str(sim_num) + ".png")
plot_QM3(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM3_simNum" + str(sim_num) + ".png")
plot_QM3b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, percentage_to_show=0.05, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM3b_simNum" + str(sim_num) + ".png")

plot_QM4(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM4_simNum" + str(sim_num) + ".png")
plot_QM4b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM4b_simNum" + str(sim_num) + ".png")
plot_QM5(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM5_simNum" + str(sim_num) + ".png")
plot_QM5b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, percentage_to_show = 0.05, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM5b_simNum" + str(sim_num) + ".png")
plot_QM6(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM6_simNum" + str(sim_num) + ".png")
plot_QM6b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, title_end = " SimNum " + str(sim_num)).savefig(fig_dir + "QM6b_simNum" + str(sim_num) + ".png")
