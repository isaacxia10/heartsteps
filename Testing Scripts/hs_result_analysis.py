# Results Analysis
#Imports

import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import pickle
import itertools
import datetime


k = 3
N_new = 500

T = 42
t = 5


overall_seed = 2018
overall_seed_state = np.random.RandomState(overall_seed)

# out_dir = "results/"
# fig_dir = "results_plots/"

out_dir = "C:/Users/isaac/Dropbox/Harvard 17-18 Senior/Thesis/Results/ST/"
fig_dir = "C:/Users/isaac/Dropbox/Harvard 17-18 Senior/Thesis/Results/ST_figs/"


#DEEP SCAN VALUES
# batch_nums = list(range(k))
# sig2_mults = [0.5,1.,2.]
# prior_cov_mults = [0.1,0.5,1.,2.]
# gammas = [0.8,0.9,0.99,1.]
# lambs = [0.1,0.5, 1.,2.]
# N_c_mults = [0.25, 0.5,0.75]
# T_cs = [5, 10, 20]

batch_nums = [1]

sig2_mults = [0.5,1.,2.]
prior_cov_mults = [0.25,1.,4.]
gammas = [0.6,0.8,0.9,0.99]
lambs = [0.1,1.,10.]
N_c_mults = [0.25,0.5]
T_cs = [5, 10, 20]

sim_count = len(batch_nums)*len(sig2_mults)*len(prior_cov_mults)*len(gammas)*len(lambs)*len(N_c_mults)*len(T_cs)


datNames = ["reward_exp","reward_0","reward_1","prob","action","fc_invoked","bandit_mean"]

paramNames = ['batch_num', 'sig2_mult', 'prior_cov_mult', 'gamma', 'lamb', 'N_c_mult', 'T_c']

## READ DATA
reward_exps = np.empty((len(batch_nums), len(sig2_mults), len(prior_cov_mults), len(gammas), len(lambs), len(N_c_mults), len(T_cs), N_new, T*t))
reward_0s = np.empty((len(batch_nums), len(sig2_mults), len(prior_cov_mults), len(gammas), len(lambs), len(N_c_mults), len(T_cs), N_new, T*t))
reward_1s = np.empty((len(batch_nums), len(sig2_mults), len(prior_cov_mults), len(gammas), len(lambs), len(N_c_mults), len(T_cs), N_new, T*t))
probs = np.empty((len(batch_nums), len(sig2_mults), len(prior_cov_mults), len(gammas), len(lambs), len(N_c_mults), len(T_cs), N_new, T*t))
actions = np.empty((len(batch_nums), len(sig2_mults), len(prior_cov_mults), len(gammas), len(lambs), len(N_c_mults), len(T_cs), N_new, T*t))
fc_invokeds = np.empty((len(batch_nums), len(sig2_mults), len(prior_cov_mults), len(gammas), len(lambs), len(N_c_mults), len(T_cs), N_new, T*t))

dats = [reward_exps,reward_0s,reward_1s,probs,actions,fc_invokeds]

for sim_num,(batch_num,sig2_mult,prior_cov_mult,gamma,lamb,N_c_mult,T_c) in zip(range(sim_count), itertools.product(batch_nums,sig2_mults,prior_cov_mults,gammas,lambs,N_c_mults,T_cs)):
    if sim_num % 100 == 0:
        print(sim_num)
    for dat, datName in zip(dats, datNames):
        dat[batch_nums.index(batch_num), sig2_mults.index(sig2_mult), prior_cov_mults.index(prior_cov_mult), gammas.index(gamma), lambs.index(lamb), N_c_mults.index(N_c_mult), T_cs.index(T_c)] = np.load(out_dir + datName + "_simNum" + str(sim_num) + ".npy")
    
    
# Create Reward_exp mean and std
reward_exp_means = np.nanmean(reward_exps, axis=-1)
reward_exp_stds = np.nanstd(reward_exps, axis=-1)
reward_exp_dat = pd.DataFrame([reward_exp_means.flatten(),reward_exp_stds.flatten()]).T

reward_exp_dat = pd.concat([reward_exp_dat,pd.DataFrame(list(itertools.product(list(range(N_new)),batch_nums,sig2_mults,prior_cov_mults,gammas,lambs,N_c_mults,T_cs)))],axis=1)

reward_exp_dat.columns = ["Mean", "Std", "User"] + paramNames
reward_exp_dat.drop("User",axis=1,inplace=True)

g = sns.pairplot(reward_exp_dat, diag_kind = "kde", kind = "reg")
g.savefig("pairplot.png",dpi=200)


# Regress endog according to parameter
endog = "Mean"
for exog in paramNames:
    model = sm.OLS(reward_exp_dat[endog],sm.add_constant(reward_exp_dat[exog])).fit()
    print(endog + "~" + exog)
    print(model.summary())
    
    
endog = "Std"
for exog in paramNames:
    model = sm.OLS(reward_exp_dat[endog],sm.add_constant(reward_exp_dat[exog])).fit()
    print(endog + "~" + exog)
    print(model.summary())

# Scatter for varying colors for params
for paramName in paramNames:
    fg = sns.FacetGrid(data=reward_exp_dat, hue=paramName, aspect=1.61, size = 6,scatter_kws={'alpha':0.3})
    fg.map(plt.scatter, "Mean", "Std").add_legend()
    fg.savefig(paramName + "_MeanStdScatter.png")