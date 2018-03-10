import datetime
from hs_test_funcs import *
import numpy as np
import datetime
import yaml
import os
import sys
import psutil

process = psutil.Process(os.getpid())
print(process.memory_info().rss)



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
    filename = "results/"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    out_dir = "results/"

if len(sys.argv) > 3:
    data_loc_pref = str(sys.argv[3])
else:
    data_loc_pref = "C:/Users/isaac/Dropbox/Harvard 17-18 Senior/Thesis/Data/"


if len(sys.argv) > 4:
    param_loc = str(sys.argv[4])
else:
    param_loc = "yamls/"

def f_baseline_identity(featVec):
    return featVec

def f_interact_identity(featVec):
    return featVec[...,:4]

# Load parameters
with open(param_loc + "/params" + str(sim_num) + ".yaml") as f:
    params = yaml.load(f)

locals().update(params)


# Generative Model fit


#Create a reward function from true coefficients, with residuals non-mandatory
def reward_func_general(S, A, coef0, coef1, f_baseline, f_interact, resid = None):
    '''Generalized reward function, can edit for different generative models.
    Works for multidimensional eta, a, and s, so long as they are of same multidimension.
    Assumes s has first element as bias for the regression.
    
    Resid must be passed in if not single dim.
    If single dim, can speed out without np.take.'''
    
    
    predictors = np.concatenate([A * f_interact(S), f_baseline(S)], 0)
    
    Theta = np.concatenate([coef1, coef0])
    
    if resid is None:
        resid = 0
        
    return(resid + np.dot(predictors, Theta))



## Creating Simulations



# Counts from HS 1
N_data = 48 # users indexed up to 48, but true count is 37
N = 37
T = 42
t = 5
nBaseline = 1+7
nInteract = 1+3 # Add 1 for bias term

N_sim = 100
T_new = 90

# start = datetime.datetime.now()

S, R, A = read_data(N_data, T, t, nBaseline, data_loc_pref)



###############
## Non-split ##
###############
resids, Thetas_fit, resid_model = resid_regression(N, T, t, nBaseline, nInteract, f_baseline_identity, f_interact_identity, R, A, S)
resids_new, A_new, S_new = sample_sim_users(resids, A, S, N_sim, T, t)
I_new = np.expand_dims(~np.isnan(S_new).any(axis=-1),-1).astype(int) #



resid_sig2 = np.nanvar(resids)
prior_mean = Thetas_fit



prior_mdl = model(np.expand_dims(prior_mean,1), np.eye(nInteract+nBaseline) * prior_cov_mult)
fc_params = [lamb, int(N_c_mult*T_c), T_c]


# sim_start = datetime.datetime.now()
reward_exp, reward_0, reward_1, prob, action, fc_invoked, bandit_mean, bandit = run_simulation(coef0 = Thetas_fit[4:12], coef1 = Thetas_fit[0:4], S_sim = S_new, I_sim = I_new, resids_sim = resids_new, f_baseline=f_baseline_identity, f_interact=f_interact_identity, reward_func = reward_func_identity, fc_params = fc_params, prior_model = prior_mdl, gamma = gamma, sigma2 = resid_sig2*sig2_mult, seed=sim_seed)

process = psutil.Process(os.getpid())
print(process.memory_info().rss)



# ##################################
# ## Split train and test batches ##
# ##################################
# train_zip, test_zip = k_fold_split(S,R,A,k,seed = split_seed) ## Seed for split

# S_train,R_train,A_train = train_zip[batch_num]


# resids_train, Thetas_fit_train, resid_model_train = resid_regression(S_train.shape[0], T, t, nBaseline, nInteract, f_baseline_identity, f_interact_identity, R_train.squeeze(), A_train, S_train)
# resids_new_train, A_new_train, S_new_train = sample_sim_users(resids_train, A_train, S_train, N_sim, T, t)
# I_new_train = np.expand_dims(~np.isnan(S_new_train).any(axis=-1),-1).astype(int) #

# if train:
#     resid_sig2 = np.nanvar(resids_train)
#     prior_mean = Thetas_fit_train


# prior_mdl = model(np.expand_dims(prior_mean,1), np.eye(nInteract+nBaseline) * prior_cov_mult)
# fc_params = [lamb, int(N_c_mult*T_c), T_c]


# # sim_start = datetime.datetime.now()
# reward_exp, reward_0, reward_1, prob, action, fc_invoked, bandit_mean, bandit = run_simulation(coef0 = Thetas_fit_train[4:12], coef1 = Thetas_fit_train[0:4], S_sim = S_new_train, I_sim = I_new_train, resids_sim = resids_new_train, f_baseline=f_baseline_identity, f_interact=f_interact_identity, reward_func = reward_func_identity, fc_params = fc_params, prior_model = prior_mdl, gamma = gamma, sigma2 = resid_sig2*sig2_mult, seed=sim_seed)



datNames = ["reward_exp","reward_0","reward_1","prob","action","fc_invoked","bandit_mean"]
dats = [reward_exp,reward_0,reward_1,prob,action,fc_invoked,bandit_mean]

for dat, datName in zip(dats, datNames):
    np.save(out_dir + datName + "_simNum" + str(sim_num) + ".npy", dat)