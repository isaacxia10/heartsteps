import datetime
from hs_test_funcs import *
import numpy as np
import datetime
import yaml
import os
import sys


'''
5/2/18
Read in:
'''
# python hs_test_runner.py <job_num> <out_dir> <data_loc_pref> <param_loc> <final_flag>
assert len(sys.argv) >= 5
job_num = int(sys.argv[1])
# print(sys.argv)

out_dir = str(sys.argv[2])

if not os.path.exists(os.path.dirname(out_dir)):
    try:
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise

data_loc_pref = str(sys.argv[3])
param_loc = str(sys.argv[4])

final_flag = False
if len(sys.argv) > 5:
    print(sys.argv[5])
    if sys.argv[5] == "final":
        final_flag = True






def f_baseline_small(featVec):
    return featVec[...,[0,3,4,5,6]]

def f_interact_small(featVec):
    return featVec[...,[0,1,2]]


def f_baseline_identity(featVec):
    return featVec

def f_interact_identity(featVec):
    return featVec[...,:4]

# Load parameters
with open(param_loc + "/params" + str(job_num) + ".yaml") as f:
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
    
    
    predictors = np.concatenate([A * f_interact(S), f_baseline(S)], -1)
    
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
nFeatures = 1+7
nBaseline_identity = 1+7
nInteract_identity = 1+3 # Add 1 for bias term
nBaseline_small = 1+4
nInteract_small = 1+2 # Add 1 for bias term

N_sim = 2500
T_new = 90

# start = datetime.datetime.now()

S, R, A = read_data(N_data, T, t, nFeatures, data_loc_pref)


##########################
## SET BANDIT MODEL ##
##########################
if small_flag:
    f_interact = f_interact_small
    f_baseline = f_baseline_small
    nInteract = nInteract_small
    nBaseline = nBaseline_small
else:
    f_interact = f_interact_identity
    f_baseline = f_baseline_identity
    nInteract = nInteract_identity
    nBaseline = nBaseline_identity




# ###############
# ## Non-split ##
# ###############
# resids, Thetas_fit, resid_model = resid_regression(N, T, t, nBaseline, nInteract, f_baseline_identity, f_interact_identity, R, A, S)
# resids_new, A_new, S_new = sample_sim_users(resids, A, S, N_sim, T, t)
# I_new = np.expand_dims(~np.isnan(S_new).any(axis=-1),-1).astype(int) #



# resid_sig2 = np.nanvar(resids)
# prior_mean = Thetas_fit



# prior_mdl = model(np.expand_dims(prior_mean,1), np.eye(nInteract+nBaseline) * prior_cov_mult)
# fc_params = [lamb, int(N_c_mult*T_c), T_c]


# # sim_start = datetime.datetime.now()
# # reward_exp, reward_0, reward_1, prob, action, fc_invoked, bandit_mean, bandit = run_simulation(coef0 = Thetas_fit[4:12], coef1 = Thetas_fit[0:4], S_sim = S_new, I_sim = I_new, resids_sim = resids_new, f_baseline=f_baseline_identity, f_interact=f_interact_identity, reward_func = reward_func_identity, fc_params = fc_params, prior_model = prior_mdl, gamma = gamma, sigma2 = resid_sig2*sig2_mult, seed=sim_seed)
# reward_exp, reward_0, reward_1, prob, action, fc_invoked, bandit = run_simulation(coef0 = Thetas_fit[4:12], coef1 = Thetas_fit[0:4], S_sim = S_new, I_sim = I_new, resids_sim = resids_new, f_baseline=f_baseline_identity, f_interact=f_interact_identity, reward_func = reward_func_identity, fc_params = fc_params, prior_model = prior_mdl, gamma = gamma, sigma2 = resid_sig2*sig2_mult, seed=sim_seed)



##################################
## Split train and test batches ##
##################################
train_zip, test_zip = k_fold_split(S,R,A,k,seed = split_seed) ## Seed for split

S_train,R_train,A_train = train_zip[batch_num]
resids_train, Thetas_fit_train, resid_model_train = resid_regression(S_train.shape[0], T, t, nBaseline_identity, nInteract_identity, f_baseline_identity, f_interact_identity, R_train.squeeze(), A_train, S_train)

_, Thetas_fit_bandit_train, _ = resid_regression(S_train.shape[0], T, t, nBaseline, nInteract, f_baseline, f_interact, R_train.squeeze(), A_train, S_train)
# First Thetas_fit is for the true generative model, second is for the bandit model

if not train:
    # Use S,R,A from test once we've obtained the train Thetas_fit
    S_train,R_train,A_train = test_zip[batch_num]
    resids_train = resid_regression_test(S_train.shape[0], T, t, nBaseline_identity, nInteract_identity, f_baseline_identity, f_interact_identity, R_train.squeeze(), A_train, S_train, Thetas_fit_train)


resids_new_train, A_new_train, S_new_train = sample_sim_users(resids_train, A_train, S_train, N_sim, T, t, seed = split_seed)
I_new_train = np.expand_dims(~np.isnan(S_new_train).any(axis=-1),-1).astype(int) #


# Use sigma^2 of residuals from training batches, for both in test and train
resid_sig2 = np.nanvar(resids_train)

# If train, use prior of 0; otherwise, use prior from training
if train:
    prior_mean = np.zeros(nInteract+nBaseline)
else:
    prior_mean = Thetas_fit_bandit_train
    
    


prior_mdl = model(np.expand_dims(prior_mean,1), np.eye(nInteract+nBaseline) * prior_cov_mult)
fc_params = [lamb, int(N_c_mult*T_c), T_c]


# sim_start = datetime.datetime.now()
# reward_exp, reward_0, reward_1, prob, action, fc_invoked, bandit_mean, bandit = run_simulation(coef0 = Thetas_fit_train[4:12], coef1 = Thetas_fit_train[0:4], S_sim = S_new_train, I_sim = I_new_train, resids_sim = resids_new_train, f_baseline=f_baseline_identity, f_interact=f_interact_identity, reward_func = reward_func_identity, fc_params = fc_params, prior_model = prior_mdl, gamma = gamma, sigma2 = resid_sig2*sig2_mult, seed=sim_seed)

regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, bandit = run_simulation(coef0 = Thetas_fit_train[nInteract_identity:nInteract_identity+nBaseline_identity], coef1 = Thetas_fit_train[0:nInteract_identity], nInteract = nInteract, nBaseline = nBaseline, S_sim = S_new_train, I_sim = I_new_train, resids_sim = resids_new_train, f_baseline=f_baseline, f_interact=f_interact, f_baseline_identity=f_baseline_identity, f_interact_identity=f_interact_identity, reward_func = reward_func_general, fc_params = fc_params, prior_model = prior_mdl, gamma = gamma, sigma2 = resid_sig2*sig2_mult, seed=sim_seed, ac_flag = ac_flag, fc_flag = fc_flag, pc_flag = pc_flag, Thetas_fit_bandit = Thetas_fit_bandit_train)


#################
## Output Data ##
#################

# datNames = ["reward_exp","reward_0","reward_1","prob","action","fc_invoked","bandit_mean"]
# dats = [reward_exp,reward_0,reward_1,prob,action,fc_invoked,bandit_mean]

# Without bandit_mean
datNames = ["regret","prob","action","opt","fc_invoked","theta_mse","treatment_pred"]
dats = [regret,prob,action,opt,fc_invoked,theta_mse,treatment_pred]

for dat, datName in zip(dats, datNames):
    np.save(out_dir + datName + "_simNum" + str(sim_num) + ".npy", dat)

# OPTIONAL
if final_flag:
    #################
    ## Save Params ##
    #################
    params["computed"] = True
    params["prior_mean"] = Thetas_fit_bandit_train.tolist()
    params["resid_sig2"] = float(resid_sig2)
    params["train"] = False
    with open(param_loc + "/params" + str(job_num) + ".yaml", "w") as f:
        yaml.dump(params, f)