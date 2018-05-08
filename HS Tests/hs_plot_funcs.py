import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import datetime
import statsmodels.api as sm

sns.reset_orig()

# QM 1 Regret
def plot_QM1(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, alpha = 0.005, title_end = ""):
    ''' Time series of cumulative expected regret averaged over all t for all users as well as for all users
    '''
    N_new = regret.shape[0]
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.cumsum(regret,axis = 1).T)
    
    for user in df_out.columns:
        df_out[user].plot(ax = ax, alpha = alpha, color = "k")
    
    df_out.mean(axis=1).plot(ax = ax, color = "k")
    
    ax.set_title('Cumulative Regret over Decision Point for All Users Together and Mean over Users (Mean Bolded) ' + str(title_end))
    ax.set_ylabel('Regret')
    ax.set_xlabel(r'$t$ (Decision Point)')
    
    return fig

# QM 2 Regret
def plot_QM2(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, alpha = 0.005, title_end = ""):
    ''' Time Series of cumulative expected regret, quantiles
    '''
    N_new = regret.shape[0]
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    df = pd.DataFrame(np.cumsum(regret,axis = 1).T)

    out = pd.DataFrame({"5%": df.quantile(0.05,axis=1),"25%": df.quantile(0.25,axis=1),"40%": df.quantile(0.40,axis=1),"Median": df.quantile(0.5,axis=1),"60%": df.quantile(0.60,axis=1),"75%":df.quantile(0.75,axis=1),"95%": df.quantile(0.95,axis=1),"Mean": df.mean(axis=1)})
    out[["5%","25%","40%","60%","75%","95%"]].plot(ax=ax, color = ["darkgoldenrod","gold","yellowgreen","skyblue","steelblue","b"],linewidth=0.5)
    out["Mean"].plot(ax=ax,color="k",linewidth=2.)
    out["Median"].plot(ax=ax,color="green",linewidth=2.)
    ax.fill_between(out.index, out["40%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["40%"], out["25%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["5%"], out["25%"],color='k',alpha=0.02)
    ax.fill_between(out.index, out["60%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["60%"], out["75%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["75%"], out["95%"],color='k',alpha=0.02)
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    
    ax.set_title('Cumulative Regret over Decision Point for All Users ' + str(title_end))
    ax.set_ylabel('Regret')
    ax.set_xlabel(r'$t$ (Decision Point)')
    
    return fig

# QM 3 Prob
def plot_QM3(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, alpha = 0.005, title_end = ""):
    ''' Time series of probs (\pi_t(1 | S_t)) for all t, 25/50/75 quantiles and mean
    '''
    fig, ax = plt.subplots(figsize = (x_size, y_size))
    df = pd.DataFrame(prob).T
    out = pd.DataFrame({"5%": df.quantile(0.05,axis=1),"25%": df.quantile(0.25,axis=1),"40%": df.quantile(0.40,axis=1),"Median": df.quantile(0.5,axis=1),"60%": df.quantile(0.60,axis=1),"75%":df.quantile(0.75,axis=1),"95%": df.quantile(0.95,axis=1),"Mean": df.mean(axis=1)})
    out[["5%","25%","40%","60%","75%","95%"]].plot(ax=ax, color = ["darkgoldenrod","gold","yellowgreen","skyblue","steelblue","b"],linewidth=0.5)
    out["Mean"].plot(ax=ax,color="k",linewidth=2.)
    out["Median"].plot(ax=ax,color="green",linewidth=2.)
    ax.fill_between(out.index, out["40%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["40%"], out["25%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["5%"], out["25%"],color='k',alpha=0.02)
    ax.fill_between(out.index, out["60%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["60%"], out["75%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["75%"], out["95%"],color='k',alpha=0.02)
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    
    ax.set_title("Action Probability over Users " + str(title_end))
    ax.set_ylabel(r"$\pi_t(1  | S_t)$")
    ax.set_xlabel("t (Decision Point)")

    return fig
        
# QM 3b Prob
def plot_QM3b(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, title_end = ""):
    ''' Time series of probs (\pi_t(1 | S_t)) averaged across all users
    '''
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    pd.DataFrame(prob.T).mean(axis=1).plot(ax = ax)
    ax.set_title('Average Action Probability over Decision Points for All Users ' + str(title_end))
    ax.set_ylabel(r'$\pi_t(1 | S_t)$')
    ax.set_xlabel('t (Decision Point)')
    
    return fig

# QM 3c Prob
def plot_QM3c(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, alpha = 0.75, override_flag = False, percentage_to_show = 1., title_end = ""):
    ''' Time series of probs (\pi_t(1 | S_t)) for all t,
    for num_show users at a time
    '''
    
    N_new = prob.shape[0]
    N_new = int(prob.shape[0] * percentage_to_show)
    prob = prob[:N_new]
    num_plots = int(np.ceil(N_new/num_show))
    
    if num_plots > 20 and (not override_flag):
        # May run out of memory if too many plots
        print("Too many plots")
        return None
    
    fig,ax = plt.subplots(num_plots, 1, figsize = (x_size, y_size*num_plots))
    for i in range(num_plots):
        pd.DataFrame(prob.T).iloc[:,int(i*num_show):int((i+1)*num_show)].plot(ax = ax[i], alpha = alpha)
        ax[i].set_title('Action Probability over Decision Point for Given Users ' + str(title_end))
        ax[i].set_ylabel(r'$\pi_t(1 | S_t)$')
        ax[i].set_xlabel('t (Decision Point)')
        
    return fig

# # QM 3 Prob vs Opt # NOT USEFUL
# def plot_QM3(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, user_inds_to_show = list(range(5)), alpha = 0.75, title_end = ""):
#     ''' Time series of probs (\pi_t(1 | S_t)) vs opt_t(S_t) for all t,
#     for some users at a time
    
#     opt_t(S_t) = 0.8{optimal action is 1 in context S_t} + 0.1 {optimal action is 0 in context S_t}
#     '''
#     N_new = prob.shape[0]
#     num_plots = len(user_inds_to_show)
#     fig,ax = plt.subplots(num_plots, 1, figsize = (x_size, y_size*num_plots))
    
#     for user,i in zip(user_inds_to_show,range(N_new)):
#         pd.DataFrame({"opt" : opt[i],"prob": prob[i]}).plot(ax = ax[i], alpha = alpha)
#         ax[i].set_title('Action Probability vs Optimal Probability for User ' + str(user) + " " + str(title_end))
#         ax[i].set_ylabel('Probability')
#         ax[i].set_xlabel('t (Decision Point)')
        
# QM 4 Prob vs Opt 
def plot_QM4(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, title_end = "", alpha = 0.002):
    ''' Time series of |probs (\pi_t(1 | S_t)) - opt_t(S_t)| for all t
    all users in light black and mean of users bolded
    
    opt_t(S_t) = 0.8{optimal action is 1 in context S_t} + 0.1 {optimal action is 0 in context S_t}
    '''

    fig, ax = plt.subplots(figsize = (x_size, y_size))
    out = pd.DataFrame(prob - opt).T
    for user in out.columns:
        out[user].plot(alpha = alpha, color = "k", ax = ax)

    np.abs(out).mean(axis = 1).plot(ax = ax, alpha = 1, color = "k")
    
    ax.set_title("Action Prob - Optimal Prob for All Users Together and Mean of Absolute Difference over Users (Mean Bolded) " + str(title_end))
    ax.set_ylabel('Probability Difference')
    ax.set_xlabel('t (Decision Point)')

    return fig
        
# QM 5 Prob vs Opt
def plot_QM5(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, title_end = ""):
    ''' Histogram of |probs (\pi_t(1 | S_t)) - opt_t(S_t)| averaged over all t for all users
    averaged over all users
    
    opt_t(S_t) = 0.8{optimal action is 1 in context S_t} + 0.1 {optimal action is 0 in context S_t}
    '''
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    pd.DataFrame(np.nanmean(np.abs(prob - opt),axis=1)).hist(ax = ax)
    ax.set_title('|Action Probability - Optimal Probability| Averaged over all t for all ' + str(prob.shape[0]) + ' Users ' + str(title_end))
    ax.set_xlabel('Probability Difference')
    return fig
        
# QM 6 Action
def plot_QM6(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 3, x_size = 5, t = 5, title_end = ""):
    ''' Hist of number of actions per day taken across all N users for all t'''    
    N_new = prob.shape[0]
    
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(action.reshape((N_new, int(action.size / action.shape[0] / 5), t)),axis=-1).flatten())
    df_out.hist(ax = ax, bins = np.arange(0,7) - 0.5, grid = False, rwidth = 0.95)
    ax.set_ylabel('Count')
    ax.set_xlabel('Num Actions')
    ax.set_title("Number of Actions per Day Across All Users " + str(title_end))
    fig.tight_layout()
    
    return fig    

# QM 6b Action
def plot_QM6b(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 3, x_size = 15, hists_per_row = 5, percentage_to_show = 0.10, title_end = ""):
    ''' Hist of number of actions per day taken for each of N users, up to percentage_to_show% of users
    '''
    N_new = prob.shape[0]
    N_max = int(N_new * percentage_to_show)
    action = action[:N_max]
    T_new = int(action.size / action.shape[0] / 5)
    num_plots = int(np.ceil(N_max/hists_per_row))
    fig,axes = plt.subplots(num_plots, hists_per_row, figsize = (x_size, y_size*num_plots), sharey=True)
    df_out = pd.DataFrame(np.sum(action.reshape((N_max, T_new, 5)),axis=-1).T)
    for user,ax in zip(df_out.columns,axes.flatten()):
        ax.set_title("User " + str(user))
        df_out.loc[:,user].hist(ax = ax, bins = np.arange(0,7) - 0.5, grid = False, rwidth = 0.95)
        ax.set_ylabel('Count')
        ax.set_xlabel('Number of Actions')
        
        
    st = fig.suptitle("Action Probability over Decision Point for Each User " + str(title_end))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig

# QM 7 Action
def plot_QM7(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, alpha = 0.005, title_end = ""):
    ''' Time Series of number of actions per day taken across all N users for all t and mean across all N users
    '''
    N_new = action.shape[0]

    T_new = int(action.size / action.shape[0] / 5)

    fig, ax = plt.subplots(figsize = (x_size, y_size))
    df = pd.DataFrame(np.sum(action.reshape((N_new, T_new, 5)),axis=-1).T)
    out = pd.DataFrame({"5%": df.quantile(0.05,axis=1),"25%": df.quantile(0.25,axis=1),"40%": df.quantile(0.40,axis=1),"Median": df.quantile(0.5,axis=1),"60%": df.quantile(0.60,axis=1),"75%":df.quantile(0.75,axis=1),"95%": df.quantile(0.95,axis=1),"Mean": df.mean(axis=1)})
    out[["5%","25%","40%","60%","75%","95%"]].plot(ax=ax, color = ["darkgoldenrod","gold","yellowgreen","skyblue","steelblue","b"],linewidth=0.5)
    out["Mean"].plot(ax=ax,color="k",linewidth=2.)
    out["Median"].plot(ax=ax,color="green",linewidth=2.)
    ax.fill_between(out.index, out["40%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["40%"], out["25%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["5%"], out["25%"],color='k',alpha=0.02)
    ax.fill_between(out.index, out["60%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["60%"], out["75%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["75%"], out["95%"],color='k',alpha=0.02)
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    
    ax.set_title('Number of Actions Per Day for Users ' + str(title_end))
    ax.set_ylabel('Num Actions')
    ax.set_xlabel('Day')
    
    return fig

# QM 7b Action
def plot_QM7b(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, title_end = ""):
    ''' Time Series of number of actions per day taken mean across all N users for all t
    '''
    N_new = action.shape[0]

    T_new = int(action.size / action.shape[0] / 5)
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(action.reshape((N_new, T_new, 5)),axis=-1).T)
    df_out.mean(axis = 1).plot(ax = ax, color = "k")
    
    ax.set_title('Number of Actions Per Day for Mean of All Users' + str(title_end))
    ax.set_ylabel('Num Actions')
    ax.set_xlabel('Day')
    
    return fig

# QM 8 Fc_invoked
def plot_QM8(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, t = 5, title_end = ""):
    ''' Hist of number of times per day feedback controller was invoked taken across all N users for all t'''    
    N_new = fc_invoked.shape[0]
    
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(fc_invoked.reshape((N_new, int(fc_invoked.size / fc_invoked.shape[0] / t), t)),axis=-1).flatten())

    df_out.hist(ax = ax, bins = np.arange(0,7) - 0.5, grid = False, rwidth = 0.95)
    ax.set_ylabel('Count')
    ax.set_xlabel('Number of Feedback Controllings')
    ax.set_title("Number of Feedback Controllings per Day Across All Users " + str(title_end))
    fig.tight_layout()
    
    return fig

# QM 8b Fc_invoked
def plot_QM8b(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 3, x_size = 15, hists_per_row = 5, percentage_to_show = 0.10, title_end = ""):
    ''' Hist of number of times per day feedback controller was invoked taken for each of N users, up to percentage_to_show% of users
    '''
    N_new = prob.shape[0]
    N_max = int(N_new * percentage_to_show)
    fc_invoked = fc_invoked[:N_max]
    T_new = int(fc_invoked.size / fc_invoked.shape[0] / 5)
    num_plots = int(np.ceil(N_max/hists_per_row))
    fig,axes = plt.subplots(num_plots, hists_per_row, figsize = (x_size, y_size*num_plots), sharey=True)
    df_out = pd.DataFrame(np.sum(fc_invoked.reshape((N_max, T_new, 5)),axis=-1).T)
    for user,ax in zip(df_out.columns,axes.flatten()):
        ax.set_title("User " + str(user))
        df_out.loc[:,user].hist(ax = ax, bins = np.arange(0,7) - 0.5, grid = False, rwidth = 0.95)
        ax.set_ylabel('Count')
        ax.set_xlabel('Number of Feedback Controllings')
        
        
    st = fig.suptitle("Action Probability over Decision Point for Each User " + str(title_end))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig

# QM 9 Fc_invoked
def plot_QM9(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, alpha = 0.005, title_end = ""):
    ''' Time Series of number of times per day feedback controller was invoked taken across all N users for all t and mean across all N users
    '''
    N_new = fc_invoked.shape[0]

    T_new = int(fc_invoked.size / fc_invoked.shape[0] / 5)
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    df = pd.DataFrame(np.sum(fc_invoked.reshape((N_new, T_new, 5)),axis=-1).T)

    out = pd.DataFrame({"5%": df.quantile(0.05,axis=1),"25%": df.quantile(0.25,axis=1),"40%": df.quantile(0.40,axis=1),"Median": df.quantile(0.5,axis=1),"60%": df.quantile(0.60,axis=1),"75%":df.quantile(0.75,axis=1),"95%": df.quantile(0.95,axis=1),"Mean": df.mean(axis=1)})
    out[["5%","25%","40%","60%","75%","95%"]].plot(ax=ax, color = ["darkgoldenrod","gold","yellowgreen","skyblue","steelblue","b"],linewidth=0.5)
    out["Mean"].plot(ax=ax,color="k",linewidth=2.)
    out["Median"].plot(ax=ax,color="green",linewidth=2.)
    ax.fill_between(out.index, out["40%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["40%"], out["25%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["5%"], out["25%"],color='k',alpha=0.02)
    ax.fill_between(out.index, out["60%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["60%"], out["75%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["75%"], out["95%"],color='k',alpha=0.02)
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    
    
    ax.set_title('Number of Feedback Controllings Per Day for Users ' + str(title_end))
    ax.set_ylabel('Number of Feedback Controllings')
    ax.set_xlabel('Day')
    
    return fig

# QM 9b Fc_invoked
def plot_QM9b(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, title_end = ""):
    ''' Time Series of number of times per day feedback controller was invoked taken mean across all N users for all t
    '''
    N_new = fc_invoked.shape[0]

    T_new = int(fc_invoked.size / fc_invoked.shape[0] / 5)
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(fc_invoked.reshape((N_new, T_new, 5)),axis=-1).T)
    df_out.mean(axis = 1).plot(ax = ax, color = "k")
    
    ax.set_title('Number of Feedback Controllings Per Day for Mean of All Users ' + str(title_end))
    ax.set_ylabel('Number of Feedback Controllings')
    ax.set_xlabel('Day')
    
    return fig


# QM 10 Theta_mse
def plot_QM10(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, alpha = 0.002, title_end = ""):
    ''' Time Series of MSE between "True" \theta bandit \theta, all users in light black and mean of users bolded
    '''
    fig, ax = plt.subplots(figsize = (x_size, y_size))
    out = pd.DataFrame(theta_mse).T

    for user in out.columns:
        out[user].plot(alpha = alpha, color = "k", ax = ax)

    out.mean(axis = 1).plot(ax = ax, alpha = 1., color = "k")

    ax.set_title(r"Bandit Model $\Theta$ MSE for All Users Together and Mean over Users (Mean Bolded)" + str(title_end))
    ax.set_ylabel(r"$\Theta$ MSE")
    ax.set_xlabel('t (Decision Point)')
    
    return fig


# QM 10b Theta_mse
def plot_QM10b(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, title_end = ""):
    ''' Time Series of number of times per day feedback controller was invoked taken mean across all N users for all t
    '''
    fig, ax = plt.subplots(figsize = (x_size, y_size))
    pd.DataFrame(theta_mse.T).mean(axis=1).plot(ax=ax)
    ax.set_title(r"Average Bandit Model $\Theta$ MSE for All Users" + str(title_end))
    ax.set_ylabel(r"$\Theta$ MSE")
    ax.set_xlabel('t (Decision Point)')
    
    return fig


# QM 11 Theta_mse
def plot_QM11(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, num_show = 4, alpha = 0.005, title_end = ""):
    ''' Time Series of MSE between "True" \theta bandit \theta, quantiles
    '''
    N_new = action.shape[0]

    T_new = int(action.size / action.shape[0] / 5)

    fig, ax = plt.subplots(figsize = (x_size, y_size))
    df = pd.DataFrame(theta_mse.T)
    out = pd.DataFrame({"5%": df.quantile(0.05,axis=1),"25%": df.quantile(0.25,axis=1),"40%": df.quantile(0.40,axis=1),"Median": df.quantile(0.5,axis=1),"60%": df.quantile(0.60,axis=1),"75%":df.quantile(0.75,axis=1),"95%": df.quantile(0.95,axis=1),"Mean": df.mean(axis=1)})
    out[["5%","25%","40%","60%","75%","95%"]].plot(ax=ax, color = ["darkgoldenrod","gold","yellowgreen","skyblue","steelblue","b"],linewidth=0.5)
    out["Mean"].plot(ax=ax,color="k",linewidth=2.)
    out["Median"].plot(ax=ax,color="green",linewidth=2.)
    ax.fill_between(out.index, out["40%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["40%"], out["25%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["5%"], out["25%"],color='k',alpha=0.02)
    ax.fill_between(out.index, out["60%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["60%"], out["75%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["75%"], out["95%"],color='k',alpha=0.02)
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    
    ax.set_title(r"Bandit Model $\Theta$ MSE for All Users " + str(title_end))
    ax.set_ylabel(r"$\Theta$ MSE")
    ax.set_xlabel('t (Decision Point)')
    
    return fig



# QM 12 Treatment Effect KDE and Timeseries
def plot_QM12(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 14, alpha = 0.005, title_end = "", t = 5):
    ''' Treatment effect (\theta_2^T f_2(S)) KDE by week in plot 1, time series of mean/median/quantiles in plot 2
    '''
    
    max_day = prob.shape[1] // t
    
    fig, axes = plt.subplots(1,2,figsize = (x_size,y_size))

    tp = pd.DataFrame(treatment_pred.T)

    for week_num, in zip(range(max_day//7)):
        x = tp.loc[tp.index[(tp.index / 7).astype(int) == week_num]].values.flatten()
        sns.distplot(x,ax = axes[0],label="Week %i"%(week_num+1),bins=50)
    axes[0].legend()

    axes[0].set_title("Treatment Effect Density by Treatment Week")
    axes[0].set_xlabel(r"$\theta_2^T f_2(S)$ (Treatment Effect)")
    axes[0].set_ylabel("p (Density)")


    out = pd.DataFrame({"5%": tp.quantile(0.05,axis=1),"25%": tp.quantile(0.25,axis=1),"40%": tp.quantile(0.40,axis=1),"Median": tp.quantile(0.5,axis=1),"60%": tp.quantile(0.60,axis=1),"75%":tp.quantile(0.75,axis=1),"95%": tp.quantile(0.95,axis=1),"Mean": tp.mean(axis=1)})
    out[["5%","25%","40%","60%","75%","95%"]].plot(ax=axes[1], color = ["darkgoldenrod","gold","yellowgreen","skyblue","steelblue","b"],linewidth=0.5)
    out["Mean"].plot(ax=axes[1],color="k",linewidth=2.)
    out["Median"].plot(ax=axes[1],color="green",linewidth=2.)
    axes[1].fill_between(out.index, out["40%"], out["Median"],color='k',alpha=0.2)
    axes[1].fill_between(out.index, out["40%"], out["25%"],color='k',alpha=0.085)
    axes[1].fill_between(out.index, out["5%"], out["25%"],color='k',alpha=0.02)
    axes[1].fill_between(out.index, out["60%"], out["Median"],color='k',alpha=0.2)
    axes[1].fill_between(out.index, out["60%"], out["75%"],color='k',alpha=0.085)
    axes[1].fill_between(out.index, out["75%"], out["95%"],color='k',alpha=0.02)
    leg = axes[1].legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)

    axes[1].set_title("Treatment Effect over Time " + str(title_end))

    axes[1].set_xlabel(r"$\theta_2^T f_2(S)$ (Treatment Effect)")
    axes[1].set_xlabel("t (Decision Point)")


    fig.tight_layout()
    
    return fig



# QM 13 Action Probability versus Treatment Effect
def plot_QM13(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 8, x_size = 15, alpha = 0.04, marker_size = 2, title_end = "", t = 5):
    ''' Scatter by Study week of probs (\pi_t(1 | S_t)) versus Treatment Effect (\Theta_2^T f_2(S))
    '''
    max_day = prob.shape[1] // t
    
    num_rows = 2
    fig, axes = plt.subplots(num_rows, int(np.ceil((max_day / 7 / num_rows))),figsize = (x_size,y_size),sharey = True, sharex = True)

    p = pd.DataFrame(prob.T)
    tp = pd.DataFrame(treatment_pred.T)

    for week_num,ax in zip(range(max_day//7),axes.flatten()):
        x = tp.loc[tp.index[(tp.index / 7).astype(int) == week_num]].values.flatten()
        y = p.loc[p.index[(p.index / 7).astype(int) == week_num]].values.flatten()
        sns.regplot(x=x, y=y, scatter_kws={"color": "blue","alpha":alpha,'s':marker_size}, line_kws={"color": "black"}, ax=ax)
        fit = sm.OLS(endog=y, exog=x).fit()
        ax.set_title(r"Week %i; $\hat{\beta}=%1.6f$, $r^2=%1.6f$"%(week_num+1,fit.params[0],fit.rsquared))
        ax.set_xlabel(r"$\theta_2^T f_2(S)$ (Treatment Effect)")
        ax.set_ylabel(r"$\pi_t(1  | S_t)$")
        
    fig.suptitle("Treatment Effect versus Action Probability, for varying weeks" + title_end, fontsize = 14)


    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

# QM 14 Prob proportion equal to 0
def plot_QM14(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, alpha = 0.005, title_end = "", pi_min = 0.1, pi_max = 0.8):
    ''' Time series of probs (\pi_t(1 | S_t)) for all t, 25/50/75 quantiles and mean
    '''
    fig, ax = plt.subplots(figsize = (x_size, y_size))
    converged = ((prob == pi_min) | (prob == pi_max)).astype(float)
    converged[(np.isnan(prob))] = np.nan
    converged = pd.DataFrame(converged.T)
    
    converged.mean(axis = 1).plot(ax = ax, alpha = 1, color = "k")
        
    ax.set_title("Proportion of Users with Clipped Probabilities over Time" + str(title_end))
    ax.set_ylabel(r"$\pi_t(1  | S_t) = \pi_{min}$ or $\pi_max$")
    ax.set_xlabel("t (Decision Point)")
    
    minor_xticks = np.arange(0, converged.shape[0], 5)

    ax.set_xticks(minor_xticks, minor = True)
    ax.grid(which = 'minor', alpha = 0.3)

    return fig



# QM 15 Prob difference from convergence
def plot_QM15(regret, prob, action, opt, fc_invoked, theta_mse, treatment_pred, y_size = 5, x_size = 10, alpha = 0.005, title_end = "", pi_min = 0.1, pi_max = 0.8):
    ''' Time series of distance from convergence min(|prob - pi_min|,|prob - pi_max|) for all t, 25/50/75 quantiles and mean
    '''
    fig, ax = plt.subplots(figsize = (x_size, y_size))
    converged = np.minimum(np.abs(prob - pi_min), np.abs(prob - pi_max))
#     converged[(np.isnan(prob))] = np.nan
    converged = pd.DataFrame(converged)
    df = pd.DataFrame(converged).T
    out = pd.DataFrame({"5%": df.quantile(0.05,axis=1),"25%": df.quantile(0.25,axis=1),"40%": df.quantile(0.40,axis=1),"Median": df.quantile(0.5,axis=1),"60%": df.quantile(0.60,axis=1),"75%":df.quantile(0.75,axis=1),"95%": df.quantile(0.95,axis=1),"Mean": df.mean(axis=1)})
    out[["5%","25%","40%","60%","75%","95%"]].plot(ax=ax, color = ["darkgoldenrod","gold","yellowgreen","skyblue","steelblue","b"],linewidth=0.5)
    out["Mean"].plot(ax=ax,color="k",linewidth=2.)
    out["Median"].plot(ax=ax,color="green",linewidth=2.)
    ax.fill_between(out.index, out["40%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["40%"], out["25%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["5%"], out["25%"],color='k',alpha=0.02)
    ax.fill_between(out.index, out["60%"], out["Median"],color='k',alpha=0.2)
    ax.fill_between(out.index, out["60%"], out["75%"],color='k',alpha=0.085)
    ax.fill_between(out.index, out["75%"], out["95%"],color='k',alpha=0.02)
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    
    ax.set_title(r"Absolute Difference of Probability from $\pi_{min}$ or $\pi_{max}$ over Time " + str(title_end))
    ax.set_ylabel(r"$\min(|\pi_t(1  | S_t)- \pi_{min}|, |\pi_t(1 | S_t) - \pi_{max}|)$")
    ax.set_xlabel("t (Decision Point)")

    return fig



def params_print(a,b,c,d,e):
    out = "|"
    if a is not None:
        out += "Sigma: " + str(sigmas[a]) + "|"
    if b is not None:
        out += "Gamma: " + str(gammas[b]) + "|"
    if c is not None:
        out += "Lambda: " + str(lambdas[c]) + "|"
    if d is not None:
        out += "N_c_mult: " + str(N_c_mults[d]) + "|"
    if e is not None:
        out += "T_c: " + str(T_cs[e]) + "|"
        
    return out