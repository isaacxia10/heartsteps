import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import datetime

# QM 1
def plot_QM1(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, alpha = 0.01, title_end = ""):
    ''' Time series of probs (\pi_t(1 | S_t)) for all t, all users in light black and mean of users bolded
    '''
    fig, ax = plt.subplots(figsize = (x_size, y_size))
    out = pd.DataFrame(prob).T
    for user in out.columns:
        out[user].plot(alpha = alpha, color = "k", ax = ax)

    out.mean(axis = 1).plot(ax = ax, alpha = 1, color = "b")
    
    ax.set_title("Action Probability for All Users Together and Mean over Users (Mean in Blue)" + str(title_end))
    ax.set_ylabel(r"$\pi_t(1  | S_t)$")
    ax.set_xlabel("t (Decision Point)")
    
    return fig



# QM 1b
def plot_QM1b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, title_end = ""):
    ''' Time series of probs (\pi_t(1 | S_t)) averaged across all users
    '''
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    pd.DataFrame(prob.T).mean(axis=1).plot(ax = ax)
    ax.set_title('Average Action Probability over Decision Points for All Users' + str(title_end))
    ax.set_ylabel(r'$\pi_t(1 | S_t)$')
    ax.set_xlabel('t (Decision Point)')
    
    return fig


# QM 1c
def plot_QM1c(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, num_show = 4, alpha = 0.75, override_flag = False, percentage_to_show = 1., title_end = ""):
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
        ax[i].set_title('Action Probability over Decision Point for Given Users' + str(title_end))
        ax[i].set_ylabel(r'$\pi_t(1 | S_t)$')
        ax[i].set_xlabel('t (Decision Point)')
        
    return fig


# QM 2
def plot_QM2(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, alpha = 0.01, reward_type = "exp", title_end = ""):
    ''' Time series of cumulative expected reward averaged over all t for all users as well as for all users
    '''
    N_new = reward_exp.shape[0]
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    if reward_type == "exp":
        df_out = pd.DataFrame(np.cumsum(reward_exp,axis = 1).T)
        name = "Cumulative Expected Reward"
    elif reward_type == "0":
        df_out = pd.DataFrame(np.cumsum(reward_0,axis = 1).T)
        name = "Cumulative Reward (Action 0)"
    elif reward_type == "1":
        df_out = pd.DataFrame(np.cumsum(reward_1,axis = 1).T)
        name = "Cumulative Reward (Action 1)"
    else:
        print('Unrecognized reward_type; use "exp", "0", or "1"')
        return None
    
    for user in df_out.columns:
        df_out[user].plot(ax = ax, alpha = alpha, color = "k")
    
    df_out.mean(axis=1).plot(ax = ax, color = "b")
    
    ax.set_title(name + ' over Decision Point for All Users Together and Mean over Users (Mean in Blue)' + str(title_end))
    ax.set_ylabel(name)
    ax.set_xlabel(r'$t$ (Decision Point)')
    
    return fig



# QM2a
def plot_QM2a(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, alpha = 1, reward_type = "exp", cumulative_flag = False, title_end = ""):
    ''' Time series of cumulative expected reward vs reward for action 1 vs reward for action 2 averaged over all t for all users as well as for all users
    '''
    N_new = reward_exp.shape[0]
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    
    df_out = pd.DataFrame([reward_exp.mean(axis=0), reward_0.mean(axis=0), reward_1.mean(axis=0)]).T
    if cumulative_flag:
        df_out = df_out.cumsum(axis=0)
        
    df_out.columns = ["Expected", "Action 0", "Action 1"]
    
    df_out.plot(ax = ax, alpha = alpha)
    
    cumulative_label = ""
    if cumulative_flag:
        cumulative_label = "Cumulative "
    ax.set_title('Mean ' + cumulative_label + 'Rewards over Decision Point for All Users Together and Mean over Users (Mean in Blue)' + str(title_end))
    ax.set_ylabel(cumulative_label + "Reward")
    ax.set_xlabel(r'$t$ (Decision Point)')
    return fig

# QM 3
def plot_QM3(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, t = 5, title_end = ""):
    ''' Hist of number of actions per day taken across all N users for all t'''    
    N_new = prob.shape[0]
    
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(action.reshape((N_new, int(action.size / action.shape[0] / 5), t)),axis=-1).flatten())
    df_out.hist(ax = ax, bins = np.arange(0,7) - 0.5, grid = False, rwidth = 0.95)
    ax.set_ylabel('Count')
    ax.set_xlabel('Num Actions')
    ax.set_title("Number of Actions per Day Across All Users" + str(title_end))
    fig.tight_layout()
    
    return fig
    

# QM 3b
def plot_QM3b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 3, x_size = 15, hists_per_row = 5, percentage_to_show = 0.10, title_end = ""):
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
        
        
    st = fig.suptitle("Action Probability over Decision Point for Each User" + str(title_end))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig

# QM 4
def plot_QM4(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, num_show = 4, alpha = 0.005, title_end = ""):
    ''' Time Series of number of actions per day taken across all N users for all t and mean across all N users
    '''
    N_new = action.shape[0]

    T_new = int(action.size / action.shape[0] / 5)
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(action.reshape((N_new, T_new, 5)),axis=-1).T)
    
    
    df_out.mean(axis = 1).plot(ax = ax, color = "b")
    for user in df_out.columns:
        df_out[user].plot(ax = ax, alpha = alpha, color = "k")
    
    ax.set_title('Number of Actions Per Day for All Users and Mean over Users (Mean in Blue)' + str(title_end))
    ax.set_ylabel('Num Actions')
    ax.set_xlabel('Day')
    
    return fig


# QM 4b
def plot_QM4b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, num_show = 4, title_end = ""):
    ''' Time Series of number of actions per day taken mean across all N users for all t
    '''
    N_new = action.shape[0]

    T_new = int(action.size / action.shape[0] / 5)
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(action.reshape((N_new, T_new, 5)),axis=-1).T)
    df_out.mean(axis = 1).plot(ax = ax, color = "b")
    
    ax.set_title('Number of Actions Per Day for Mean of All Users' + str(title_end))
    ax.set_ylabel('Num Actions')
    ax.set_xlabel('Day')
    
    return fig



# QM 5
def plot_QM5(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, t = 5, title_end = ""):
    ''' Hist of number of times per day feedback controller was invoked taken across all N users for all t'''    
    N_new = fc_invoked.shape[0]
    
    fig,ax = plt.subplots(figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(fc_invoked.reshape((N_new, int(fc_invoked.size / fc_invoked.shape[0] / 5), t)),axis=-1).flatten())
    df_out.hist(ax = ax, bins = np.arange(0,7) - 0.5, grid = False, rwidth = 0.95)
    ax.set_ylabel('Count')
    ax.set_xlabel('Number of Feedback Controllings')
    ax.set_title("Number of Feedback Controllings per Day Across All Users" + str(title_end))
    fig.tight_layout()
    
    return fig
    

# QM 5b
def plot_QM5b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 3, x_size = 15, hists_per_row = 5, percentage_to_show = 0.10, title_end = ""):
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
        
        
    st = fig.suptitle("Action Probability over Decision Point for Each User" + str(title_end))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig

# QM 6
def plot_QM6(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, num_show = 4, alpha = 0.005, title_end = ""):
    ''' Time Series of number of times per day feedback controller was invoked taken across all N users for all t and mean across all N users
    '''
    N_new = fc_invoked.shape[0]

    T_new = int(fc_invoked.size / fc_invoked.shape[0] / 5)
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(fc_invoked.reshape((N_new, T_new, 5)),axis=-1).T)
    
    
    df_out.mean(axis = 1).plot(ax = ax, color = "b")
    for user in df_out.columns:
        df_out[user].plot(ax = ax, alpha = alpha, color = "k")
    
    ax.set_title('Number of Feedback Controllings Per Day for All Users and Mean over Users (Mean in Blue)' + str(title_end))
    ax.set_ylabel('Number of Feedback Controllings')
    ax.set_xlabel('Day')
    
    return fig


# QM 6b
def plot_QM6b(reward_exp, reward_0, reward_1, prob, action, fc_invoked, y_size = 5, x_size = 10, num_show = 4, title_end = ""):
    ''' Time Series of number of times per day feedback controller was invoked taken mean across all N users for all t
    '''
    N_new = fc_invoked.shape[0]

    T_new = int(fc_invoked.size / fc_invoked.shape[0] / 5)
    fig,ax = plt.subplots(1, 1, figsize = (x_size, y_size))
    df_out = pd.DataFrame(np.sum(fc_invoked.reshape((N_new, T_new, 5)),axis=-1).T)
    df_out.mean(axis = 1).plot(ax = ax, color = "b")
    
    ax.set_title('Number of Feedback Controllings Per Day for Mean of All Users' + str(title_end))
    ax.set_ylabel('Number of Feedback Controllings')
    ax.set_xlabel('Day')
    
    return fig