def generate_new_users(Eta, A, S, N_new, T_new):
    '''
    Generates new random user
    
    s
    Inputs:
        Eta: Residuals
        A: Actions
        S: States
        N_new: Int of number of new users to generate from sampling
        T_new: Int of number of days for each new user
    
    Returns:
        Eta_new: Matrix of Etas for sampled users
        A_new: Matrix of associated actions for sampled users
        S_new: Matrix of associated states for sampled users
    *Assumes
    '''
    
    # Obtain original dimensions of data from actions A
    N = A.shape[0]
    T = A.shape[1]
    t = A.shape[2]

    # Component dims of A and S 
    a_dim = A.shape[3]
    s_dim = S.shape[3]

    # Sample random users from original data
    users_to_sample = 3
    sampled_users = np.empty((N_new, users_to_sample)).astype(int)

    # Loop to reset sampling without replacement for each new user
    for i in range(N_new):
        sampled_users[i] = np.random.choice(N, size = users_to_sample, replace = False)

    # Sampled Generated residuals
    Eta_new = np.take(Eta, sampled_users, 0).reshape(N_new, users_to_sample * T, t)[:,:T_new,:]
    # Sampled actions
    A_new = np.take(A, sampled_users, 0).reshape(N_new, users_to_sample * T, t, a_dim)[:,:T_new,:,:]
    # Sampled states
    S_new = np.take(S, sampled_users, 0).reshape(N_new, users_to_sample * T, t, s_dim)[:,:T_new,:,:]

    return Eta_new, A_new, S_new

# Generate new Users
Eta_new, A_new, S_new = generate_new_users(Eta, A, S, 20, 90)

def reward_func(Eta, A, S, Theta):
    '''Basic reward function, can edit for different generative models'''
    return Eta + np.concatenate([A, S], A.ndim-1).dot(Theta)

print(reward_func(Eta_new, A_new, S_new, np.array(range(12))))