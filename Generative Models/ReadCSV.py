import pandas as pd
from rpy2.robjects import pandas2ri
pandas2ri.activate()

#Functions for reading two different .csv files
def read_hs1(yoozer):
    pd_df = pd.read_csv('suggest-analysis-kristjan.csv') #'suggest-analysis-kristjan.csv')
    pdf = pd_df[:179]
    snda = pd_df['send.active'] == 1
    snd = pd_df['send'] == 0
    usr = pd_df['user'] == yoozer
    ddf = pd_df[(snda | snd) & usr]
    ddf = ddf.reset_index(drop=True)


   return ddf,pd_df

def read_hs1_gf(yoozer):
    pd_df = pd.read_csv('suggest-kristjan.csv') #'suggest-analysis-kristjan.csv')
    pdf = pd_df[:179]
    snda = pd_df['send.active'] == 1
    snd = pd_df['send'] == 0
    usr = pd_df['user'] == yoozer
    ddf = pd_df[(snda | snd) & usr]
    ddf = ddf.reset_index(drop=True)


   return ddf,pd_df


for yoozer in pd_df.user.unique():

    ddf,pd_df = read_hs1(yoozer)

       #Make features
        #Center and scale

       decision_ind = ddf['decision.index.nogap']
        state = (ddf['jbsteps30pre.log'] - np.mean(pd_df['jbsteps30pre.log']))/np.std(pd_df['jbsteps30pre.log'])
        reward_h = ddf['jbsteps30.log']
        send_any = ddf['send']
        send_active = ddf['send.active']
        #total_sent = ddf['totalSent']



       # Study day index
        dazze = ddf['study.day.nogap']
        
       
       day_ind  = (ddf['study.day.nogap'] - np.mean(pd_df['study.day.nogap']))/np.std(pd_df['study.day.nogap'])#Number sent in last whatever
        #Add feature for # of week period (hsteps v2) WATCH OUT FOR COLINEARITY WITH INTERCEPT
      
        # Work indicator
        #wrk_ind = ddf['location.category']
        wrk_ind = ddf['loc.is.work'] #compare to string “work”
        # Location indicator
        loc_ind = ddf['loc.is.other']
        #loc_ind = ddf['location.category']
        steps_yest = (ddf['steps.yesterday.sqrt'] - np.mean(pd_df['steps.yesterday.sqrt']))/np.std(pd_df['steps.yesterday.sqrt'])
      
        steps_sd = (ddf['window7.steps60.sd'] - np.mean(pd_df['window7.steps60.sd']))/np.std(pd_df['window7.steps60.sd'])
        temp = (ddf['temperature'] - np.mean(pd_df['temperature']))/np.std(pd_df['temperature'])
        temp[ddf['temperature'] == -1024] = 0
        
       ddfgf,pd_dfgf = read_hs1_gf(yoozer)
        
       steps_gf = (np.log(ddfgf['gfsteps30pre'] + .5) - np.mean(np.log(pd_dfgf['gfsteps30pre'] + .5))/np.std(np.log(pd_dfgf['gfsteps30pre']+.5)))


#For making features vectors
for day in range(T):
           vc = state[day]
                    featVec[:,dpt] = np.ones(n)
                    featVec[4,dpt] = vc
                    
                   featVec[1,dpt] = day_ind[day]
                    
                   featVec[5,dpt] = int(wrk_ind[day])
                    featVec[2,dpt] = int(loc_ind[day])
                    featVec[6,dpt] = steps_yest[day]
                  
                    featVec[3,dpt] = float(steps_sd[day])
                    featVec[7,dpt] = float(temp[day])
                    featVec[8,dpt] = float(steps_gf[day])
                
                   featVec[np.isnan(featVec)] = 0