import pandas as pd
from openskill.models import PlackettLuce
import time

model = PlackettLuce()

start_time = time.time()

df = pd.read_csv('./out/df_with_elo.csv')

#First we want to add the win probability which we can calculate with the mu and sigma before
#We also calculate the rating using mu - 3*sigma which gives us a rating where the teams true rating has a 99.7%
#Likelihood of being higher so a conservative rating 
def  calc_new_columns(group):
    #Create ratings
    mu_before = group['mu_before']
    sigma_before = group['sigma_before']

    t1 = model.create_rating([mu_before.iloc[0], sigma_before.iloc[0]])
    t2 = model.create_rating([mu_before.iloc[1], sigma_before.iloc[1]])

    #Predict wins
    predictions = model.predict_win([[t1], [t2]])
    group['win_prob'] = predictions

    #Calculate rating_before using ordinal
    t1_rating = t1.ordinal()
    t2_rating = t2.ordinal()

    group['rating_before'] = [t1_rating, t2_rating]
    group['opp_rating_before'] = [t2_rating, t1_rating]

    #Calculate rating_after using mu - 3*sigma since thats the same as ordinal but we dont want to create
    #extra Rating objects 
    mu_after = group['mu_after']
    sigma_after = group['sigma_after']

    t1_rating_after = (mu_after.iloc[0]) - 3*(sigma_after.iloc[0])
    t2_rating_after = (mu_after.iloc[1]) - 3*(sigma_after.iloc[1])
    group['rating_after'] = [t1_rating_after, t2_rating_after]

    return group

df = df.groupby('gameid', group_keys=False).apply(calc_new_columns)

df['rating_gain'] = df['rating_after'] - df['rating_before']

end_time = time.time()

print(f'Runtime: {end_time-start_time} seconds')
print(df[['mu_before', 'sigma_before', 'win_prob', 'rating_before', 'result', 'rating_after', 'rating_gain']].head())

df.to_csv('./out/df.csv', index=False)