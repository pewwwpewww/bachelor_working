from openskill.models import PlackettLuce
import pandas as pd
import time

df = pd.read_csv('./out/final_df.csv')

teams = df['teamid'].unique()

#Use the PlackettLuce model 

model = PlackettLuce()

#Create the ratings with the default values for mu (average skill of the player) and sigma (uncertainty in the skill rating)
ratings = {team: model.rating(name=str(team)) for team in teams}

#This doesnt work yet but we need to set the mu and sigma values before and after in the table probably and the mu is the skill rating basically
#Set the current values as default values for the teams 
#df['elo_before'] = df['teamid'].map(lambda teamid: ratings[teamid].mu)

#df['opp_elo_before'] = (                    #For each game we expect two rows (one for each team) and then we reverse the elo_before
 #   df.groupby('gameid')['elo_before']      #and put that as the opp_elo_before
  #  .transform(lambda rows: rows[::-1].values if len(rows) == 2 else rows)
#)

#Save mu and sigma to dataframe
df['mu_before'] = pd.NA
df['sigma_before'] = pd.NA
df['mu_after'] = pd.NA
df['sigma_after'] = pd.NA

#Sort games by date
df = df.sort_values(by='date')

start_time = time.time()

#We need to iterate over all matches and for each update the rating 
for gameid, game in df.groupby('gameid'):
    if len(game) != 2:
        print(f'Irregular match with more/less than two teams: {gameid}: with ')
        break

    #Get the current ratings 
    team1_id, team2_id = game['teamid'].values

    team1_rating = ratings[team1_id]
    team2_rating = ratings[team2_id]

    #Record the before state 
    df.loc[game.index, 'mu_before'] = [team1_rating.mu, team2_rating.mu]
    df.loc[game.index, 'sigma_before'] = [team1_rating.sigma, team2_rating.sigma]


    #Determine winner 
    match = None
    first_won = None
    if game.iloc[0]['result'] == 1:     #Team 1 won
        match = [[team1_rating], [team2_rating]]
        first_won = True
    else:                               #Team 2 won
        match = [[team2_rating], [team1_rating]]
        first_won = False

    assert match != None
    assert first_won != None

    #Update the ratings
    updated_ratings = model.rate(match)
    [team1_new], [team2_new] = updated_ratings

    #Update ratings according to won 
    if first_won: 
        ratings[team1_id], ratings[team2_id] = team1_new, team2_new
    else:
        ratings[team2_id], ratings[team1_id] = team2_new, team1_new
    
    
    #Save the after state into database
    df.loc[game.index, "mu_after"] = [ratings[team1_id].mu, ratings[team2_id].mu]
    df.loc[game.index, "sigma_after"] = [ratings[team1_id].sigma, ratings[team2_id].sigma]
    
end_time = time.time()  
print("Runtime of the for loop:", end_time - start_time, "seconds")
test = df[['mu_before', 'mu_after', 'sigma_before', 'sigma_after']]
print(df.isnull().sum())

#Save to out 
df.to_csv('./out/df_with_elo.csv', index=False)

    
