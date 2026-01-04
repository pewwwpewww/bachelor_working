# %%
import pandas as pd
import os

# %%
#Get all the files from the os
file_list = os.listdir('./data')
print('-' * 100)
print(f"Data for years: {[file_name[:4] for file_name in file_list]} collected")
print('-' * 100)

# %%
#Parse folder for csv files and combine
df_list = []

for file_name in file_list:
    path = os.path.join('data', file_name)
    df = pd.read_csv(path, dtype={'url': str})

    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)
print(f"Combined into one dataset of shape {combined_df.shape}")
print('-' * 100)
print(f"Combined dataset info: ")
print(combined_df.info())

# %% [markdown]
# Check incompleteness

# %%

i = 0
for val in combined_df['datacompleteness']:
    if val == 'partial':
        i += 1

print('-' * 100)
print(f"{i/combined_df.shape[0]}% of rows are partially complete")
incomplete_indices = df.index[df['datacompleteness'] == 'partial'].tolist()




# %%
#create player and team map 
player_map = combined_df.set_index('playerid')['playername'].to_dict()

team_map = combined_df.set_index('teamid')['teamname'].to_dict()

# %%
#Delete the url column since it is unneeded 
combined_df = combined_df.drop('url', axis=1)
print('-' * 100)
print(f"Removed column url for shape of {combined_df.shape}")


# %%
assert team_map['oe:team:47ae4f5f4aea5a7a0ab0b9778844cc2'] == 'Fnatic Academy'

# %%
unique_game_id = combined_df['gameid'].unique()


filtered_df = combined_df[combined_df['datacompleteness'] == 'complete']
filtered_unique_games = filtered_df['gameid'].unique()

filtered_players = filtered_df['playerid'].unique()
filtered_teams = filtered_df['teamid'].unique()

print('-' * 100)
print(f"We have {len(unique_game_id)} unique games of which {len(filtered_unique_games)} have complete data")
print(f"Looking at complete data we have {len(filtered_teams)} unique teams consisting of {len(filtered_players)} unique players")
print('-' * 100)

# %%
df.to_csv('combined.csv', index=False)
print("Saved resulting csv in ./combined.csv")


