##################################################
import pandas as pd
import os
##################################################  

def create_combined_df(file_list):
    df_list = []
    for file_name in file_list:
        path = os.path.join('data', file_name)
        df = pd.read_csv(path, dtype={'url': str})

        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined into one dataset of shape {combined_df.shape}")
    return combined_df

def delete_by_str(str, game_metadata, game_player_stats):
    missing_criterion = game_player_stats[game_player_stats[str].isnull()]
    missing_criterion = missing_criterion['gameid'].unique()

    game_metadata = game_metadata[~game_metadata['gameid'].isin(missing_criterion)]
    game_player_stats = game_player_stats[~game_player_stats['gameid'].isin(missing_criterion)]

    return game_metadata, game_player_stats


def clean_dataframe(df):
    #Clean incomplete data
    df = df[df['datacompleteness'] == 'complete']

    #Split into game_metadata and game_player_stats
    game_metadata = df[['gameid', 'date', 'league', 'playoffs', 'gamelength']].drop_duplicates(subset='gameid')
    game_player_stats = df[[
    'gameid', 'playerid', 'teamid', 'side', 'position','champion', 'result', 'kills', 'deaths', 'assists', 
    'damagetochampions', 'visionscore', 'earnedgold', 'total cs', 'golddiffat15', 'csdiffat15',
    'wardsplaced', 'wpm', 'wardskilled', 'wcpm', 'damagetakenperminute', 'dpm', 'damageshare', 'golddiffat25', 'golddiffat20',
    'assistsat10', 'killsat10', 'monsterkills', 'cspm'
    ]]

    #Cleaning up null entrys (or ones that arent needed)
    #Deleting player entries of coaches
    game_player_stats = game_player_stats[game_player_stats['position'] != 'team']
    
    #Deleting missing team ids
    game_metadata, game_player_stats = delete_by_str('teamid', game_metadata, game_player_stats)

    #Remove games which didnt go till minute 15, 20 or 25
    game_metadata, game_player_stats = delete_by_str('golddiffat15', game_metadata, game_player_stats)
    game_metadata, game_player_stats = delete_by_str('golddiffat20', game_metadata, game_player_stats)
    game_metadata, game_player_stats = delete_by_str('golddiffat25', game_metadata, game_player_stats)

    #Remove games where damagetochampions is missing 
    game_metadata, game_player_stats = delete_by_str('damagetochampions', game_metadata, game_player_stats)

    #remove rows where earnedgold is missing
    game_metadata, game_player_stats = delete_by_str('earnedgold', game_metadata, game_player_stats)

    #remove missing vision score
    game_metadata, game_player_stats = delete_by_str('visionscore', game_metadata, game_player_stats)
    
    #remove missing playerid
    game_metadata, game_player_stats = delete_by_str('playerid', game_metadata, game_player_stats)

    return game_metadata, game_player_stats

def create_final_df(game_metadata, game_player_stats):
    #Delete entries where all players have the same teamid (should be exactly )
    #Count unique teams per game
    team_counts = game_player_stats.groupby("gameid")["teamid"].nunique()

    #Keep only games with at least 2 teams
    valid_games = team_counts[team_counts == 2].index

    game_player_stats = game_player_stats[game_player_stats["gameid"].isin(valid_games)]
    game_metadata = game_metadata[game_metadata["gameid"].isin(valid_games)]
    
    #aggregrate the player level stats to team level stats
    df = (
        game_player_stats
        .groupby(['gameid', 'teamid', 'result', 'side'], as_index = False)
        .agg({
            'kills': 'sum',
            'deaths': 'sum',
            'assists': 'sum',
            'visionscore': 'sum',
            'earnedgold': 'sum',
            'golddiffat15': 'sum',
            'total cs': 'sum',
            'wardsplaced': 'sum',
            'wardskilled': 'sum',
            'wcpm': 'sum',
            'damagetochampions': 'sum',
            'monsterkills': 'sum',
            'killsat10': 'sum',
            'assistsat10': 'sum',
            'cspm': 'mean'
        })
    )

    #add the side_adv
    df['side_adv'] = (df['side'] == 'Blue').astype(int)

    #Now we need to add the role specific golddifference 
    adc_df = (
        game_player_stats[game_player_stats["position"] == "bot"]
        .loc[:, ["gameid", "teamid", "golddiffat15", "damagetochampions", "earnedgold", "damagetakenperminute", "dpm", "damageshare", "kills", "deaths", "assists"
                 , "golddiffat20", "golddiffat25", "total cs", "cspm"
                 ]]
        .rename(columns={"golddiffat15": "adc_golddiffat15", "damagetochampions": "adc_damagetochampions", "earnedgold": "adc_earnedgold", 
                         "damagetakenperminute": "adc_damagetakenperminute", "dpm": "adc_dpm", "damageshare": "adc_damageshare", "kills": "adc_kills",
                         "deaths": "adc_deaths", "assists": "adc_assists", "golddiffat20": "adc_golddiffat20", "golddiffat25": "adc_golddiffat25",
                         "total cs": "adc_total_cs", "cspm": "adc_cspm"})
    )

    df = df.merge(adc_df, on=["gameid", "teamid"], how="left")

    jungle_df = (
        game_player_stats[game_player_stats["position"] == "jng"]
        .loc[:, ["gameid", "teamid", "golddiffat15", "damagetochampions", "earnedgold", "damagetakenperminute", "dpm", "damageshare", "kills", "deaths", "assists"
                 , "golddiffat20", "golddiffat25", "total cs", "monsterkills", "killsat10", "assistsat10", "cspm"
                 ]]
        .rename(columns={"golddiffat15": "jng_golddiffat15", "damagetochampions": "jng_damagetochampions", "earnedgold": "jng_earnedgold",
                         "damagetakenperminute": "jng_damagetakenperminute", "dpm": "jng_dpm", "damageshare": "jng_damageshare", "kills": "jng_kills", "assists": "jng_assists",
                         "deaths": "jng_deaths", "golddiffat20": "jng_golddiffat20", "golddiffat25": "jng_golddiffat25", "total cs": "jng_total_cs", "monsterkills": "jng_monsterkills",
                        "killsat10": "jng_killsat10", "assistsat10": "jng_assistsat10", "cspm": "jng_cspm"
                         })
    )

    df = df.merge(jungle_df, on=["gameid", "teamid"], how="left")

    #Merge the gamelength, playoffs and date into the df
    df = df.merge(
        game_metadata[['gameid', 'date', 'gamelength', 'playoffs']],
        on="gameid",
        how="left"
    )

    #Add the wppm (wardsplacedperminute)
    df['wppm'] = df['wardsplaced'] / (df['gamelength'] / 60)
    #use earnedgold for adc and jungle to add the damage efficiency (see notebook)
    df['adc_dmgefficiency'] = df['adc_damagetochampions'] / df['adc_earnedgold']
    df['jng_dmgefficiency'] = df['jng_damagetochampions'] / df['jng_earnedgold']

    return df


def __main__():
    file_list = os.listdir('./data')
    print(f"Data for years: {[file_name[:4] for file_name in file_list]} collected")

    combined_df = create_combined_df(file_list)

    #Create player and team map
    player_map = combined_df.set_index('playerid')['playername'].to_dict()

    team_map = combined_df.set_index('teamid')['teamname'].to_dict()

    #Clean the combined df
    game_metadata, game_player_stats = clean_dataframe(combined_df)

    #Create the dataframe we want to use 
    final_df = create_final_df(game_metadata, game_player_stats)

    print(f'Amount of unique games: {len(game_metadata)}')
    print(f'Resulting dataframe:')
    print(final_df.dtypes)
    #rint(final_df.isnull().sum())

    print("Saving final_df to csv...")
    final_df.to_csv('./out/final_df.csv', index=False)
__main__()