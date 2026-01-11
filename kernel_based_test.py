import pandas as pd 
import numpy as np 
import time
from causallearn.utils.cit import CIT

start_time = time.time()

df = pd.read_csv('./out/df.csv')

#Add kill participation creation since not in the saved df.csv
#Add kill participation 
df['adc_killpart'] = ((df['adc_kills'] + df['adc_assists']) / df['kills']) 
df['jng_killpart'] = ((df['jng_kills'] + df['jng_assists']) / df['kills']) 

#add pre 10 kill participation for jng
df['jng_killpartat10'] = ((df['jng_killsat10']+df['jng_assistsat10']) / df['killsat10'])


#Clean the dataframe
df = df.sample(n=4000, random_state=42)
#a = np.random.randn(1000, 1)
#b = np.random.randn(1000, 1)
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.to_list()
df_numeric = df_numeric.dropna()
data = df_numeric.values

#Function to get index of the column since index is lost when converting to data
def col_idx(col_name):
    return numeric_cols.index(col_name)

#Create the kci object see link in notion for explanationm (use fastkci)
#kci_obj1 = CIT(np.hstack((a,2*a+10)), 'kci', kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian')
#kci_obj2= CIT(np.hstack((a,b)), 'kci', kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian')
kci_obj = CIT(data, 'kci', kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian')


def run_test(x,y,s):
    X = col_idx(x)
    Y = col_idx(y)
    S = [col_idx(val) for val in s]

    pValue = kci_obj(X,Y,S)
    print(f'Conditional independence of {x} and {y} given {s}: p-Value: {pValue}')

#Test 1
#run_test('win_prob', 'result', ['golddiffat15'])

#Test 2
#run_test('win_prob', 'kills', [])

#Test 3
#run_test('win_prob', 'kills', ['golddiffat15'])

#Test 4
#run_test('visionscore', 'result', ['golddiffat15'])

#Test 5 
#run_test('golddiffat15', 'win_prob', ['kills'])

#Test 6
#run_test('kills', 'wardsplaced', ['visionscore'])

#Test 7 
#run_test('golddiffat15', 'win_prob', [])

#Test 8
#run_test('kills', 'wardsplaced', [])

#Test 9
#run_test('gamelength', 'visionscore', ['wardsplaced', 'wardskilled', 'wcpm'])

#run_test('gamelength', 'visionscore', ['wcpm', 'wardskilled'])

#run_test('gamelength', 'visionscore', [])

#test 
#run_test('adc_deaths', 'result', []) --> p_value 0.0

#run_test('adc_cspm', 'result', ['golddiffat15']) #Conditional independence of adc_cspm and result given ['golddiffat15']: p-Value: 1.1102230246251565e-15 
#likely implies direct edge between adc_cspm and result and not just over golddiffat15 which is good for our model

#run_test('adc_deaths', 'result', []) #Conditional independence of adc_deaths and result given []: p-Value: 0.0


#run_test('adc_killpart', 'result', []) #Conditional independence of adc_killpart and result given []: p-Value: 0.0
#run_test('adc_killpart', 'result', ['adc_kills', 'kills']) #Conditional independence of adc_killpart and result given ['adc_kills']: p-Value: 8.207711602592482e-07
#Conditional independence of adc_killpart and result given ['adc_kills', 'kills']: p-Value: 0.007986799943024092
#run_test('jng_killpartat10', 'result', ['killsat10', 'jng_killsat10']) Conditional independence of jng_killpartat10 and result given ['killsat10', 'jng_killsat10']: p-Value: 0.6374709654324907
#this is actually quite important showing us that maybe jng_killpartat10 doesnt really show an influence that much or that its to early ? 
#run_test('jng_killpart', 'result', ['jng_kills', 'kills']) #Conditional independence of jng_killpart and result given ['jng_kills', 'kills']: p-Value: 1.7813572087410634e-06
#the difference between early and late game killparticipation shows us actually how little the very early game in terms of kill participation 
#impact -> should imply that later kills just mean more in terms of objective, longer respawn timers etc
#important results


#run_test('adc_kills', 'result', ['adc_killpart', 'kills']) #Conditional independence of adc_kills and result given ['adc_killpart', 'kills']: p-Value: 0.5558284836926493
#this is important because implies that modeling adc_kills with no direct edge seems correct and also that our model at that point is right
#run_test('jng_killsat10', 'result', ['jng_killpartat10', 'killsat10'])
#Conditional independence of jng_killsat10 and result given ['jng_killpartat10', 'killsat10']: p-Value: 0.5181200823562804

#run_test('adc_damagetakenperminute', 'result', [])
#run_test('adc_damagetakenperminute', 'result', ['adc_deaths'])
#Conditional independence of adc_damagetakenperminute and result given []: p-Value: 0.0
#Conditional independence of adc_damagetakenperminute and result given ['adc_deaths']: p-Value: 0.028017142069173273
#possibly an edge between adc_damagetakenperminute and result directly 

#run_test('jng_killpartat10', 'result', [])#Conditional independence of jng_killpartat10 and result given []: p-Value: 1.3159044392363484e-05
#run_test('jng_killpartat10', 'result', ['jng_killsat10'])#Conditional independence of jng_killpartat10 and result given ['jng_killsat10']: p-Value: 1.932185982200796e-05
#run_test('jng_killpartat10', 'result', ['jng_killsat10', 'jng_assistsat10'])#Conditional independence of jng_killpartat10 and result given ['jng_killsat10', 'jng_assistsat10']: p-Value: 6.438481064963142e-07

#dpm tests
#run_test('adc_dpm', 'result', [])#Conditional independence of adc_dpm and result given []: p-Value: 0.0
#run_test('adc_dpm', 'result', ['kills'])#Conditional independence of adc_dpm and result given ['kills']: p-Value: 5.089002552693955e-10
#run_test('adc_dpm', 'kills', ['adc_kills']) #Conditional independence of adc_dpm and kills given ['adc_kills']: p-Value: 0.008828889960715514
#run_test('adc_dpm', 'kills', []) #Conditional independence of adc_dpm and kills given []: p-Value: 0.0

#test if for collider structure in kills -> adc_killpart <- adc_kills 
#run_test('adc_kills', 'kills', []) #Conditional independence of adc_kills and kills given []: p-Value: 0.0
#run_test('adc_kills', 'kills', ['adc_killpart'])Conditional independence of adc_kills and kills given ['adc_killpart']: p-Value: 0.0

#run_test('kills', 'result', ['golddiffat15', 'adc_killpart', 'jng_killpartat10', 'adc_dpm', 'jng_dpm'])
#Conditional independence of kills and result given ['golddiffat15', 'adc_killpart', 'jng_killpartat10', 'adc_dpm', 'jng_dpm']: p-Value: 0.0

#run_test('win_prob', 'visionscore', [])
#run_test('win_prob', 'visionscore', ['gamelength'])
#Conditional independence of win_prob and visionscore given []: p-Value: 0.6707283960482469 --> cannot reject independence 
#Conditional independence of win_prob and visionscore given ['gamelength']: p-Value: 0.0818494535828802 --> conditioning on collider flips it  
#need to cleanly again figure out why collider flips it tbh
#implies that gamelength is a collider 

#run_test('rating_before', 'rating_after', ['win_prob']) #Conditional independence of rating_before and rating_after given ['win_prob']: p-Value: 0.0

run_test('win_prob', 'golddiffat15', []) #Conditional independence of win_prob and golddiffat15 given []: p-Value: 0.042567836301135364
#mediator ? 