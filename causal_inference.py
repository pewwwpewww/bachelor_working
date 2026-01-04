import pandas as pd 
from dowhy import CausalModel
import dowhy
import warnings
import time
from sklearn.preprocessing import StandardScaler

#Block futurewarnings for now
warnings.simplefilter(action='ignore', category=FutureWarning)

start_time = time.time()

#Load data
df = pd.read_csv('./out/df.csv')

#Try to standardize the units 
'''scaler = StandardScaler()
df[['rating_gain', 'adc_golddiffat15', 'jng_golddiffat15', 'adc_damagetochampions']] = scaler.fit_transform(
    df[['rating_gain', 'adc_golddiffat15','jng_golddiffat15', 'adc_damagetochampions']]
)'''

#Define causal graph
causal_graph = '''
digraph {
    wardsplaced -> visionscore;
    wardskilled -> visionscore;
    adc_damageshare -> kills;
    adc_damagetakenperminute -> kills;
    adc_dpm -> kills;
    adc_dpm -> adc_golddiffat15;
    adc_dpm -> adc_kills;
    jng_dpm -> kills;
    jng_dpm -> jng_golddiffat15;
    jng_dpm -> jng_kills;
    adc_damagetochampions -> adc_golddiffat15;
    wcpm -> visionscore;
    wppm -> visionscore;
    visionscore -> gamelength;
    win_prob -> gamelength;
    rating_before -> win_prob;
    opp_rating_before -> win_prob;
    win_prob -> golddiffat15;
    golddiffat15 -> result;
    result -> rating_after;
    rating_before -> rating_after;
    opp_rating_before -> rating_after;
    win_prob -> result;
    rating_after -> rating_gain;
    visionscore -> kills; 
    visionscore -> deaths; 
    visionscore -> assists;
    kills -> golddiffat15;
    kills -> adc_golddiffat20;
    kills -> jng_golddiffat20;
    kills -> adc_golddiffat25;
    kills -> jng_golddiffat25;
    adc_kills -> kills;
    adc_kills -> adc_golddiffat15;
    jng_kills -> kills;
    jng_kills -> jng_golddiffat15;
    adc_golddiffat20 -> result;
    jng_golddiffat20 -> result;
    adc_golddiffat25  -> result;
    jng_golddiffat25 -> result
    side_adv -> win_prob;
    jng_golddiffat15 -> golddiffat15;
    adc_golddiffat15 -> golddiffat15;
    adc_gda15_bin -> golddiffat15;
    jng_gda15_bin -> golddiffat15;
    damagetochampions -> kills;
    adc_damagetochampions -> kills;
    jng_damagetochampions -> kills;  
}
'''

#Method that does all causal inference steps given a treatment and outcome 
def causal_inference(treatment, outcome, method_name = 'backdoor.linear_regression', data = df):
    #Create causal model
    model = CausalModel(
        data = data,
        treatment = treatment,
        outcome = outcome,
        graph = causal_graph
    )

    #Identify causal effects
    identified_estimand = model.identify_effect()

    #Estimate the causal effect using backdoor adjustment with linearregression
    estimate = model.estimate_effect(
        identified_estimand,
        method_name = method_name
    )
    
    refute_results = None
    #refute_results = model.refute_estimate(identified_estimand, estimate, method_name='random_common_cause')

    return estimate, refute_results

#Sanity checks: We look at simple questions which should give us an effect to see if we can trust our model 
#est_kills_gda15, _ = causal_inference(treatment='kills', outcome='golddiffat15')
#print(f'Effect of kills on golddiffat15: {est_kills_gda15.value}')

#est_gda15_result, _ = causal_inference(treatment='golddiffat15', outcome='result')
#print(f'Effect of gda15 on result: {est_gda15_result.value}')


est_dpm_jng, _ = causal_inference('jng_dpm', 'result')
est_dpm_adc, _ = causal_inference('adc_dpm', 'result')

print(f'Estimate for jng:{est_dpm_jng.value}, and for adc: {est_dpm_adc.value} ')

exit()
'''
Effect of kills on golddiffat15: 100.62406432450757
Effect of gda15 on result: 9.169265658215409e-05 --> 9% for 1k gold lead
'''



'''
Test for late game influence and for the kills also try after for with adding golddiffat15 to confounders so golddiffat15 _> adc_golddiffat20
Effect of golddiffat20 on the result: adc = 7.294455483897178e-05, jng = 7.972010771317706e-05
Effect of golddiffat25 on the result: adc = 7.344157951577612e-05, jng = 8.285096454307972e-05
Estimate of effect of kills on the adc pn thje result: 0.08611688971284154
Estimate of effect of kills on the jng pn thje result: 0.08462263551309324
'''
est_jng_kills_gda15, _ = causal_inference('jng_kills', 'golddiffat15')
print(f'Estimate of effect of jng kills on gda15: {est_jng_kills_gda15.value}')
exit()
#est_adc_kills_result, _ = causal_inference('adc_kills', 'result')
est_jng_kills_result, _ = causal_inference('jng_kills', 'result')
print(f'Estimate of effect of kills on the adc pn thje result: {est_jng_kills_result.value}')
exit()
est_adc_gda20_result, _ = causal_inference('adc_golddiffat20', 'result')
est_adc_gda25_result, _ = causal_inference('adc_golddiffat25', 'result')
est_jng_gda20_result, _ = causal_inference('jng_golddiffat20', 'result')
est_jng_gda25_result, _ = causal_inference('jng_golddiffat25', 'result')

print(f'Effect of golddiffat20 on the result: adc = {est_adc_gda20_result.value}, jng = {est_jng_gda20_result.value}')
print(f'Effect of golddiffat25 on the result: adc = {est_adc_gda25_result.value}, jng = {est_jng_gda25_result.value}')

exit()
''' Test to check the effect of having a 1k, 2k .... gold lead at minute 15 on the adc role vs jng role
Causal effect of 1k gold lead on result on adc: 0.14573174860931093, and on jng: 0.16110055490279374
Causal effect of 2k gold lead on result on adc: 0.1511908329284833, and on jng: 0.21663943766593013
'''
df['adc_gda15_bin'] = (df['adc_golddiffat15'] >= 2000).astype(int)
df['jng_gda15_bin'] = (df['jng_golddiffat15']>= 2000).astype(int)
est_adc1k_result,_ = causal_inference('adc_gda15_bin', 'result')
est_jng1k_result,_ = causal_inference('jng_gda15_bin', 'result')

print(f'Causal effect of 1k gold lead on result on adc: {est_adc1k_result.value}, and on jng: {est_jng1k_result.value}')

exit()
#Check the causal effect on damage participation 
df['adc_damageshare'] = df['adc_damageshare']*100
est_dmgshare_result, _ = causal_inference('adc_damageshare', 'result')
print(f'Estimate of the causal effect of adc damage share % on the result: {est_dmgshare_result.value}')

exit()
'''
Testing damageshare with scaling it to percentage points so time 100
Estimate of the causal effect of adc damage share % on the result: 0.002387178396946188
as expected its the same as dividing by 100 
One percent 
'''
exit()

#Causal effect of damage taken per minute on the result 
est_dmgtkn_result, _ = causal_inference('adc_damagetakenperminute', 'result')
print(f'Estimate of causal effect of the adc damage taken per minute on the result: {est_dmgtkn_result.value}')

#Causal effect of dpm on result and rating gain
est_dpm_result, _ = causal_inference('adc_dpm', 'result')
print(f'Estimate of effect of dpm on result (adc): {est_dpm_result.value}')

'''
This is without damageshare scaled
Estimate of the causal effect of adc damage share % on the result: 0.2387178396710013 
Estimate of causal effect of the adc damage taken per minute on the result: -0.001186170175543344
Estimate of efect of dpfm on result (adc): 0.001179002621388675
'''

#Prohibit data to 2025 and 2024 and see effect 
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df_recent = df[df["date"] >= "2024-01-01"] 


#Now estimate some stuff for only the recent data 
est_recent_adcgda_result, _ = causal_inference('adc_golddiffat15', 'result', data=df_recent)
print(f'Estimate for effect with recent data of adc gda15 on the result: {est_recent_adcgda_result.value}')

''' Estimate for effect with recent data of adc gda15 on the result: 6.422752108159457e-05 '''
exit()

#Test for causal effect of gold difference on adc on the match outcome as in the result and in the elo gain 
est_adc_gd_result, _  = causal_inference('adc_golddiffat15', 'result')
est_adc_gd_rating_gain, _  = causal_inference('adc_golddiffat15', 'rating_gain')
print(f'Estimate of the causal effect of gold difference at 15 on the adc role on the result: {est_adc_gd_result.value}, and on the ratinggain: {est_adc_gd_rating_gain.value}')


#Test for causal effect of the same just with jng
est_jng_gd_result, _ = causal_inference('jng_golddiffat15', 'result')
est_jng_gd_rating_gain, _ = causal_inference('jng_golddiffat15', 'rating_gain')
print(f'Estimate of the causal effect of gold difference at 15 on the jng role on the result: {est_jng_gd_result.value}, and on the ratinggain: {est_jng_gd_rating_gain.value}')

#Test for causal effect of damage to champions for each role and its effect on the result and the rating gain 
est_adc_dmg_result, _ = causal_inference('adc_damagetochampions', 'result')
est_adc_dmg_rating_gain, _ = causal_inference('adc_damagetochampions', 'rating_gain')
print(f'Estimate of the causal effect of damage to champions on the adc role on the result: {est_adc_dmg_result.value}, and on the ratinggain: {est_adc_dmg_rating_gain.value}')

est_jng_dmg_result, _ = causal_inference('jng_damagetochampions', 'result')
est_jng_dmg_rating_gain, _ = causal_inference('jng_damagetochampions', 'rating_gain')
print(f'Estimate of the causal effect of damage to champions on the jng role on the result: {est_jng_dmg_result.value}, and on the ratinggain: {est_jng_dmg_rating_gain.value}')

end_time = time.time()

print(f'Execution time: {end_time - start_time} seconds which are {(end_time - start_time) / 60}')


'''
Test 1 without standardized units
Estimate of the causal effect of gold difference at 15 on the adc role on the result: 0.0001499440458673229, and on the ratinggain: -2.4674218963127714e-05
Estimate of the causal effect of gold difference at 15 on the jng role on the result: 0.00018551874792693823, and on the ratinggain: -3.229135925478244e-05
Estimate of the causal effect of damage to champions on the adc role on the result: 1.4393510064492077e-05, and on the ratinggain: 3.7930299548460944e-07
Estimate of the causal effect of damage to champions on the jng role on the result: 1.2514964150889263e-05, and on the ratinggain: -7.863782881256665e-07
Execution time: 3675.6683530807495 seconds which are 61.261139218012495
'''

'''
Test 2 with stand. ...
Estimate of the causal effect of gold difference at 15 on the adc role on the result: 0.16731363360009155, and on the ratinggain: -0.006426749255533167
Estimate of the causal effect of gold difference at 15 on the jng role on the result: 0.16560686894963628, and on the ratinggain: -0.006637898035545599
Estimate of the causal effect of damage to champions on the adc role on the result: 0.13932528404265587, and on the ratinggain: -0.0017132273853039896
Estimate of the causal effect of damage to champions on the jng role on the result: 1.2816039808016821e-05, and on the ratinggain: -2.0611856425345812e-07
Execution time: 2897.914140701294 seconds which are 48.29856901168823
'''