import pandas as pd


df = pd.read_csv('./out/df.csv')
print(df[['mu_before', 'sigma_before', 'win_prob', 'rating_before', 'opp_rating_before', 'result', 'rating_after', 'rating_gain']].head())
exit()


from dowhy import CausalModel
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('./test/League_esport_dataset/out/df.csv')

causal_graph = '''
digraph {
    wardsplaced -> visionscore;
    wardskilled -> visionscore;
    wcpm -> visionscore;
    visionscore -> gamelength;
    win_prob -> gamelength;
    rating_before -> win_prob;
    win_prob -> golddiffat15;
    golddiffat15 -> result;
    result -> rating_after;
    rating_before -> rating_after;
    win_prob -> result;
    rating_after -> rating_gain;
    visionscore -> kills; 
    visionscore -> deaths; 
    visionscore -> assists;
    kills -> golddiffat15;
    side_adv -> win_prob;
    jng_golddiffat15 -> golddiffat15;
    adc_golddiffat15 -> golddiffat15;
}
'''

def causal_inference(treatment, outcome):
    #Create causal model
    model = CausalModel(
        data = df,
        treatment = treatment,
        outcome = outcome,
        graph = causal_graph
    )

    #Identify causal effects
    identified_estimand = model.identify_effect()

    #Estimate the causal effect using backdoor adjustment with linearregression
    estimate = model.estimate_effect(
        identified_estimand,
        method_name = 'backdoor.linear_regression'
    )

    refute_results = None
    #refute_results = model.refute_estimate(identified_estimand, estimate, method_name='random_common_cause')

    return estimate, refute_results


#Try for adc_golddiffat15 on results 
estimate1, _ = causal_inference('adc_golddiffat15', 'result')
print(estimate1.value)