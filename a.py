import pandas as pd 

df = pd.read_csv('./out/df.csv')

print(df['adc_dpm'])


sum_mean = df['damagetochampions'].mean()
adc = df['adc_damagetochampions'].mean()
jng = df['jng_damagetochampions'].mean()

print(f'ADC damage mean : {adc}')
print(f'Total damage mean: {sum_mean}')
print(f'jng damage mean : {jng}')
print(f'ADC does {(adc / sum_mean)*100}% of the damage and jng: {(jng/sum_mean)*100}%')

adc_dmg_eff_mean = df['adc_dmgefficiency'].mean()
jng_dmg_eff_mean = df['jng_dmgefficiency'].mean()

print(f'Damage per gold --> adc: {adc_dmg_eff_mean}, jng; {jng_dmg_eff_mean}')