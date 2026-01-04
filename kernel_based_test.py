import pandas as pd 
import numpy as np 
import time
from causallearn.utils.cit import CIT

start_time = time.time()

df = pd.read_csv('./out/df.csv')

#Clean the dataframe
df = df.sample(n=4000, random_state=42)
#a = np.random.randn(1000, 1)
#b = np.random.randn(1000, 1)
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.to_list()
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

run_test('gamelength', 'visionscore', [])
