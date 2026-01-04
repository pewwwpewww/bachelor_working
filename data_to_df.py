#This runs all python files in order to go from the data to final dataframe

import subprocess
import time

files = [
    'conv_csv.py',
    'openskill_impl.py',
    'conv_to_final.py'
]

total_time = 0

for file in files:
    print(f'Running the file: {file}')
    start_time = time.time()

    #Run the file
    subprocess.run(['python', file], check=True)
    end_time = time.time()
    print(f'Finished running {file}, with a runtime of {end_time - start_time} seconds')
    total_time += end_time - start_time

print(f'Finished running all files with total runtime of {total_time} seconds which is {total_time / 60} minutes ')