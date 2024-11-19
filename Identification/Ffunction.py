import numpy as np
import pandas as pd 
import os 

def fFunction(result_file, t, c):
    expected_header = "s p x y z fg_int bg_int"
    assert( os.path.isfile(result_file))
    
    # Read the content of the input file
    
    with open(result_file, 'r') as file:
        lines = file.readlines()

    # Find the index where the header starts
    header_start_index = None
    stopper_index = -1 # sometimes "[" isnt present
    for i, line in enumerate(lines):
        # Normalize line by stripping extra spaces
        normalized_line = ' '.join(line.split())

        if normalized_line == expected_header:
            header_start_index = i
        if "[" in line:
            stopper_index = i
            break

    if header_start_index is None:
        print('HEADER NOT FOUND!')
        return np.nan, np.nan
    
    # Extract header and data
    header_and_data = lines[header_start_index:stopper_index]

    # Split the header and data
    header = header_and_data[0].split()  # Extract and split the header
    data = [line.split() for line in header_and_data[1:]]  # Split each line of data

    # Directly create the DataFrame
    df = pd.DataFrame(data, columns=header)

    Ltotal = 0
    Lnoise = 0
    for index, row in df.iterrows():
        signal = row["fg_int"]
        noise = row["bg_int"]
        if(signal is not None and noise is not None and float(noise)!=0):
            if((float(signal)/float(noise)) < t):
                Lnoise+=1
            Ltotal+=1
        

    f_score = -1*Ltotal + c*Lnoise
    return f_score