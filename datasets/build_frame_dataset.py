import os
import pandas as pd 
import argparse
from pathlib import Path 

def process_vessel(vessel_file):

    vessel_name = os.path.splitext(vessel_file)[0]
    vessel_name = vessel_name.removesuffix("_rotated")

    # Determine split

    # Create output folder

    # save frame file 

    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Must read in vessel information
    parser.add_argument("--vessels_summary_file ", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    df_path = Path(args.vessels_summary_file)
    vessels_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    df = pd.read_excel(df_path)
    df['Vessel_Name'] = df['Patient'] + '_' + df['Vessel']

    output_dir.mkdir(exist_ok=True, parents=True)






