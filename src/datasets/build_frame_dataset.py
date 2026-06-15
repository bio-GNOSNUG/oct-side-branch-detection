import os
import pandas as pd 
import numpy as np
import re
import argparse
from pathlib import Path 
from tqdm import tqdm

def process_vessel(vessel_dataset, vessel_file, vessel_name, save_dir):

    # Determine testset
    split = vessel_dataset[vessel_dataset['Vessel_Name']== vessel_name]['set'].unique().item()
    split = split.lower()

    # May be multiple file names per vessel, differing only by frame id
    vessel_id_name = vessel_dataset[vessel_dataset['Vessel_Name']== vessel_name]['filename'].iloc[0]
    vessel_id_name = vessel_id_name.removesuffix(".jpg")
    vessel_id_name = re.sub(r"_\d{4}$", "", vessel_id_name)

    if split is None:
        print(f"{vessel_name} not found in annotations")
        return
    
    try:
        vessel_frames = np.load(vessel_file, mmap_mode="r")

    except Exception as e:
        print(f'Could not load {vessel_file}: {e}')
        return
    
    save_folder = os.path.join(save_dir, 
                               split, 
                               vessel_id_name,
                               "oct_frames")
    
    os.makedirs(save_folder,exist_ok=True)

    # save frame file 
    for frame_id, frame in enumerate(vessel_frames):
        save_path = os.path.join(save_folder,f"{frame_id:04d}.npy")
        np.save(save_path, frame)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--vessels_summary_file", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    df_path = Path(args.vessels_summary_file)
    vessels_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(df_path)
    df['Vessel_Name'] = (df["filename"]
                         .str.removesuffix(".jpg")
                         .str[5:]
                         .str.replace(r"_\d{4}$", "", regex=True))

    output_dir.mkdir(exist_ok=True, parents=True)

    vessel_files = list(vessels_dir.rglob("*.npy"))
    missing = []

    for file in tqdm(vessel_files, desc="Processing vessels"):

        name = file.stem
        name = name.removesuffix("_rotated")

        match = df[df["Vessel_Name"] == name]

        if len(match) == 0:
            missing.append(name)
            print(f"{name} not found")


        process_vessel(vessel_dataset=df,
                        vessel_file=file, 
                        vessel_name=name,
                        save_dir= output_dir)

    print('Finished')
    print(f"Missing vessels: {missing}")





