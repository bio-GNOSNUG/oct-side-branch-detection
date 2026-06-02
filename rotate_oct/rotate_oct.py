from scipy.ndimage import rotate as rotate_image
import numpy as np
import pydicom
import re
import argparse
from pathlib import Path 

def dcm_to_npy(path):
    dcm = pydicom.dcmread(path)
    npy = dcm.pixel_array
    return npy

def extract_matching_labels(file_path):
 
    # Open the file with the appropriate encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines in the file
        lines = file.readlines()

    # Define a regular expression pattern to match the lines with <Point ... />
    pattern = re.compile(r'<Point (\d+)\s+(\d+)\s+([-+]?\d+\.\d+) \/>')

    # List to hold all extracted points
    oct_keypt_ids = []
    oct_keypt_rot = []
    for line in lines:
        match = pattern.search(line)
        if match:
            # Extract the three groups of numbers
            oct_keypt_ids.append(int(match.group(2)))
            oct_keypt_rot.append(float(match.group(3)))

    # Initialize variables to hold the second numbers from ResampleParameters1 and ResampleParameters2
    ivus_len = None
    oct_len = None

    # Define a regular expression pattern to match the ResampleParameters lines
    resample_pattern = re.compile(r'<ResampleParameters\d+>\s*(\d+)\s+(\d+)\s+\d+\s*</ResampleParameters\d+>')

    for line in lines:
        match = resample_pattern.search(line)
        if match:
            if ivus_len is None:
                ivus_len = int(match.group(2)) + 1
            else:
                oct_len = int(match.group(2)) + 1
                break  # Once both parameters are found, we can stop searching

    assert ivus_len is not None
    assert oct_len is not None
    assert ivus_len > 10
    assert oct_len > 10

    ivus_oct_rough_match = []
    ivus_oct_keypt_match = []
    # Initialize a flag to detect when to start processing lines
    process_lines = False
    # Define a regular expression pattern to match lines with two numbers
    pattern = re.compile(r'^(\d+)\s+(\d+)\s+\d+$')
    for line in lines:
        line = line.strip()  # Trim whitespace from the start and end of the line
        if line == '</ImageRotationPoints>':
            process_lines = True  # Set the flag to start processing the following lines
            continue  # Skip to the next iteration of the loop
        if process_lines:
            # Check if the line matches the pattern
            match = pattern.match(line)
            if match:
                first_num, second_num = match.groups()
                ivus_oct_rough_match.append([int(first_num), int(second_num)])
                
                if line.split(' ')[-1] == '1':
                    ivus_oct_keypt_match.append([int(first_num), int(second_num)])
            elif '<' in line and '>' in line:
                break  # Optional: Stop processing if another XML-like tag is encountered
                
    ivus_oct_keypt_match_many_to_one = []
    for oct_keypt in oct_keypt_ids:
        for ivus_id, oct_id in ivus_oct_rough_match:
            if oct_keypt == oct_id:
                ivus_oct_keypt_match_many_to_one.append([ivus_id, oct_id])
                #print(ivus_id, oct_id, oct_keypt)
            
    return np.array(oct_keypt_ids), np.array(oct_keypt_rot), np.array(ivus_oct_rough_match), np.array(ivus_oct_keypt_match), np.array(ivus_oct_keypt_match_many_to_one), ivus_len, oct_len

def interpolate_oct_angles(oct_keypt_ids, oct_keypt_rot, oct_len, cropped=False):
    
    oct_keypt_rot = adjust_angles(oct_keypt_rot)
    
    oct_keypt_rot_comb = np.stack([oct_keypt_ids, np.rad2deg(oct_keypt_rot)], axis=1)
    # append an extra points at start and end so that we can interpolate from the first rot for first and last OCT frame. 
    if not cropped:
        oct_keypt_rot_comb = np.concatenate(
            [np.array([0,oct_keypt_rot_comb[0,1]]).reshape(-1,2), 
             oct_keypt_rot_comb, 
             np.array([oct_len-1,oct_keypt_rot_comb[-1,1]]).reshape(-1,2)], axis=0)
    
    oct_interpolated_rot = interpolate_pts(oct_keypt_rot_comb)
    
    return oct_interpolated_rot, oct_keypt_rot

def adjust_angles(rad):
    # Convert radians to degrees
    deg = np.degrees(rad)
    
    for i in range(1, len(deg)):
        # Calculate the difference between the current angle and the previous one
        diff = deg[i] - deg[i-1]
        
        # If the difference is greater than 180 degrees, adjust the angle
        if diff > 180:
            deg[i] -= 360
        elif diff < -180:
            deg[i] += 360
            
    # Convert degrees back to radians
    adjusted_rad = np.radians(deg)
    return adjusted_rad

def interpolate_pts(points):

    # Extract x and y values
    x_values = points[:, 0]
    y_values = points[:, 1]

    # Create an array of x values for which you want interpolated y values
    x_interp = np.arange(x_values[0], x_values[-1] + 1)

    # Perform the linear interpolation
    y_interp = np.interp(x_interp, x_values, y_values)

    # If you want to see the results
    interpolated_points = np.column_stack((x_interp, y_interp))
    
    return interpolated_points


def process_one(dcm_path, txt_path, output_path):

    # for single vessel. e.g 'BASE-002-LAD_BL.txt'
    # e.g 'dicom/BASE-002-LAD_BL.dcm'
    oct_npy = dcm_to_npy(dcm_path)

    oct_keypt_ids, oct_keypt_rot, _, _, _, _, len_oct = extract_matching_labels(txt_path)

    oct_interpolated_rot, _ = interpolate_oct_angles(
        oct_keypt_ids,
        oct_keypt_rot,
        len_oct,
        cropped=False
    )

    rotated_frames = []

    for frame_id, frame in enumerate(oct_npy):

        rot = oct_interpolated_rot[frame_id][1]

        rotated_frames.append(
            rotate_image(frame, rot, reshape=False)
        )

    rotated_frames = np.array(rotated_frames)

    np.save(output_path, rotated_frames)

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--matching_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    matching_dir = Path(args.matching_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for dcm_path in dicom_dir.rglob("*.dcm"):

        name = dcm_path.stem
        txt_path = matching_dir / f"{name}.txt"

        if not txt_path.exists():
            print(f"Skipping {name}, no matching file")
            continue

        process_one(
            dcm_path,
            txt_path,
            output_dir / f"{name}_rotated.npy"
        )


    
