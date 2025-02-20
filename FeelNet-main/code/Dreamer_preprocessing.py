import scipy.io as sio
import numpy as np
import os


# Define a function to load DREAMER.mat and save the required EEG data and labels as txt files
def convert_dreamer_mat_to_txt(mat_file_path, output_dir):
    # Load the .mat file
    mat = sio.loadmat(mat_file_path)

    # Extract relevant data
    # Assuming the structure of the loaded mat file contains 'DREAMER' key
    dreamer_data = mat.get('DREAMER')

    if dreamer_data is None:
        raise ValueError("The provided .mat file does not contain 'DREAMER' key.")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_subjects = len(dreamer_data['Data'][0][0][0])
    for subject_idx in range(num_subjects):
        subject_data = dreamer_data['Data'][0][0][0][subject_idx][0]

        # Create a directory for each subject
        person_dir = os.path.join(output_dir, f'person{subject_idx + 1}')
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        # Extract EEG data (all clips combined) and save it
        eeg_data = subject_data['EEG'][0][0]['stimuli'][0]
        combined_eeg_data = []
        for clip_idx in range(len(eeg_data)):
            combined_eeg_data.append(eeg_data[clip_idx][0])
        combined_eeg_data = np.vstack(combined_eeg_data)
        eeg_file_path = os.path.join(person_dir, f'stimuli_eeg{subject_idx + 1}.txt')
        np.savetxt(eeg_file_path, combined_eeg_data, fmt='%0.4f', delimiter='	')

        # Extract emotion ratings (valence, arousal, dominance)
        valence = subject_data['ScoreValence'][0]
        arousal = subject_data['ScoreArousal'][0]
        dominance = subject_data['ScoreDominance'][0]

        # Save valence, arousal, dominance to respective text files in each person's directory
        valence_file_path = os.path.join(person_dir, 'valence.txt')
        np.savetxt(valence_file_path, valence, fmt='%0.4f', delimiter='	')

        arousal_file_path = os.path.join(person_dir, 'arousal.txt')
        np.savetxt(arousal_file_path, arousal, fmt='%0.4f', delimiter='	')

        dominance_file_path = os.path.join(person_dir, 'dominance.txt')
        np.savetxt(dominance_file_path, dominance, fmt='%0.4f', delimiter='	')

    print(f'Saved all data to {output_dir}')


# Example usage
mat_file_path = './dreamer/DREAMER.mat'  # replace with the actual path to your DREAMER.mat file
output_dir = './dreamer_txt_output'  # directory to save txt files
convert_dreamer_mat_to_txt(mat_file_path, output_dir)

