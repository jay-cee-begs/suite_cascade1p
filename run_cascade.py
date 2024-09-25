import os, warnings
import sys
import glob
import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from CASCADE_functions import plots_and_basic_info, cascade_this
from functions_data_transformation import get_file_name_list, create_output_csv, csv_to_pickle, create_experiment_overview
import configurations


def main():
    # ## get the names of the deltaF files from the functions_data_transformation.py file

    deltaF = get_file_name_list(folder_path = configurations.main_folder, file_ending = "deltaF.npy")
    # if len(deltaF_files) == 0:
    #     deltaF_files = get_file_name_list(folder_path = configurations.main_folder, file_ending = "deltaF.npy")
    deltaF_files =get_file_name_list(folder_path = configurations.main_folder, file_ending = "deltaF.npy")
    try:

        predictions_deltaF_files = get_file_name_list(folder_path = configurations.main_folder, file_ending = "predictions_deltaF.npy") ## get the names of the predicted spike files
        if len(predictions_deltaF_files) == 0:
            predictions_deltaF_files = []
    except FileNotFoundError as e:
        print("Cascade Predictions do not exist yet")
        predictions_deltaF_files = []
    #TODO find a way to go through the directories and search for predictions deltaF; if for sample; != prediction file; calculate prediction file
    if len(predictions_deltaF_files) != len(deltaF_files):
        print("Cascade predictions for this dataset are missing, generating now...")
        for file in deltaF_files:
            plots_and_basic_info(file)
            cascade_this(file, configurations.nb_neurons)
        print("Done Generating Prediction Files")
    else:
        print("Cascade prediction files already exist")
    

    predictions_deltaF_files = get_file_name_list(folder_path = configurations.main_folder, file_ending = "predictions_deltaF.npy") ## get the names of the predicted spike files
    output_directories = get_file_name_list(folder_path = configurations.main_folder, file_ending = "samples")
    
    # for file, output in zip(predictions_deltaF_files, output_directories):
    #     histogram_total_estimated_spikes(file, output)
    # #TODO figure out how to compile group histograms
    # # for group in groups:
    # #     plot_group_histogram(group, predictions_deltaF_files)
    
    #     spike_maximum = get_max_spike_across_frames(predictions_deltaF_files)

    # for file, output in zip(predictions_deltaF_files, output_directories):
    #     plot_total_spikes_per_frame(file, spike_maximum, output)
    #     plot_average_spike_probability_per_frame(file, output)

    create_output_csv(configurations.main_folder, overwrite = True)
    csv_to_pickle(configurations.main_folder, overwrite = True)
    #TODO add an output for final_df for within python stuff
    # create_final_df(configurations.main_folder)
    create_experiment_overview(configurations.main_folder, configurations.groups)

if __name__ == "__main__":
    main()


"""To run:
activate cascade
import run_cascade
if __name__ == "__main__":
    run_cascade.main()
    
    """