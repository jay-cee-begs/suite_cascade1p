
import os
import pandas as pd
import numpy as np
import functions_general
from functions_general import calculate_deltaF, basic_stats_per_cell, basic_estimated_stats_per_cell, summed_spike_probs_per_cell, return_baseline_F
from configurations import groups, main_folder, EXPERIMENT_DURATION, FRAME_INTERVAL, BIN_WIDTH, FILTER_NEURONS
import functions_plots
from functions_plots import getImg, getStats, dispPlot

SUITE2P_STRUCTURE = {
    "F": ["suite2p", "plane0", "F.npy"],
    "Fneu": ["suite2p", "plane0", "Fneu.npy"],
    "spks": ["suite2p", "plane0", "spks.npy"],
    "stat": ["suite2p", "plane0", "stat.npy"],
    "iscell": ["suite2p", "plane0", "iscell.npy"],
    "deltaF": ["suite2p", "plane0", "deltaF.npy"],
    "ops":["suite2p", "plane0", "ops.npy"],
    "cascade_predictions": ["suite2p", "plane0", "predictions_deltaF.npy"]
}

def load_npy_array(npy_path):
    return np.load(npy_path, allow_pickle=True) #functionally equivalent to np.load(npy_array) but iterable; w/ Pickle

def load_npy_df(npy_path):
    return pd.DataFrame(np.load(npy_path, allow_pickle=True)) #load suite2p outputs as pandas dataframe

def check_deltaF(folder_name_list):
    for folder in folder_name_list:
        location = os.path.join(folder, *SUITE2P_STRUCTURE["deltaF"])
        if os.path.exists(location):
            continue
        else:
            calculate_deltaF(location.replace("deltaF.npy","F.npy"))
            if os.path.exists(location):
                continue
            else:
                print("something went wrong, please calculate delta F manually by inserting the following code above: \n F_files = get_file_name_list(folder_path = main_folder, file_ending = 'F.npy') \n for file in F_files: calculate_deltaF(file)")

def get_file_name_list(folder_path, file_ending, supress_printing = False): ## accounts for possible errors if deltaF files have been created before
    file_names = []
    other_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_ending=="F.npy" and file.endswith(file_ending) and not file.endswith("deltaF.npy"):
                    file_names.append(os.path.join(root, file))
            elif file_ending=="deltaF.npy" and file.endswith(file_ending) and not file.endswith("predictions_deltaF.npy"):
                    file_names.append(os.path.join(root, file))
            elif file_ending=="predictions_deltaF.npy" and file.endswith(file_ending):
                 file_names.append(os.path.join(root, file))
            elif file_ending=="samples":
                if file.endswith("F.npy") and not file.endswith("deltaF.npy"):
                    file_names.append(os.path.join(root, file)[:-21])
            else:
                 if file.endswith(file_ending): other_files.append(os.path.join(root, file))
    if file_ending=="F.npy" or file_ending=="deltaF.npy" or file_ending=="predictions_deltaF.npy":
        if not supress_printing:
            print(f"{len(file_names)} {file_ending} files found:")
            print(file_names)
        return file_names
    elif file_ending=="samples":
        check_deltaF(file_names)  #checks if deltaf exists, else calculates it
        if not supress_printing:
            print(f"{len(file_names)} folders containing {file_ending} found:")
            print(file_names)
        return file_names
    else:
        print("Is the file ending spelled right?")
        return other_files

def get_sample_dict(main_folder):
    """returns a dictionary of all wells and the corresponding sample/replicate, the samples are sorted by date, everything sampled on the first date is then sample1, on the second date sample2, etc."""
    well_folders = get_file_name_list(main_folder, "samples", supress_printing = True)
    date_list= []
    sample_dict = {}
    for well in well_folders:
        date_list.append(os.path.basename(well)[0:6]) ## append dates; should change if the date is not in the beginning of the file name usually [:6]
    distinct_dates = [i for i in set(date_list)]
    distinct_dates.sort(key=lambda x: int(x))
 
    for i1 in range(len(well_folders)):
        for i2, date in enumerate(distinct_dates):
            if date in well_folders[i1]: # if date in list
                sample_dict[well_folders[i1]]=f"sample_{i2+1}"
    return sample_dict

def create_df(suite2p_dict): ## creates df structure for single sample (e.g. well_x) csv file, input is dict resulting from load_suite2p_paths
    """this is the principle function in which we will create our .csv file structure; and where we will actually use
        our detector functions for spike detection and amplitude extraction"""
 
    ## spike_amplitudes = find_predicted_peaks(suite2p_dict["cascade_predictions"], return_peaks = False) ## removed
    # spikes_per_neuron = find_predicted_peaks(suite2p_dict["cascade_predictions"]) ## removed
 
    estimated_spike_total = np.array(summed_spike_probs_per_cell(suite2p_dict["cascade_predictions"]))
    # estimated_spike_std = np.std(np.array(summed_spike_probs_per_cell(suite2p_dict["cascade_predictions"])))
    basic_cell_stats = basic_estimated_stats_per_cell(suite2p_dict['cascade_predictions'])
    F_baseline = return_baseline_F(suite2p_dict["F"], suite2p_dict["Fneu"])
    avg_instantaneous_spike_rate, avg_cell_sds, avg_cell_cvs, avg_time_stamp_mean, avg_time_stamp_sds, avg_time_stamp_cvs = basic_stats_per_cell(suite2p_dict["cascade_predictions"])
   
    ## all columns of created csv below ##
 
    df = pd.DataFrame({"IsUsed": suite2p_dict["IsUsed"],
                    #    "ImgShape": ImgShape,
                    #    "npix": suite2p_dict["stat"]["npix"],
                    #    "xpix": suite2p_dict["stat"]["xpix"],
                    #    "ypix": suite2p_dict["stat"]["ypix"],
                    #    "Skew": suite2p_dict["stat"]["skew"],
                       "Baseline_F": F_baseline,
                       "EstimatedSpikes": estimated_spike_total,
                       "SD_Estimated_Spks":basic_cell_stats[1],
                       "cv_Estimated_Spks":basic_cell_stats[2],
                       "Total Frames": len(suite2p_dict["F"].T)-64,
                       "SpikesFreq": avg_instantaneous_spike_rate, ## -64 because first and last entries in cascade are NaN, thus not considered in estimated spikes)
                    #    "Baseline_F": F_baseline,
                    #    "Spikes_std": avg_cell_sds,
                    #    "Spikes_CV": avg_cell_cvs, 
                       "group": suite2p_dict["Group"],
                       "dataset":suite2p_dict["sample"],
                       "file_name": suite2p_dict["file_name"]})
    #if use_suite2p_iscell == True:
    #else:
        # continue
    df["IsUsed"] = df["EstimatedSpikes"] > 0

    df.index.set_names("NeuronID", inplace=True)
    return df

def load_suite2p_paths(data_folder, groups, main_folder, use_iscell=False):  ## creates a dictionary for the suite2p paths in the given data folder (e.g.: folder for well_x)
    """here we define our suite2p dictionary from the SUITE2P_STRUCTURE...see above"""
    suite2p_dict = {
        "F": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["F"])),
        "Fneu": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["Fneu"])),
        "stat": load_npy_df(os.path.join(data_folder, *SUITE2P_STRUCTURE["stat"]))[0].apply(pd.Series),
        "ops": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["ops"])).item(),
        "cascade_predictions": load_npy_array(os.path.join(data_folder, *SUITE2P_STRUCTURE["cascade_predictions"])),
    }
 
    if use_iscell == False:
        suite2p_dict["IsUsed"] = [(suite2p_dict["stat"]["skew"] >= 1)] 
        suite2p_dict["IsUsed"] = pd.DataFrame(suite2p_dict["IsUsed"]).iloc[:,0:].values.T
        suite2p_dict["IsUsed"] = np.squeeze(suite2p_dict["IsUsed"])
    else:
        suite2p_dict["IsUsed"] = load_npy_df(os.path.join(data_folder, *SUITE2P_STRUCTURE["iscell"]))[0].astype(bool)
 #TODO make sure that changing "path" to "data_folder" for using IsCell natively will still work
    if not groups:
        raise ValueError("The 'groups' list is empty. Please provide valid group names.")

    print(f"Data folder: {data_folder}")
    print(f"Groups: {groups}")
    print(f"Main folder: {main_folder}")
    found_group = False
    for group in groups: ## creates the group column based on groups list from configurations file
        if (str(group)) in data_folder:
            group_name = group.split(main_folder)[-1].strip("\\/")
            suite2p_dict["Group"] = group_name
            found_group = True
            print(f"Assigned Group: {suite2p_dict['Group']}")
    
    # debugging
    if "IsUsed" not in suite2p_dict:
        raise KeyError ("'IsUsed' was not defined correctly either")
    if "Group" not in suite2p_dict:
        raise KeyError("'Group' key not found in suite2p_dict.")
    if not found_group:
        raise KeyError(f"No group found in the data_folder path: {data_folder}")

    sample_dict = get_sample_dict(main_folder) ## creates the sample number dict
   
    suite2p_dict["sample"] = sample_dict[data_folder]  ## gets the sample number for the corresponding well folder from the sample dict
 
    suite2p_dict["file_name"] = str(os.path.join(data_folder, *SUITE2P_STRUCTURE["cascade_predictions"]))
 
    return suite2p_dict


def create_output_csv(input_path, overwrite=False, check_for_iscell=False): ## creates output csv for all wells and saves them in .csv folder
    """This will create .csv files for each video loaded from out data fram function below.
        The structure will consist of columns that list: "Amplitudes": spike_amplitudes})
        
        col1: ROI #, col2: IsUsed (from iscell.npy); boolean, col3: Skew (from stats.npy); could be replaced with any 
        stat >> compactness, col3: spike frames (relative to input frames), col4: amplitude of each spike detected measured 
        from the baseline (the median of each trace)"""
    
    well_folders = get_file_name_list(input_path, "samples", supress_printing = True)

    output_path = input_path+r"\csv_files"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for folder in well_folders:
        output_directory = (os.path.relpath(folder, input_path)).replace("\\", "-")
        translated_path = os.path.join(output_path, f"{output_directory}.csv")
        if os.path.exists(translated_path) and not overwrite:
            print(f"CSV file {translated_path} already exists!")
            continue

        suite2p_dict = load_suite2p_paths(folder, groups, input_path, use_iscell=check_for_iscell)

        # output_df = create_df(load_suite2p_paths(folder, groups, input_path))
        output_df = create_df(suite2p_dict)
    

        output_df.to_csv(translated_path)
        print(f"csv created for {folder}")

        suite2p_dict = load_suite2p_paths(folder, groups, input_path, use_iscell=check_for_iscell)
        ops = suite2p_dict["ops"]
        Img = getImg(ops)
        scatters, nid2idx, nid2idx_rejected, pixel2neuron = getStats(suite2p_dict["stat"], Img.shape, output_df)

        image_save_path = os.path.join(input_path, f"{folder}_plot.png") #TODO explore changing "input path" to "folder" to save the processing in the same 
        dispPlot(Img, scatters, nid2idx, nid2idx_rejected, pixel2neuron, suite2p_dict["F"], suite2p_dict["Fneu"], image_save_path)

    print(f"{len(well_folders)} .csv files were saved under {main_folder+r'/csv_files'}")

## create .pkl and final df ##
def get_pkl_file_name_list(folder_path): 
    pkl_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files


def list_all_files_of_type(input_path, filetype):
    return [os.path.join(input_path, path) for path in os.listdir(input_path) if path.endswith(filetype)]

def csv_to_pickle(main_folder, overwrite=True):
    """creates pkl, output -> main_folder+r'\pkl_files'"""
    csv_files = list_all_files_of_type(main_folder+r"/csv_files", ".csv")
    print((csv_files))
    output_path = main_folder+r"/pkl_files"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in csv_files:
        df = pd.read_csv(file)
        pkl_path = os.path.join(output_path, 
                                        f"{os.path.basename(file[:-4])}"
                                        f"Dur{int(EXPERIMENT_DURATION)}s"
                                        f"Int{int(FRAME_INTERVAL*1000)}ms"
                                        f"Bin{int(BIN_WIDTH*1000)}ms"
                                            + ("_filtered" if FILTER_NEURONS else "") +
                                        ".pkl")
        if os.path.exists(pkl_path) and not overwrite:
            print(f"Processed file {pkl_path} already exists!")
            continue

        df.to_pickle(pkl_path)
        print(f"{pkl_path} created")
    print(f".pkl files saved under {main_folder+r'/pkl_files'}")

def create_final_df(main_folder):
    """ creates the final datat frame (all the wells in one dataframe) from which further analyses can be done"""
    pkl_files = get_pkl_file_name_list(main_folder)
    df_list = []
    for file in pkl_files:
        df = pd.read_pickle(file)
        df_list.append(df)
    final_df = pd.concat(df_list, ignore_index=True)
    if len(get_file_name_list(main_folder, "samples")) != len(pkl_files):
        raise Exception("The amount of .pkl files doesn't match the amount of samples, please delete all .csv and .pkl files and start over") ##Check this exception later
    return final_df
    ##alternative df from cell_stats dict, add previous functions back in then

def calculate_iqr_and_outliers(data):
    """Calculates IQR and identifies outliers in the data."""
    try:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound))
    except IndexError as e:
        Q1, Q3, IQR, lower_bound, upper_bound, outliers = np.nan() 
    return IQR, len(outliers)

def get_unique_prefixes(group_names, prefix_length=3):
    return {name[:prefix_length] for name in group_names}

def create_experiment_overview(main_folder, groups):
    dictionary_list = []
    
    for group in groups:
        groups_predictions_deltaF_files = get_file_name_list(folder_path=group, file_ending="predictions_deltaF.npy", supress_printing=True)
        
        for file in groups_predictions_deltaF_files:
            array = np.load(rf"{file}", allow_pickle=True)
            avg_cell_instantaneous_spike_rate, cell_sds, cell_cvs, time_stamp_means, time_stamp_sds, time_stamp_cvs = basic_stats_per_cell(array)
            
            active_neurons = sum(np.nansum(row) > 0 for row in array)
            neuron_count = len(array)
            estimated_spikes = [np.nansum(row) for row in array]

            # Load F, Fneu arrays
            F_file = file.replace('predictions_deltaF.npy', 'F.npy')
            Fneu_file = file.replace('predictions_deltaF.npy', 'Fneu.npy')
            F = np.load(rf"{F_file}", allow_pickle=True)
            Fneu = np.load(rf"{Fneu_file}", allow_pickle=True)
            baseline_F = return_baseline_F(F, Fneu)

            # Separate and average the baseline fluorescence
            inactive_baseline = [cell for row, cell in zip(array, baseline_F) if np.nansum(row) == 0]
            active_baseline = [cell for row, cell in zip(array, baseline_F) if np.nansum(row) > 0]

            avg_inactive_cell = np.nanmean(inactive_baseline)
            avg_active_cell = np.nanmean(active_baseline)
            total_estimated_spikes = round(sum(estimated_spikes), 2)

            dictionary_list.append({
                'Prediction_File': file[len(main_folder)+1:], 
                'Neuron_Count': neuron_count,
                'Active_Neuron_Count': active_neurons, 
                'Active_Neuron_Proportion': round(active_neurons/neuron_count * 100, 2),
                'Active_Neuron_F0': avg_active_cell,
                "Inactive_Neuron_F0": avg_inactive_cell,
                'Total_Estimated_Spikes': total_estimated_spikes, 
                "Total_Estimated_Spikes_proportion_scaled": total_estimated_spikes / (active_neurons/neuron_count),
                'Avg_Estimated_Spikes_per_cell': total_estimated_spikes / active_neurons,
                "SC_Avg_Instantaneous_Firing_Rate(Hz)": avg_cell_instantaneous_spike_rate,
                "Instantaneous_Spikes_CV": cell_cvs,
                "Network_Framewise_Avg_Instantaneous_Firing_Freq": time_stamp_means,
                "Network_Framewise_CV": time_stamp_cvs,
                "Group": group[len(main_folder)+1:]
            })
    
    # Create DataFrame from dictionary list
    df = pd.DataFrame(dictionary_list)

    unique_prefixes = get_unique_prefixes(df['Group'])

    # Create a dynamic categorization function
    def categorize_time_point(group_name):
        for prefix in unique_prefixes:
            if group_name.startswith(prefix):
                return prefix
        return 'N/A'

    # Add a new column 'Time_Point' based on the unique prefixes
    df['Time_Point'] = df['Group'].apply(categorize_time_point)

    # Ensure 'N/A' categories are handled
    df = df[df['Time_Point'] != 'N/A']
    # Calculate summary statistics for each unique group
    summary_stats = df.groupby(['Group', 'Time_Point']).agg({
        'Neuron_Count': ['mean', 'std','median'],
        'Active_Neuron_Count': ['mean', 'std','median'],
        'Active_Neuron_Proportion': ['mean', 'std','median'],
        'Active_Neuron_F0': ['mean', 'std','median'],
        'Inactive_Neuron_F0': ['mean', 'std','median'],
        'Total_Estimated_Spikes': ['mean', 'std','median'],
        'Total_Estimated_Spikes_proportion_scaled': ['mean', 'std','median'],
        'Avg_Estimated_Spikes_per_cell': ['mean', 'std','median'],
        "SC_Avg_Instantaneous_Firing_Rate(Hz)": ['mean', 'std','median'],
        "Instantaneous_Spikes_CV": ['mean', 'std','median'],
        "Network_Framewise_Avg_Instantaneous_Firing_Freq": ['mean', 'std','median'],
        "Network_Framewise_CV": ['mean', 'std','median']
    })

    # Save both raw data and summary statistics to CSV
    df.to_csv(main_folder + r'\new_experiment_summary.csv', index=False)
    summary_stats.to_pickle(main_folder + r'\summary_stats.pkl')
    summary_stats.to_csv(main_folder + r'\summary_stats.csv', index = True)

    return df, summary_stats
