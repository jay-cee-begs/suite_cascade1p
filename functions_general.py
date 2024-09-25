
import os, warnings
import numpy as np
import matplotlib.pyplot as plt
from configurations import FRAME_INTERVAL, main_folder
from scipy.signal import find_peaks, peak_prominences


def find_predicted_peaks(cascade_predictions, return_peaks = True):
    """User overview to find"""
    peaks_list = []
    amplitudes_list = []

    for cell in cascade_predictions:
    
        peaks, _ = find_peaks(cell, distance = 5)  ## adjust !!!
        amplitudes = cell[peaks]

        peaks_list.append(peaks)
        amplitudes_list.append(amplitudes)


    if return_peaks:
        return peaks_list
    else:
        return amplitudes_list

def return_baseline_F(F, Fneu):
    """Returns the calculated baseline fluorescence for each cell and appends to the final dictionary"""
    savepath = rf"{F}".replace("\\F.npy","") ## make savepath original folder, indicates where deltaF.npy is saved


    baseline_F = []
    for f, fneu in zip(F, Fneu):
        corrected_trace = f - (0.7*fneu) ## neuropil correction

        amount = int(0.125*len(corrected_trace))
        middle = 0.5*len(corrected_trace)
        F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], 
                    corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
        F_baseline = np.mean(F_sample)
        baseline_F.append(F_baseline)
    baseline_F = np.array(baseline_F)
    # baseline_F = np.mean(baseline_F)
    # np.save(f"{savepath}/F_baseline.npy", baseline_F, allow_pickle=True)

    return baseline_F


def basic_stats_per_cell(predictions_file):
    '''returns cell_means, cell_means, cell_cvs for all cells in file, 
    mean/SD/cv based on predicited spikes for this cell
    also returns time_stamp_mean, time_stamp_sds, and time_stamp_cvs for each
    frame besides the first and last 32 frames'''
    # cell_means = []
    cell_sds = []
    cell_cvs = []
    cell_instant_spike_rate = []
    time_stamp_mean = []
    time_stamp_sds = []
    time_stamp_cvs =  []

    frames = predictions_file.shape[1] #Number of columns
    cells = predictions_file.shape[0] #Number of rows
    sum = []
    for cell in predictions_file:
        mean=np.nanmean(cell)
        sum.append(np.nansum(cell))
        if mean > 0:

            cell_instant_spike_rate.append(mean/FRAME_INTERVAL)
        # cell_means.append(mean)
            sd=np.nanstd(cell)
            cell_sds.append(sd)
            # if mean != 0:
            cv_cell = sd/mean
            # else:
            #     cv_cell = np.nan ## cells that don't fire (--> mean spike probability 0) --> makes cv nan
            cell_cvs.append(cv_cell)
        else:
            cell_sds.append(np.nan)
            cell_cvs.append(np.nan)
    
    for col_idx in range(frames):
        col_data = predictions_file[:, col_idx]
        col_sum = np.nansum(col_data)
        col_mean = col_sum / cells #manually calculating the mean because of errors in np.nanmean()
        col_sd = np.nanstd(col_data)
        time_stamp_mean.append(col_mean)
        time_stamp_sds.append(col_sd)

        if col_mean != 0:
            cv_time = col_sd / col_mean
        else:
            cv_time = np.nan
        time_stamp_cvs.append(cv_time)
        
    # Compute averages over frames for each row (cell)
    # avg_cell_means = np.nanmean(cell_means)
    avg_instantaneous_spike_rate = np.nanmean(cell_instant_spike_rate)
    avg_cell_sds = np.nanmean(cell_sds)
    avg_cell_cvs = np.nanmean(cell_cvs)
    
    # Compute averages over cells for each column (time stamp)
    avg_time_stamp_mean = np.nanmean(time_stamp_mean)
    avg_time_stamp_sds = np.nanmean(time_stamp_sds)
    avg_time_stamp_cvs = np.nanmean(time_stamp_cvs)
    
    return avg_instantaneous_spike_rate, avg_cell_sds, avg_cell_cvs, avg_time_stamp_mean, avg_time_stamp_sds, avg_time_stamp_cvs

def basic_estimated_stats_per_cell(predictions_file):
    '''returns means, SDs, cvs for all cells in file, mean/SD/cv based on predicited spikes for this cell'''
    means = []
    sds = []
    cvs = []
    for cell in predictions_file:
        mean=np.nanmean(cell)
        means.append(mean)
        sd=np.nanstd(cell)
        sds.append(sd)
        if mean != 0:
            cv_cell = sd/mean
        else:
            cv_cell = np.nan ## cells that don't fire (--> mean spike probability 0) --> makes cv nan
        cvs.append(cv_cell)
    return means, sds, cvs
 
def summed_spike_probs_per_cell(prediction_deltaF_file):

    summed_spike_probs_cell = []

    for cell in prediction_deltaF_file:
        summed_spike_probs_cell.append(np.nansum(cell))

    return summed_spike_probs_cell


def calculate_deltaF(F_file):

    savepath = rf"{F_file}".replace("\\F.npy","") ## make savepath original folder, indicates where deltaF.npy is saved

    F = np.load(rf"{F_file}", allow_pickle=True)

    Fneu = np.load(rf"{F_file[:-4]}"+"neu.npy", allow_pickle=True)

    deltaF= []

    for f, fneu in zip(F, Fneu):
        corrected_trace = f - (0.7*fneu) ## neuropil correction

        amount = int(0.125*len(corrected_trace))
        middle = 0.5*len(corrected_trace)
        F_sample = (np.concatenate((corrected_trace[0:amount], corrected_trace[int(middle-amount/2):int(middle+amount/2)], 
                    corrected_trace[len(corrected_trace)-amount:len(corrected_trace)])))  #dynamically chooses beginning, middle, end 12.5%, changeable
        F_baseline = np.mean(F_sample)
        deltaF.append((corrected_trace-F_baseline)/F_baseline)

    deltaF = np.array(deltaF)

    np.save(f"{savepath}/deltaF.npy", deltaF, allow_pickle=True)

    print(f"delta F calculated for {F_file[len(main_folder)+1:-21]}")

    csv_filename = f"{F_file[len(main_folder)+1:-21]}".replace("\\", "-") ## prevents backslahes being replaced in rest of code

    if not os.path.exists(main_folder + r'\csv_files_deltaF'): ## creates directory if it doesn't exist
        os.mkdir(main_folder + r'\csv_files_deltaF')

    np.savetxt(f"{main_folder}/csv_files_deltaF/{csv_filename}.csv", deltaF, delimiter=";") ### can be commented out if you don't want to save deltaF as .csv files (additionally to .npy)

    ## if done by pandas, version needs to be checked, np.savetxt might be enough anyways ##
    # df = pd.DataFrame(deltaF)
    # df.to_csv(f"{main_folder}"+"/csv_files/"+f"{file[len(main_folder)+1:-21]}"+".csv", index = False, header = False)

    print(f"delta F traces saved as deltaF.npy under {savepath}\n")


