import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage import binary_dilation, binary_fill_holes
import scipy.stats as stats
import pickle
from PIL import Image
import seaborn as sns #needed for aggregated feature plots
# import pynapple as nap #TODO if you need Pynapple plots, you cannot use alongside cascade as it will break the code
from configurations import *

def random_individual_cell_histograms(deltaF_file, plot_number):
    ## for individual cells, random sample of plot_number, (can also be set to randoms sample of size plot_number, i this case use code below to calculate plot number and then pass it to function) ##
    ### ROI_number = len(np.load(file)) ## needs to be connected with plot number below if we always want to show fixed percentage of all possible histograms
    ### plot_number = int(0.05*ROI_number) # plots random 5% of all cells
    ### if plot_number <4: plot_number = 4
    
    array = np.load(rf"{deltaF_file}")
    sample = random.sample(range(0, len(array)), plot_number)
    for i in sample: ## alterantive i in range(len(array)) to plot all
      plt.figure(figsize=(5,5))
      plt.hist(array[i], density=True, bins=200)
      plt.title(f'Histogram df/F fluorescence cell {i}')
      plt.show()

def deltaF_histogram_across_cells(deltaF_file):
    array = np.load(rf"{deltaF_file}")
    list = array.flatten()
    list_cleaned = [x for x in list if not np.isnan(x)]
    plt.figure(figsize=(5,5))
    plt.hist(list_cleaned, density=True, bins=200)
    plt.title(f'Histogram df/F {deltaF_file[len(main_folder)+1:]}')
    plt.show()

def histogram_total_estimated_spikes(prediction_deltaF_file, output_directory):
    array = np.load(rf"{prediction_deltaF_file}")
    print(f"\n{prediction_deltaF_file}\nNumber of neurons in dataset: {len(array)}")
    estimated_spikes = []
    for i in range(len(array)):
        estimated_spikes.append(np.nansum(array[i]))
    print(f"For {prediction_deltaF_file[len(main_folder)+1:-38]} {int(sum(estimated_spikes))} spikes were predicted in total")
    plt.figure(figsize=(5,5))
    plt.hist(estimated_spikes, bins=50, color = 'm')
    plt.xlabel("Number of predicted estimated spikes")
    plt.ylabel("Number of Neurons")
    plt.title(f'Total number of predicted spikes') # \n {prediction_deltaF_file[len(main_folder)+1:-38]}
    plt.text(0.65, 0.9, f"Total Spikes \nPredicted: {int(sum(estimated_spikes))}", transform=plt.gca().transAxes)
    figure_output_path = os.path.join(output_directory, 'spks_histogram.png')
    plt.savefig(figure_output_path, bbox_inches = 'tight')
    print(f'Well Histograms for estimated spikes saved under {figure_output_path}')
    #plt.show()

def plot_group_histogram(group, predictions_deltaF_files): ## plots histograms of total spikes per neuron for each group, possible to add a third group
    group_arrays = []
    estimated_spikes = []
    for file in predictions_deltaF_files:
        if str(group) in file:
            array = np.load(rf"{file}")
            group_arrays.append(array)        
    print(f"{len(group_arrays)} files found for group {group[len(main_folder)+1:]}")
    group_array = np.concatenate(group_arrays, axis=0)
    print(f"{len(group_array)} total neurons in {group}")
    for i in range(len(group_array)):
        estimated_spikes.append(np.nansum(group_array[i]))
    print(f"For group {group[len(main_folder)+1:]} {int(sum(estimated_spikes))} spikes were predicted in total")
    plt.figure(figsize=(5,5))
    plt.hist(estimated_spikes, bins=50, density=True)
    plt.ylim(0, 0.5)
    plt.xlim(0,100) ## maybe make dynamic (get_max_spike_across_frames() could be useful or slight alteration), so it's the same for all groups
    plt.title(f'Histogram estimated total number of spikes, {group[len(main_folder)+1:]}') ## y proprtion of neurons, x number of events, title estimated distribution total spike number
    plt.xlabel("Number of estimated spikes")
    group_name = group[len(main_folder) + 1]
    save_path = os.path.join(main_folder, f'histogram_{group_name}.png')
    plt.savefig(save_path)
    plt.show()

    ## add titles axes labeling etc.

def single_cell_peak_plotting(input_f, title): ## input f needs to be single cell
    threshold = np.nanmedian(input_f)+np.nanstd(input_f)
    peaks, _ = find_peaks(input_f, distance = 5, height = threshold)
    plt.figure(figsize=(5,5))
    plt.plot(input_f)
    plt.plot(peaks, input_f[peaks], "x")
    plt.plot(np.full_like(input_f, threshold), "--",color = "grey") ## height in find_peaks
    plt.plot(np.full_like(input_f, np.nanmean(input_f)), "--", color = 'r')
    plt.title(title)
    plt.xlabel("frames")
    plt.show()

    ## not sure how useful, maybe calculate peaks by AUC??? ##

def visualization_process_single_cell(F_files, deltaF_files, predictions_deltaF_files, cells_plotted):
    for file_number in range(len(predictions_deltaF_files)):
        ## try with corrected trace too ??
        prediction_array = np.load(rf"{predictions_deltaF_files[file_number]}", allow_pickle=True)
        rawF_array = np.load(rf"{F_files[file_number]}", allow_pickle=True)
        deltaF_array = np.load(rf"{deltaF_files[file_number]}", allow_pickle=True)
        sample = np.random.randint(0,len(prediction_array), cells_plotted)
        for cell in sample:
            print(f"raw fluorescence {predictions_deltaF_files[file_number][len(main_folder)+1:-38]}, cell {cell}")
            single_cell_peak_plotting(rawF_array[cell], f"Raw fluorescence {predictions_deltaF_files[file_number][-45:-38]}, cell {cell}")
            print(f"delta F {predictions_deltaF_files[file_number][len(main_folder)+1:-38]}, cell {cell}")
            single_cell_peak_plotting(deltaF_array[cell], f"DeltaF {predictions_deltaF_files[file_number][-45:-38]}, cell {cell}")
            print(f"cascade predictions {predictions_deltaF_files[file_number][len(main_folder)+1:-38]}, cell {cell}")
            single_cell_peak_plotting(prediction_array[cell], f"Cascade predictions {predictions_deltaF_files[file_number][-45:-38]}, cell {cell}")
## maybe move those not used anymore to unused to other functions script

def get_max_spike_across_frames(predictions_deltaF_file_list):
    total_list=[]
    for file in predictions_deltaF_file_list:
        prediction_array = np.load(rf"{file}", allow_pickle=True)
        sum_rows = np.nansum(prediction_array, axis=0)
        total_list.extend(sum_rows)
    return(max(total_list))
## maybe move cause not related to plotting

def plot_total_spikes_per_frame(prediction_deltaF_file, max_spikes_all_samples, output_directory):
    '''calculates the total spikes across whole culture at certain time point \n the first input is a prediction_deltaF_file, the second input determines the scaling of the y axis and can be calculated by get_max_spikes_across_data()'''
    prediction_array = np.load(rf"{prediction_deltaF_file}", allow_pickle=True)
    sum_rows = np.nansum(prediction_array, axis=0)
    avg_rows = np.nanmean(prediction_array, axis = 0)
    plt.figure(figsize=(10,5))
    plt.plot(sum_rows, color = "green")
    plt.plot(np.full_like(sum_rows, np.mean(avg_rows)), "--", color = "k")
    plt.title(f'Estimated Network Spike Predictions')
    # plt.text(0.315, -0.115, f"{prediction_deltaF_file[len(main_folder)+1:-38]}", horizontalalignment='center', verticalalignment = "center", transform=plt.gca().transAxes)
    plt.ylim(0,max_spikes_all_samples+10) ## make dynamic
    plt.ylabel("Number of Predicted Spikes")
    plt.xlabel(f'Frame Number (10 frame = 1s)')
    save_path = os.path.join(output_directory, 'total_spikes_per_frame.png')
    plt.savefig(save_path)
    print(f'Total Spikes per frame saved under {save_path}')
    #plt.show()

def plot_average_spike_probability_per_frame(predictions_deltaF_file, output_directory):
    ''' plots average spike probability across all cells divided by total number of cells in dataset (regardless of active or not), standardizes output of plot_total_spikes_per_frame()'''
    prediction_array = np.load(rf"{predictions_deltaF_file}", allow_pickle=True)
    sum_rows = np.nansum(prediction_array, axis=0)
    average = sum_rows/(len(prediction_array))
    plt.figure(figsize=(10,5))
    plt.plot(average, color = "green", label="average spike probability")
    ## actief_aandeel = (get_active_proportion_list(file)) ##used to also plot "proportion" line, not used anymore cause interpretation difficult
    #plt.plot(actief_aandeel, color = "magenta", label = "proportion of active cells")
    #plt.legend()
    plt.title(f'Average spike probability across cells per frame')
    plt.text(0.315, -0.115, f"{predictions_deltaF_file[len(main_folder)+1:-38]}", horizontalalignment='center', verticalalignment = "center", transform=plt.gca().transAxes)
    plt.ylim(0,1)
    save_path = os.path.join(output_directory, 'avg_spike_probability_per_frame.png')
    plt.savefig(save_path)
    print(f'Average spike probability per frame saved under {save_path}')
    #plt.show()

## ROI image
def getImg(ops):
    """Accesses suite2p ops file (itemized) and pulls out a composite image to map ROIs onto"""
    Img = ops["meanImg"] # Also "max_proj", "meanImg", "meanImgE"
    mimg = Img # Use suite-2p source-code naming
    mimg1 = np.percentile(mimg,1)
    mimg99 = np.percentile(mimg,99)
    mimg = (mimg - mimg1) / (mimg99 - mimg1)
    mimg = np.maximum(0,np.minimum(1,mimg))
    mimg *= 255
    mimg = mimg.astype(np.uint8)
    return mimg

    #redefine locally suite2p.gui.utils import boundary
def boundary(ypix,xpix):
    """ returns pixels of mask that are on the exterior of the mask """
    ypix = np.expand_dims(ypix.flatten(),axis=1)
    xpix = np.expand_dims(xpix.flatten(),axis=1)
    npix = ypix.shape[0]
    if npix>0:
        msk = np.zeros((np.ptp(ypix)+6, np.ptp(xpix)+6), bool) 
        msk[ypix-ypix.min()+3, xpix-xpix.min()+3] = True
        msk = binary_dilation(msk)
        msk = binary_fill_holes(msk)
        k = np.ones((3,3),dtype=int) # for 4-connected
        k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1 # for 8-connected
        out = binary_dilation(msk==0, k) & msk

        yext, xext = np.nonzero(out)
        yext, xext = yext+ypix.min()-3, xext+xpix.min()-3
    else:
        yext = np.zeros((0,))
        xext = np.zeros((0,))
    return yext, xext

#gets neuronal indices
def getStats(stat, frame_shape, output_df):
    """Accesses suite2p stats on ROIs and filters ROIs based on cascade spike probability being >= 1 into nid2idx and nid2idx_rejected (respectively)"""
    MIN_PROB = 0 
    pixel2neuron = np.full(frame_shape, fill_value=np.nan, dtype=float)
    scatters = dict(x=[], y=[], color=[], text=[])
    nid2idx = {}
    nid2idx_rejected = {}
    print(f"Number of detected ROIs: {stat.shape[0]}")
    for n in range(stat.shape[0]):
        estimated_spikes = output_df.iloc[n]["EstimatedSpikes"]

        if estimated_spikes > MIN_PROB:
            nid2idx[n] = len(scatters["x"]) # Assign new idx
        else:
            nid2idx_rejected[n] = len(scatters["x"])

        ypix = stat.iloc[n]['ypix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1
        xpix = stat.iloc[n]['xpix'].flatten() - 1 #[~stat.iloc[n]['overlap']] - 1

        valid_idx = (xpix>=0) & (xpix < frame_shape[1]) & (ypix >=0) & (ypix < frame_shape[0])
        ypix = ypix[valid_idx]
        xpix = xpix[valid_idx]
        yext, xext = boundary(ypix, xpix)
        scatters['x'] += [xext]
        scatters['y'] += [yext]
        pixel2neuron[ypix, xpix] = n

    return scatters, nid2idx, nid2idx_rejected, pixel2neuron


def dispPlot(MaxImg, scatters, nid2idx, nid2idx_rejected,
             pixel2neuron, F, Fneu, save_path, axs=None):
             if axs is None:
                fig = plt.figure(constrained_layout=True)
                NUM_GRIDS=12
                gs = fig.add_gridspec(NUM_GRIDS, 1)
                ax1 = fig.add_subplot(gs[:NUM_GRIDS-2])
                fig.set_size_inches(12,14)
             else:
                 ax1 = axs
                 ax1.set_xlim(0, MaxImg.shape[0])
                 ax1.set_ylim(MaxImg.shape[1], 0)
             ax1.imshow(MaxImg, cmap='gist_gray')
             ax1.tick_params(axis='both', which='both', bottom=False, top=False, 
                             labelbottom=False, left=False, right=False, labelleft=False)
             print("Neurons count:", len(nid2idx))
            #  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True) 
            #  mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_rainbow) 

             def plotDict(n2d2idx_dict, override_color = None):
                 for neuron_id, idx in n2d2idx_dict.items():
                     color = override_color if override_color else mapper.to_rgba(scatters['color'][idx])
                            # print(f"{idx}: {scatters['x']} - {scatters['y'][idx]}")
                            
                     sc = ax1.scatter(scatters["x"][idx], scatters['y'][idx], color = color, 
                                      marker='.', s=1)
             plotDict(nid2idx, 'g')
             plotDict(nid2idx_rejected, 'm')
             ax1.set_title(f"{len(nid2idx)} neurons used (green) out of {len(nid2idx)+len(nid2idx_rejected)} neurons detected (magenta - rejected)") 

             plt.savefig(save_path)
             plt.close(fig)

def create_suite2p_ROI_masks(stat, frame_shape, nid2idx):
    """Function designed to do what was done above, except mask the ROIs for detection in other programs (e.g. FlouroSNNAP)"""
    #Make an empty array to contain the nid2idx masks
    roi_masks = np.zeros(frame_shape, dtype=int)

    #Iterate through the ROIs in nid2idx and fill in the masks
    for n in nid2idx.keys():
        ypix = stat.iloc[n]['ypix'].flatten() - 1
        xpix = stat.iloc[n]['xpix'].flatten() - 1

        #Ensure the indices are within the bounds of the frame_shape

        valid_idx = (xpix >= 0) & (xpix<frame_shape[1]) & (ypix >=0) & (ypix < frame_shape[0])
        ypix = ypix[valid_idx]
        xpix = xpix[valid_idx]

        #Set ROI pixels to mask

        roi_masks[ypix, xpix] = 255 # n + 1 helps to differentiate masks from background
    # plt.figure(figsize=(10, 10))
    # plt.imshow(roi_masks, cmap='gray', interpolation='none')
    # # plt.colorbar(label='ROI ID')
    # plt.title('ROI Mask')
    # plt.tight_layout()
    # plt.show()
    im = Image.fromarray(roi_masks)
    im.save(output_path)
    return im, roi_masks
    
# example call roi_masks = create_suite2p_ROI_masks(stat, getImg(ops).shape, nid2idx)

# def pynapple_plots(file_path, output_directory):#, video_label):
#     import warnings
#     warnings.filterwarnings('ignore')
    
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     df_cell_stats = data['cell_stats']
    
    
#     my_tsd = {}
#     for idx in df_cell_stats['SynapseID'][0:]:
#         my_tsd[idx] = nap.Tsd(t=df_cell_stats[df_cell_stats['SynapseID']==idx]['PeakTimes'][idx],
#                             d=df_cell_stats[df_cell_stats['SynapseID']==idx]['Amplitudes'][idx],time_units='s')
        
#     Interval_1 = nap.IntervalSet(0,180)
#     # Interval_2 = nap.IntervalSet(250,290)
#     # Interval_3 = nap.IntervalSet(290,450)
    
#     interval_set = [Interval_1]#,
#                 #Interval_2]
    
#     #Make the figure
#     plt.figure(figsize=(6,6))
#     plt.subplot(2,1,1)
#     for i, idx in enumerate(df_cell_stats['SynapseID']):
# #     
#         plt.eventplot(df_cell_stats[df_cell_stats['SynapseID']==idx]['PeakTimes'],lineoffsets=i,linelength=0.8)
# #     
#         plt.ylabel('SynapseID')
#         plt.xlabel('Time (s)')
#         plt.ylim(0,1500)
#         plt.tight_layout()
#     plt.subplot(2,1,2)
#     for i in range(1): #change range for multiple intervals
#         # plt.title(file_path)
#         # plt.title(f'interval {i+1}')
#         for idx in my_tsd.keys():
#             plt.plot(my_tsd[idx].restrict(interval_set[i]).index,my_tsd[idx].restrict(interval_set[i]).values,color=f'C{idx}',marker='o',ls='',alpha=0.5)
#         plt.ylabel('Amplitude')
#         plt.ylim(0,1000)
#         plt.xlabel('Spike time (s)')
#         plt.tight_layout()

#     base_file_name = os.path.splitext(os.path.basename(file_path))[0]
    
#     #Check if output 
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
    
#     figure_output_path = os.path.join(output_directory, f'{base_file_name}_figure.png')

#     plt.savefig(figure_output_path)
#     plt.show()

#             ## You can then just group the amplitude as you want for later analysis

#     transient_count = []
#     for idx in my_tsd.keys():
#         transient_count.append(my_tsd[idx].restrict(interval_set[0]).shape[0])




_available_tests = {
    "mann-whitney-u": stats.mannwhitneyu,
    "wilcoxon": stats.wilcoxon,
    "paired_t": stats.ttest_rel,
}
def get_significance_text(series1, series2, test="mann-whitney-u", bonferroni_correction=1, show_ns=False, 
                          cutoff_dict={"*":0.05, "**":0.01, "***":0.001, "****":0.00099}, return_string="{text}\n{pvalue:.4f}"):
    statistic, pvalue = _available_tests[test](series1, series2)
    levels, cutoffs = np.vstack(list(cutoff_dict.items())).T
    levels = np.insert(levels, 0, "n.s." if show_ns else "")
    text = levels[(pvalue < cutoffs.astype(float)).sum()]
    return return_string.format(pvalue=pvalue, text=text) #, text=text

def add_significance_bar_to_axis(ax, series1, series2, center_x, line_width):
    significance_text = get_significance_text(series1, series2, show_ns=True)
    
    original_limits = ax.get_ylim()
    
    ax.errorbar(center_x, original_limits[1], xerr=line_width/2, color="k", capsize=12)
    ax.text(center_x, original_limits[1], significance_text, ha="center", va="bottom", fontsize = 32)
    
    extended_limits = (original_limits[0], (original_limits[1] - original_limits[0]) * 1.2 + original_limits[0])
    ax.set_ylim(extended_limits)
    
    return ax

def aggregated_feature_plot(summary_stats, df, feature="SpikesFreq", agg_function="median", comparison_function="mean",
                            palette="Set3", significance_check=False, group_order=None, control_group=None, ylim=0, y_label="", x_label=""):
    """
    Add a 'group_order' parameter that takes a list of groups in the desired order.
    """
    # Flatten the multi-level columns
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    
    # Filter the required feature and reset index
    feature_col = f"{feature}_{agg_function}"
    grouped_df = summary_stats[[feature_col]].reset_index()
    grouped_df.columns = ["Group", "Time_Point", feature]
    
    if control_group is not None:
        control_avg = grouped_df[grouped_df['Group'] == control_group][feature].agg(comparison_function)
        grouped_df[feature] = grouped_df[feature].apply(lambda x: (x / control_avg) * 100)
    
    fig = plt.figure(figsize=(48, 16))
    ax = fig.add_subplot()
    color_palette = sns.color_palette(palette)
    
    sns.violinplot(x="Time_Point", y=feature, data=grouped_df, ax=ax, palette=palette, order=group_order, inner="quartile", width=0.5)

    if group_order:
        tick_positions = {group: pos for pos, group in enumerate(group_order)}
    else:
        tick_positions = {ax.get_xticklabels()[index].get_text(): ax.get_xticks()[index] for index in range(len(ax.get_xticklabels()))}

    if significance_check:
        sub_checks = [significance_check] if not any(isinstance(element, list) for element in significance_check) else significance_check
        for sub_check in sub_checks:
            add_significance_bar_to_axis(ax, 
                                         grouped_df[grouped_df["Group"] == sub_check[0]][feature], 
                                         grouped_df[grouped_df["Group"] == sub_check[1]][feature],
                                         (tick_positions[sub_check[0]] + tick_positions[sub_check[1]]) / 2,
                                         abs(tick_positions[sub_check[0]] - tick_positions[sub_check[1]]))

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
    ax.set_ylim([ylim, ax.get_ylim()[1]])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(y_label, fontsize=64)
    ax.set_xlabel(x_label, fontsize=64)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=44)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=44)

    return fig