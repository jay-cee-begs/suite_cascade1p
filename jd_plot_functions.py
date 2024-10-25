import numpy as np
import scipy.io as sio


#%matplotlib widget # can be commented back in to make plots interactive
import pandas as pd
#%pip install --upgrade matplotlib
import seaborn as sns
from statannotations.Annotator import Annotator
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
from configurations import main_folder

def load_and_adjust(TimePoints, Groups):
    df = pd.read_csv(main_folder + r'\new_experiment_summary.csv')
    df2 = df
    df2['File'] = df.Prediction_File.apply(lambda x: x.split('\\')[1] if isinstance(x, str) else x)
    df2.drop('Prediction_File', axis=1, inplace=True)
    df2.insert(0, 'File', df2.pop('File')) 
    df2.insert(6,'Active_Percentage', (df2.Active_Neuron_F0/df.Inactive_Neuron_F0) )
    df2['Time_Point'] = df2['Time_Point'].replace(TimePoints)
    df2['Group'] = df2['Group'].replace(Groups)
    return df2



def general_plotting_function(df, x, y, type, plotby):

    if type == 'swarm':                     # only works when not trying to plot by index (eg. only df2 not df5 with multiindex) 
        df_sort = df.groupby(df[plotby]) 
        for s in df_sort.groups.keys():
            df_unique = df_sort.get_group(s)
            s1 = sns.swarmplot(x=x, y=y, data=df_unique, hue='Group')
            s1 = sns.lineplot(x=x, y=y, data=df_unique ,err_style="bars") 
            plt.show()
        return s1

    else: 
        f= sns.catplot(
            data=df, kind=type, col = plotby,
            x=x, y=y,  aspect=0.5, height=4, hue='Group')
        #modified_df = df.rename(columns=lambda name: name.replace('_', ' '))

        f.set_axis_labels(" ", "Total Estimated Spikes")
        return f
    

def ez_sign_plot(df, x, feature, type, plotby, testby,
                 stat_test=None, group_order=None, y_label="", x_label="", location='inside', legend=False, palette='Set3', aspct=0.5, hght=4): 
    # types = ['violin', 'box', 'swarm', 'bar', 'point', 'strip']
    # if type not in types:
    #     raise ValueError(f"Type must be one of {types}") 
    if type == 'swarm':                                                                                     # only works when not trying to plot by index 
        df_sort = df.groupby(df[plotby])                                                                    # sorts by whatever column you want
        for s in df_sort.groups.keys():                                                                     #iterates over the groups unique keys
            df_unique = df_sort.get_group(s)
            fig, ax = plt.subplots()                                                                        #create ax to use change it later if needed
            sns.swarmplot(x=x, y=feature, data=df_unique, hue='Group', ax=ax)
            sns.lineplot(x=x, y=feature, data=df_unique, err_style="bars", ax=ax)
            # plot specifics
            ax.set_title(df_unique.Time_Point.iloc[0])
            ax.set_xlabel(x_label, fontsize=15)
            ax.set_ylabel(y_label, fontsize=15)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

        print("Swarm plot does not support statistical tests, please use another plot type")
         

    else:
        if type == 'violin':                                                                                # violin plot extra to allow us to plot quartiles inside, otherwise not needed
            fig= sns.catplot(
                data=df, kind=type, col = plotby, inner = 'quartiles',
                x=x, y=feature,  aspect=aspct, height=hght, hue='Group', palette=palette, legend=legend)
            print('Inner lines display quartiles, change in function if needed')
                
        else:
            fig= sns.catplot(
                data=df, kind=type, col = plotby,
                x=x, y=feature,  aspect=aspct, height=hght, hue='Group', palette=palette, legend=legend)
        
        for ax in fig.axes.flat:
            ax.set_xlabel(x_label, fontsize=15)
            ax.set_ylabel(y_label, fontsize=15)
            ax.set_xticklabels(ax.get_xticklabels(),fontsize=10)
        

    if stat_test:
        for ax in fig.axes.flat:
            annotator = Annotator(ax, testby, x=x, y=feature, data=df, x_order=group_order, y_order=None)
            annotator.configure(test=stat_test, text_format='star', loc=location, hide_non_significant=True)
            annotator.apply_and_annotate()   
        print("Statistical test applied, n.s. bars hidden, change in function if needed")

    plt.show()

    return fig 



# import configurations

# if __name__ == "__main__":
#     TimePoints = configurations.TimePoints
#     Groups = configurations.Groups
#     path = configurations.main_folder + '\\extension'
#     load_and_adjust(TimePoints, Groups)

#     ez_sign_plot(df, x, feature, type, plotby, testby,
#                  stat_test=None, group_order=None, y_label="", x_label="", location='inside', legend=False, palette='Set3', aspct=0.5, hght=4): 