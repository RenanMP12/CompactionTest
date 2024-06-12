import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_set_experiments(experimetal_data_dictionary):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    name = experimetal_data_dictionary['name']
    del experimetal_data_dictionary['name']
    
    marker = ['s', 'o', '^', '*']
    color = ['red', 'blue']
    
    fig, ax1 = plt.subplots()
    ax1.grid()
    ax2 = ax1.twinx()
    
    for i in range(len(experimetal_data_dictionary)):
        x = experimetal_data_dictionary[i].iloc[:,0].to_numpy()
        y1 = experimetal_data_dictionary[i].iloc[:,1].to_numpy()
        y2 = 6 - experimetal_data_dictionary[i].iloc[:,2].to_numpy()
       
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Force [N]')
        ax1.plot(x, y1, color = color[0], marker = marker[i], markevery = 1500)
        
        ax2.set_ylabel('Displacement [mm]')
        ax2.plot(x, y2, color = color[1], marker = marker[i], markevery = 1500)
        
        ax1.set_xlim(0,1000)  
        ax1.set_ylim(0,600)  
    
    sfile = 'Graphics'
    if os.path.exists(sfile) == False:
        os.makedirs(sfile)
    fig.savefig(sfile + '/' + name + '_experimental_data.png')
    experimetal_data_dictionary['name'] = name
    
def plot_statistical_analysis(experimetal_data_dictionary):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    name = experimetal_data_dictionary['name']
    del experimetal_data_dictionary['name']    

    fig, ax1 = plt.subplots()
    ax1.set_ylim(0,600)  
    ax1.set_xlim(0,900)  
    ax1.grid()
    ax2 = ax1.twinx()

    x_ = np.arange(0, 890, 0.01)

    columns1 = ['y1_' + str(i) for i in range(len(experimetal_data_dictionary))] + ['Time (s)']
    df_force = pd.DataFrame(columns = columns1)

    columns2 = ['y2_' + str(i) for i in range(len(experimetal_data_dictionary))] + ['Time (s)']
    df_displacement = pd.DataFrame(columns = columns2)
    
    marker = ['s', 'o', '^', '*']

    for i in range(len(experimetal_data_dictionary)):
        x = experimetal_data_dictionary[i].iloc[:,0].to_numpy() - experimetal_data_dictionary[i].iloc[0,0]
        y1 = experimetal_data_dictionary[i].iloc[:,1].to_numpy()
        y2 = 6 - experimetal_data_dictionary[i].iloc[:,2].to_numpy()

        interp_func_1 = interp1d(x, y1)
        interp_func_2 = interp1d(x, y2)

        df_force['y1_' + str(i)] = interp_func_1(x_)
        df_displacement['y2_' + str(i)] = interp_func_2(x_)
        
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force [N]')
    ax1.plot(x_, df_force.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1).to_numpy(), 
             color = 'red', label = 'Force [N]', marker = marker[i], markevery = 3000)
    ax1.fill_between(x_, df_force.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1).to_numpy() - df_force.iloc[:, 0:len(experimetal_data_dictionary)].std(axis = 1).to_numpy(), 
                         df_force.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1).to_numpy() + df_force.iloc[:, 0:len(experimetal_data_dictionary)].std(axis = 1).to_numpy(), 
                         color = 'red', alpha = 0.4)
    
    ax2.set_ylabel('Displacement [mm]')
    ax2.plot(x_, df_displacement.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1).to_numpy(),
             color = 'blue', label = 'Displacement [mm', marker = marker[i], markevery = 3000)
    ax2.fill_between(x_, df_displacement.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1).to_numpy() - df_displacement.iloc[:, 0:len(experimetal_data_dictionary)].std(axis = 1).to_numpy(), 
                         df_displacement.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1).to_numpy() + df_displacement.iloc[:, 0:len(experimetal_data_dictionary)].std(axis = 1).to_numpy(), 
                         color = 'blue', alpha = 0.4)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    sfile = 'Graphics'
    if os.path.exists(sfile) == False:
        os.makedirs(sfile)
    fig.savefig(sfile + '/' + name + '_statistical.png')
    experimetal_data_dictionary['name'] = name
    
    df = pd.DataFrame()
    df['Time'] = x_
    df['Force-mean'] = df_force.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1)
    df['Force-std'] = df_force.iloc[:, 0:len(experimetal_data_dictionary)].std(axis = 1)
    
    df['Displacement-mean'] = df_displacement.iloc[:, 0:len(experimetal_data_dictionary)].mean(axis = 1)
    df['Displacement-std'] = df_displacement.iloc[:, 0:len(experimetal_data_dictionary)].std(axis = 1)
    
    
    sfile = 'Spreadsheet'
    if os.path.exists(sfile) == False:
        os.makedirs(sfile)
        
    df[['Time', 'Force-mean', 'Force-std']][(df['Time'] < 320) & (df['Time'] > 210)].loc[::500,:].to_csv(sfile + '/output_' + name + '.csv', index = False)

    return df

def comparison_plot(data_dictionary):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    name = data_dictionary['name']
    del data_dictionary['name']    
    
    file_name = ''
    for i in name:
        file_name = file_name + i
        file_name = file_name + '_'
        
    
    marker = ['s', 'o', '^', '*']
    
    fig, ax1 = plt.subplots()
    ax1.set_ylim(0,600)  
    ax1.set_xlim(0,900)  
    ax1.grid()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force [N]')
    ax2.set_ylabel('Displacement [mm]')
    
    for i in range(len(data_dictionary)):
        x = data_dictionary[i]['Time'].to_numpy()
        y1 = data_dictionary[i]['Force-mean'].to_numpy(dtype = float)
        y2 = data_dictionary[i]['Displacement-mean'].to_numpy(dtype = float)
        
        y1_std = data_dictionary[i]['Force-std'].to_numpy(dtype = float)
        y2_std = data_dictionary[i]['Displacement-std'].to_numpy(dtype = float)
        
        ax1.plot(x, y1, label = name[i].replace("_"," "), marker = marker[i], markevery = 3000)
        ax2.plot(x, y2)
        ax1.legend()
        
        ax1.fill_between(x, y1 - y1_std, y1 + y1_std, alpha = 0.4)
        ax2.fill_between(x, y2 - y2_std, y2 + y2_std, alpha = 0.4)
        
    sfile = 'Graphics'
    if os.path.exists(sfile) == False:
        os.makedirs(sfile)
    
    fig.savefig(sfile + '/' + file_name + 'comparison_data.png')