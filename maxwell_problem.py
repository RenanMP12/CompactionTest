# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:47:49 2024

@author: rmportel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pymoo.core.problem import ElementwiseProblem

plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = [12.0, 6.00]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["lines.linewidth"] = 2

prev_sum = 0.05

class Maxwell_Model(ElementwiseProblem):
    
    
    def __init__(self, compaction_data):
        xl = [0, 0, 0, 0, 0, 0, 0]
        xu = [1e+6, 1e+6, 1e+2, 1e+6, 1e+2, 1e+6, 1e+2]

        super().__init__(n_var=7, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)
        self.compaction_data = pd.read_csv(compaction_data) 
        
    def plot_compaction(self, soma, x, sigma_exp, sigma_num):
        global prev_sum
        
        if np.sqrt(soma) < prev_sum:
            prev_sum = soma

            fig, ax = plt.subplots()
            t = self.compaction_data['Time'].to_numpy(dtype = float)
            
            ax.plot(t, sigma_num/1e6, label = 'Maxwell', color = 'blue')
            ax.plot(t, sigma_exp.to_numpy(dtype = float)/1e6, 
                    label = 'Experimental', marker = 'x', color = 'red')
            ax.set_ylabel(u'\u03C3 [MPa]')
            ax.set_xlabel('Time [s]')
            
            ax.set_xlim([200, 320])
            ax.set_xticks([220, 240, 260, 280, 300, 320])
            
            ax.set_ylim([0.11, 0.16])
            ax.set_yticks([0.110, 0.120, 0.130, 0.140, 0.150])
            
            ax.grid()
            ax.legend()
            
            plt.savefig('outputs/compaction_maxwell_stage_2.jpg')
    
    def _evaluate(self, x, out, *args, **kwargs):
        # x = [E_0, E_1, tau_1, E_2, tau_2, E_3, tau_3]
        sigma_exp = self.compaction_data['Force-mean']/(np.pi*28**2)
        
        sigma_exp *= 10**6
        number_of_points = len(sigma_exp)
        sigma_num = np.zeros(number_of_points)
        Delta_t = self.compaction_data['Time'][1] - self.compaction_data['Time'][0]
        t = 0
        Step = 0
        while Step < number_of_points:
            strainlevel = self.compaction_data['Strain'][Step]
            sigma_num[Step] = (x[0] + x[1]*np.exp(-t/x[2]) + x[3]*np.exp(-t/x[4]) + x[5]*np.exp(-t/x[6]))*strainlevel
            
            t += Delta_t
            Step += 1

        # erro
        soma = 0.
        for j in range(number_of_points):
            soma += ((sigma_exp[j]-sigma_num[j])/1e+5)**2
        
        self.plot_compaction(soma, x, sigma_exp, sigma_num)
          
        out["F"] = np.sqrt(soma)
        
        return out