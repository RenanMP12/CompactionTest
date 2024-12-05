# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:01:45 2024

@author: rmportel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pymoo.core.problem import ElementwiseProblem

plt.rcParams["axes.labelsize"] = 30
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = [12.0, 6.00]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 30
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["lines.linewidth"] = 2.5
plt.rcParams["lines.markersize"] = 10
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30

prev_sum = 0.03

class Compaction_Problem(ElementwiseProblem):
    
    def __init__(self, compaction_data):
        xl = [0.25, 0, 0,0]
        xu = [1.00, 1e+6, 1e+6, 1e+12]

        super().__init__(n_var=4, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)
        
        self.compaction_data = pd.read_csv(compaction_data) 
        
    ## Gruenwald-Letnikov coefficients
    def gruenwald(self, alpha: float, number_of_points: int) -> list[float]:
        """
        Function to compute the Gruenwald coeficients 

        Parameters
        ----------
        alpha : float
            DESCRIPTION. Fractional-Zener parameter
            
        number_of_points : int
            DESCRIPTION. Number of points extracted from the curve to fit

        Returns
        -------
        Grunwald: list[float]
            DESCRIPTION. List of Gruenwald-Letnikov coefficients to compute the
            i-th stress and strain values 

        """
        Grunwald = np.zeros(number_of_points+1)
        Grunwald[0] = 1.
        for i in range (number_of_points):
            Grunwald[i+1] = (i-alpha)/(i+1)*Grunwald[i]
        return Grunwald       

    ## A & B parameters
    def ab_parameters(self, x: list[float]) -> [float, float]:
        """
        Function to compute the parameters "a" and "b" for the Fractional-Zener
        rheological model

        Parameters
        ----------
        x : list[float]
            DESCRIPTION. List of rheological coeficients

        Returns
        -------
        a: float
            DESCRIPTION.
            
        b: float
            DESCRIPTION.

        """
        alpha = x[0]
        p = x[1]
        E0 = x[2]
        E1 = x[3]
        Delta_t = self.compaction_data['Time'][1] - self.compaction_data['Time'][0]
        a = (p/E1)*Delta_t**(-alpha)
        b = a*(E0 + E1)
        return a, b
    
    def plot_compaction(self, soma, x, sigma_exp, sigma_num):
        global prev_sum
        
        if np.sqrt(soma) < prev_sum:
            prev_sum = soma

            fig, ax = plt.subplots()
            
            t = self.compaction_data['Time'].to_numpy(dtype = float)
            ax.plot(t, sigma_num/1e6, label = 'Fractional Zener', color = 'blue')
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
            
            plt.savefig('outputs/compaction_fractional_zener_stage_2.jpg')

    def _evaluate(self, x, out, *args, **kwargs):
        sigma_exp = self.compaction_data['Force-mean']/(np.pi*28**2)
        sigma_exp *= 10**6
        
        number_of_points = len(sigma_exp)
        
        sigma_num = np.zeros(number_of_points)
        
        a, b = self.ab_parameters(x)
        c2 = 1+a; c1 = (x[2]+b)/c2
        gr = self.gruenwald(x[0], number_of_points)
    
        Step = 0
        while Step < number_of_points:
            strainlevel = self.compaction_data['Strain'][Step]
            StrainFrac = 0.0
            StressFrac = 0.0
            for j in range(1, Step+1):
                StrainFrac = StrainFrac + gr[j]*strainlevel
                StressFrac = StressFrac + gr[j]*sigma_num[Step-j]
            sigma_num[Step] = c1*strainlevel+b*StrainFrac/c2-a*StressFrac/c2
            Step = Step + 1

        # erro
        soma = 0.
        for j in range(number_of_points):
            soma += ((sigma_exp[j]-sigma_num[j])/1e+5)**2
        
        self.plot_compaction(soma, x, sigma_exp, sigma_num)
          
        out["F"] = np.sqrt(soma)
        
        return out
    