# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:01:45 2024

@author: rmportel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pymoo.core.problem import ElementwiseProblem

prev_sum = 0.03

class Compaction_Problem(ElementwiseProblem):
    
    def __init__(self, compaction_data):
        xl = [0.25, 0, 0,0]
        xu = [1.00, 1e+6, 1e+6, 1e+12]

        super().__init__(n_var=4, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)
        
        self.compaction_data = pd.read_csv(compaction_data) 
        
    ## Gruenwald-Letnikov coefficients
    def gruenwald(self, alpha, number_of_points):
        Grunwald = np.zeros(number_of_points+1)
        Grunwald[0] = 1.
        for i in range (number_of_points):
            Grunwald[i+1] = (i-alpha)/(i+1)*Grunwald[i]
        return Grunwald       

    ## A & B parameters
    def ab_parameters(self, x):
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
            
            #s = '\u03B1 = ' + "{:.2f}".format(x[0]) 
            #t = '$p$ = ' + "{:.2e} Pa".format(x[1]) 
            #u = '$E_{0}$ = ' + "{:.2e} Pa".format(x[2])
            #v = '$E_{1}$ = ' + "{:.2e} Pa".format(x[3]) 
            #w = 'e = ' + "{:.2f}".format(np.sqrt(soma)) 
            #fig.text(0.95, 0.8, s, fontsize=12)
            #fig.text(0.95, 0.7, t, fontsize=12)
            #fig.text(0.95, 0.6, u, fontsize=12)
            #fig.text(0.95, 0.5, v, fontsize=12)
            #fig.text(0.95, 0.0, w, fontsize=12)
            t = self.compaction_data['Time'].to_numpy(dtype = float)
            ax.plot(t, sigma_num, label = 'Numerical', color = 'blue')
            ax.plot(t, sigma_exp.to_numpy(dtype = float), label = 'Experimental', marker = 'x', color = 'red')
            ax.set_ylabel(u'\u03C3 [Pa]')
            ax.set_xlabel('Time [s]')
            ax.grid()
            ax.legend()

    def _evaluate(self, x, out, *args, **kwargs):
        sigma_exp = self.compaction_data['Force-mean']/(np.pi*28**2)
        sigma_exp *= 10**6
        
        number_of_points = len(sigma_exp)
        
        sigma_num = np.zeros(number_of_points)
        strainlevel = 0.494949494949495
        
        a, b = self.ab_parameters(x)
        c2 = 1+a; c1 = (x[2]+b)/c2
        gr = self.gruenwald(x[0], number_of_points)

        Step = 0
        while Step < number_of_points:
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
    