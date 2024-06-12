# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:00:40 2024

@author: rmportel
"""

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from compaction_problem import Compaction_Problem
from maxwell_problem import Maxwell_Model
from pymoo.optimize import minimize
from pymoo.core.repair import NoRepair
#from pymoo.visualization.scatter import Scatter
from pymoo.operators.sampling.lhs import LHS
#from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination
import time
from pathlib import Path
#import matplotlib.pyplot as plt
import os 

max_iter = 60000
my_pop_size = 7000

folder = 'Spreadsheet'

itens = os.listdir('Spreadsheet')
itens.pop(0)

for i in itens:
    file = folder + '/' + i
    
    name = Path(file).stem[7:].replace("_", " ")

    problem = Compaction_Problem(file)

    #problem = Maxwell_Model(file)

    algorithm = PSO(pop_size = my_pop_size,
                    sampling=LHS(),
                    #sampling = FloatRandomSampling(),
                    w=0.9,
                    c1=2.0,
                    c2=2.0,
                    adaptive=True,
                    initial_velocity="random",
                    max_velocity_rate=0.20,
                    pertube_best=True,
                    repair=NoRepair(),
                    save_history=True)

    # termination criterion for the optimization algorithm
    termination = DefaultSingleObjectiveTermination(
        ftol = 1e-6, # tolerance of the objective space
        period = 100, # number of generations to compare tolerances        
        n_max_gen = 100, # maximum number of generations
        n_max_evals = max_iter) # maximum number of function evaluations

    start_wallclock_time = time.time()
    start_cpu_time = time.process_time()
    resmin = minimize(problem, algorithm, termination, seed=1, verbose=False)
    print('------------------' + name + '------------------')
    print("Wall clock Time: {0:.6e}".format(time.time() - start_wallclock_time))
    print("CPU Time: {0:.6e}".format(time.process_time() - start_cpu_time))
    print("Function evaluations: ", resmin.algorithm.evaluator.n_eval)
    print("Function value: ", resmin.F[0])
    print("Best X: ", resmin.X)
    print("\n")



    n_evals = np.array([e.evaluator.n_eval for e in resmin.history])
    opt = np.array([e.opt[0].F for e in resmin.history])
'''
fig1, ax2 = plt.subplots()
ax2.title("Convergence")
ax2.plot(n_evals, opt, "--")
ax2.yscale("log")
'''