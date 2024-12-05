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
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultSingleObjectiveTermination
import time
from pathlib import Path
import os 
from tqdm import tqdm


rheological_model = 'Maxwell' # 'Fractional Zener'
stage = 'Second'

max_iter = 60000
my_pop_size = 7000

folder = 'Spreadsheet'
folder_name_2 = 'outputs'
info = f'{rheological_model} - {stage} Stage'

itens = os.listdir(folder)
itens.pop(0)

def create_folder(folder_name = 'Figures'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


for i in tqdm(itens, desc = 'Overall Optimization'):
    
    file = folder + '/' + i
    
    name = Path(file).stem[7:].replace("_", " ")
    
    if rheological_model == 'Maxwell':
        problem = Maxwell_Model(file)
        
    elif rheological_model == 'Fractional Zener':
        problem = Compaction_Problem(file)
 

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
        ftol = 1e-6,            # tolerance of the objective space
        period = 100,           # number of generations to compare tolerances        
        n_max_gen = 100,        # maximum number of generations
        n_max_evals = max_iter) # maximum number of function evaluations

    start_wallclock_time = time.time()
    start_cpu_time = time.process_time()
    resmin = minimize(problem, algorithm, termination, seed=1, verbose=False)
    
    create_folder(folder_name_2)
    with open(f'{folder_name_2}/{info}.txt', 'a') as f:
        print(f'{name:-^60}', file = f)
        print("Wall clock Time: {0:.6e}".format(time.time() - start_wallclock_time), file = f)
        print("CPU Time: {0:.6e}".format(time.process_time() - start_cpu_time), file = f)
        print("Function evaluations: ", resmin.algorithm.evaluator.n_eval, file = f)
        print("Function value: ", resmin.F[0], file = f)
        print("Best X: ", resmin.X, file = f)
        print("\n", file = f)



    n_evals = np.array([e.evaluator.n_eval for e in resmin.history])
    opt = np.array([e.opt[0].F for e in resmin.history])
