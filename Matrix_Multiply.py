import numpy as np
import sys
import os
import random
from joblib import Parallel, delayed
import multiprocessing

num_samples = 10
A_list = list()
B_list = list()
C_list = list()
def initialize_lists():
    A = np.random.normal(0,1,(32,512))
    B = np.random.normal(0,1,(512,32))
    C = np.dot(A,B).flatten()
    return [A, B, C]

   
num_cores = multiprocessing.cpu_count()
    
results = Parallel(n_jobs=2)(delayed(initialize_lists)() for i in range(num_samples))
print(results)