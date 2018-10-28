import matplotlib as mpl
import itertools
import PyVal
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ipyparallel as ipp
import seaborn as sns
import gzip
from tqdm import tqdm_notebook
from functools import reduce
import numpy as np

N = 4
row_val = 0.4
col_val = 0.4
p = 0.1
T = 1.0 
r = 0.0
S0 = 1.0
sigma = 0.4
default_scale = 1.0
netType = 2
    
N_MC = 10000
N_nets = 1
nw = PyVal.BS_Network()
print("Runing N=" +str(N)+", col sum="+str(col_val)+", p="+str(p))
nw.run(N, p, row_val, col_val, 2, T, r, S0, sigma, N_MC,  N_nets, default_scale, netType)
k_list = nw.k_vals()[0]
res = []
print("klist")
for k in k_list:
	res.append({'N': N, 'Number Of Samples': nw.get_N_samples(k)[0], 'default scale': default_scale,\
	   'conn': k, 'col sum': col_val, 'T':T, 'r': r , 'sigma': sigma, 'p': p, 'S0': S0, \
	   'M': nw.get_M(k), 'M var': nw.get_M_var(k),\
	   'Assets': np.array(nw.get_assets(k))[0], 'Assets var': np.array(nw.get_assets_var(k))[0],\
	   'RS': np.array(nw.get_rs(k))[0],  'RS var': np.array(nw.get_rs_var(k))[0],\
	   'Delta': nw.get_delta_jacobians(k),  'Delta var': nw.get_delta_jacobians_var(k),\
	   'Vega': np.array(nw.get_vega(k)),  'Vega var': np.array(nw.get_vega_var(k)),\
	   'Theta': np.array(nw.get_theta(k)),  'Theta var': np.array(nw.get_theta_var(k)),\
	   'Rho': np.array(nw.get_rho(k)),  'Rho var': np.array(nw.get_rho_var(k)),\
	   'Solvent': np.array(nw.get_solvent(k)), 'Solvent var': np.array(nw.get_solvent_var(k)),\
	   'Pi': np.array(nw.get_pi(k)),  'Pi var': np.array(nw.get_pi_var(k)),\
	    'IO Degree Distribution': np.array(nw.get_io_deg_dist())\
	})
