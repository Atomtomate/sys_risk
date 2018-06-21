import PyVal
import numpy as np
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()-2

val = 0.2
k_min = 0.1
n_points = 3

def run_sim(N, val, nPoints):
    nw = PyVal.BS_Network()
    MList = []
    rsList= []
    deltaList = []
    solventList = []
    rsVarList= []
    deltaVarList = []
    solventVarList = []
    #vOutList = []
    pList = np.union1d(np.linspace(0.01/N,1.8/N,n_points), np.linspace(0.7/N,1.3/N,n_points))
    for p in pList:
        nw.run(N, p, val, 2, 1.0, 0.0, 350, 20)
        MList.append(nw.get_M().copy())
        rsList.append(nw.get_rs().copy())
        solventList.append(nw.get_solvent().copy())
        deltaList.append(nw.get_delta_jacobians().copy())
        rsVarList.append(nw.get_rs_var().copy())
        solventVarList.append(nw.get_solvent_var().copy())
        deltaVarList.append(nw.get_delta_jacobians_var().copy())
    res = {'N': N, 'conn': N*np.array(pList), 'val': val, 'M': MList,\
		'RS': rsList, 'Delta': deltaList, 'Solvent': solventList, 'RS var': rsVarList, 'Delta var': deltaVarList, 'Solvent var': solventVarList}
    return res

inputs = [(500, val, 3)]#, (75, val, 4), (100, val, 4), (150, val, 4), (200, val, 4), (300, val, 4)]
#results = Parallel(n_jobs=num_cores)(delayed(run_sim)(*i) for i in inputs)
results = []
for inp in inputs:
    results.append(run_sim(*inp))

#deltaSum = np.array([np.sum(el,axis=(1,2)) for el in delta])
#vOutList = [(1.0-np.array(MList[i])).dot(np.array(rsList[i])) for i in range(len(pList))]
#plt.plot(np.array(connectivity).T, np.array(deltaSum).T/nList);

#bak = {'M': M, 'rs': rs, 'delta': delta, 'solvent': solvent, 'rs var': rs_var, 'delta var': delta_var, 'solvent var': solvent_var}
np.save("test_par.npy", results)
