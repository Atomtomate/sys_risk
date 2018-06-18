import PyVal
import numpy as np
import pickle
import matplotlib.pyplot as plt



pList = np.linspace(0.001,0.1,20)
nList = np.arange(10, 80, 5)

M = []
rs = []
delta = []
solvent = []


#for N in nList:
N = 80
eye2 = np.hstack((np.eye(N),np.eye(N)))
nw = PyVal.BS_Network()
MList = []
rsList= []
deltaList = []
solventList = []
#vOutList = []
for p in pList:
	nw.run(N, p, 0.8, 0, 2.0, 0.0, 4000)
	MList.append(nw.get_M().copy())
	rsList.append(nw.get_rs().copy())
	solventList.append(nw.get_solvent().copy())
	deltaList.append(nw.get_delta_jacobians().copy())
M.append(np.array(MList))
rs.append(np.array(rsList))
delta.append(np.array(deltaList))
solvent.append(np.array(solventList))

vOut = np.array([(1.0 - np.sum(MList[i], axis=0))*rsList[i].T for i in range(len(pList))])
vOut = vOut[:,0,:N] + vOut[:,0,N:]
vOutSum = np.sum(vOut, axis=1)
vInt = np.array([el[:N] + el[N:] for el in rsList])
vIntSum = np.sum(vInt, axis=1)
#vInt = np.array([MList[i].dot(rsList[i]) for i in range(len(pList))])

bak = {'M': M, 'rs': rs, 'delta': delta, 'solvent': solvent}
np.save("bak_n80_pVar.npy", bak)
