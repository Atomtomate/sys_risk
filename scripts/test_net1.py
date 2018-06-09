import PyVal
import numpy as np

N = 80
nw = PyVal.BS_Network()

pList = np.linspace(0.1,1.0,10)
MList = []
rsList= []
deltaList = []
solventList = []

for p in pList:
    nw.run(60, p, 0.9, 0, 2.0, 0.0, 10000)
    MList.append(nw.get_M().copy())
    rsList.append(nw.get_rs().copy())
    solventList.append(nw.get_solvent().copy())
    deltaList.append(nw.get_delta_jacobians().copy())
