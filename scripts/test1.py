import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

float_max = 999

#S_t   # equity of N firms at time t
#F_t   # debt value of N firms at time t
#V_t   # business asset values at time t
#Z_t   # (S_t, F_t)
#v_ij  # ratio of i's investment value in business asset V_j
#f_ij  # ratio of debt valueheld by i to the total debt value of j
#s_ij  # ratio of equity
#z   # z = (s_ij, f_ij)
#B     #face value
# g(Z) = [np.max(v*V + z*Z - B, 0.), np.min(v*V + z*Z, B)]

V0 = np.array([])               # initial business assets
#A = Vik Vk + Sij Sj _ fij F    # asset value

#A_t = v * V_t + z * Z

class Test1:
    def __init__(self, N_firms, max_it, L):
        self.N   = N_firms
        #example p 139
        self.vij = np.eye(4)#np.random.rand(N_firms, N_firms)
        self.zij = np.array([[.0,.2,.3,.1,.0,.0,.1,.0],[.2,.0,.2,.1,.0,.0,.0,.1],\
                [.1,.1,.0,.3,.1,.0,.0,.1],[.1,.1,.1,.0,.0,.1,.0,.0]])
        self.B   = np.array([.8,.8,.8,.8])#np.random.uniform(0,100,N_firms)
        self.L   = L
        self.max_it  = max_it
        self.Z   = np.zeros((2*N_firms,))


    def run(self):
        Zl = np.zeros(2*self.N)
        zero  = np.zeros(self.N)
        for r in range(self.max_it):
            V = np.array([2.0,0.5,0.6,0.6])#float_max*np.random.uniform(0., 1., size=(self.N,)) #
            tmp = self.vij.dot(V) + self.zij.dot(Zl)
            t1 = np.maximum(tmp - self.B, zero)
            t2 = np.minimum(tmp, self.B)
            Zl = np.hstack((t1, t2))
            print(Zl)
        self.Z = self.Z + Zl
