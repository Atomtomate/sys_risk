
def reduce_to_scalars_old(df, conn_scale):
    for ix in df.index:
        for ind in df.loc[ix].index:
            N = int(df.loc[ix,'N'])
            if ind == 'conn':
                df.loc[ix, ind] = df.loc[ix, ind]/conn_scale
            elif ind == 'Number Of Samples':
                df.loc[ix, ind] = df.loc[ix, ind][1]
            elif ind == 'RS':
                df.loc[ix, 'S'] = np.average(df.loc[ix, ind][:N])
                df.loc[ix, 'R'] = np.average(df.loc[ix, ind][N:])
                #df.loc[ix, 'R'] = df.loc[ix, ind]
            elif ind == 'Delta' or ind == 'Delta var' or ind == 'Vega' or ind == 'Vega var':
                df.loc[ix, ind] = np.sum(df.loc[ix, ind])/N
            elif ind == 'Rho'or ind == 'Theta' or ind == 'Pi' or ind == 'Rho var' or ind == 'Theta var' or ind == 'Pi var':
                df.loc[ix,"equity " + ind] = np.average(df.loc[ix, ind][:N])
                df.loc[ix,"debt " + ind] = np.average(df.loc[ix, ind][N:])
            elif ind == 'M' or ind == 'M var':
                df.loc[ix, 'avg col sum'] = np.average(np.sum(df.loc[ix, ind][:,N:], axis=0))
                df.loc[ix, 'avg row sum'] = np.average(np.sum(df.loc[ix, ind], axis=1))
                df.loc[ix, ind] = np.sum(df.loc[ix, ind])/N
            else:
                df.loc[ix, ind] = np.average(df.loc[ix, ind])
    df.drop(columns='RS', inplace=True)
    df.drop(columns='Delta', inplace=True)
    df.drop(columns='Delta var', inplace=True)
    df.drop(columns='Vega', inplace=True)
    df.drop(columns='Vega var', inplace=True)
    df.drop(columns='Rho', inplace=True)
    df.drop(columns='Rho var', inplace=True)
    df.drop(columns='Theta', inplace=True)
    df.drop(columns='Theta var', inplace=True)
    df.drop(columns='Pi', inplace=True)
    df.drop(columns='Pi var', inplace=True)
    df.drop(columns='M', inplace=True)
    df.drop(columns='M var', inplace=True)



def reduce_star_to_scalars(df, N, conn_scale):
    #cols = df.columns.values
    #cols = np.delete(cols, np.where((cols == 'RS') | (cols == 'Delta') | (cols == 'Delta var') | (cols == 'Vega') |\
    #                            (cols == 'Vega') | (cols == 'Vega var') | (cols == 'Rho') | (cols == 'Rho var') |\
    #                            (cols == 'Theta') | (cols == 'Theta var') | (cols == 'Pi') | (cols == 'Pi var') |\
    #                            (cols == 'M') | (cols == 'M var') | (cols == 'N_MC') | (cols == 'N_nets') |\
    #                            (cols == 'col sum') ))
    cols =  np.array(['N', 'r', 'T', 'conn', 'default scale', 'col sum', 'S0', 'sigma',\
                      'Solvent', 'Solvent var', 'Assets', 'Assets var', 'R', 'S', \
                      'equity Delta', 'equity Delta var', 'debt Delta', 'debt Delta var',\
                      'equity Vega', 'equity Vega var', 'debt Vega', 'debt Vega var',\
                      'equity Rho', 'equity Rho var', 'debt Rho', 'debt Rho var',\
                      'equity Theta', 'equity Theta var', 'debt Theta', 'debt Theta var',\
                      'Pi', 'Pi var'])
    df2 = pd.DataFrame(columns = cols, dtype=np.float64)
    df2_debug = pd.DataFrame(columns = np.array(['M'])) #, 'io degree'
    df2['Number Of Samples'] = df['Number Of Samples'].transform(lambda x: x[1])
    df2['S'] = df['RS'].transform(lambda x: np.average(x[1:N]))
    df2['R'] = df['RS'].transform(lambda x: np.average(x[N+1:]))
    df2['S 0'] = df['RS'].transform(lambda x: np.average(x[0]))
    df2['R 0'] = df['RS'].transform(lambda x: np.average(x[N]))
    #df2['equity Delta'] = df['Delta'].transform(lambda x: np.sum(x[:N,1:])/(N-1))
    #df2['equity Delta 0'] = df['Delta'].transform(lambda x: np.sum(x[:N,0]))
    #df2['equity Delta var'] = df['Delta var'].transform(lambda x: np.sum(x[:N,:])/(N-1))
    #df2['equity Delta 0 var'] = df['Delta var'].transform(lambda x: np.sum(x[:N,0]))
    #df2['debt Delta'] = df['Delta'].transform(lambda x: np.sum(x[N:,1:])/(N-1))
    #df2['debt Delta 0'] = df['Delta'].transform(lambda x: np.sum(x[N:,0]))
    #df2['debt Delta var'] = df['Delta var'].transform(lambda x: np.sum(x[N:,1:])/(N-1))
    #df2['debt Delta 0 var'] = df['Delta var'].transform(lambda x: np.sum(x[N:,0]))
    for el in ['Delta', 'Vega', 'Rho', 'Theta']:
        df2['equity ' + el] = df[el].transform(lambda x: np.sum(x[1:,1:N])/(N-1))
        df2['equity ' + el + ' 0'] = df[el].transform(lambda x: np.sum(x[:,:1]))
        df2['equity ' +el+ ' var'] = df[el+' var'].transform(lambda x: np.sum(x[:,1:N])/(N-1))
        df2['equity ' +el+ ' var 0'] = df[el+' var'].transform(lambda x: np.sum(x[:,:1]))
        df2['debt ' + el] = df[el].transform(lambda x: np.sum(x[:,N+1:])/(N-1))
        df2['debt ' + el+' 0'] = df[el].transform(lambda x: np.sum(x[:,N]))
        df2['debt ' +el+' var'] = df[el+' var'].transform(lambda x: np.sum(x[:,N+1:])/(N-1))
        df2['debt ' +el+' var 0'] = df[el+' var'].transform(lambda x: np.sum(x[:,N]))
    df2_debug['M'] = df['M']
    #df2_debug['io degree'] = df['IO Degree Distribution']
    df2['Assets'] = df['Assets'].transform(lambda x: np.average(x[1:N]))
    df2['Assets 0'] = df['Assets'].transform(lambda x: x[0])
    df2['Assets var'] = df['Assets var'].transform(lambda x: np.average(x[1:N]))
    df2['Assets 0 var'] = df['Assets var'].transform(lambda x: x[0])
    df2['Solvent'] = df['Solvent'].transform(lambda x: np.average(x[1:N]))
    df2['Solvent 0'] = df['Solvent'].transform(lambda x: x[0])
    df2['Solvent var'] = df['Solvent var'].transform(lambda x: np.average(x[1:N]))
    df2['Solvent 0 var'] = df['Solvent var'].transform(lambda x: x[0])
    df2['Pi'] = df['Pi'].transform(lambda x: np.average(x[1:N]))
    df2['Pi 0'] = df['Pi'].transform(lambda x: x[0])
    df2['Pi var'] = df['Pi var'].transform(lambda x: np.average(x[1:N]))
    df2['Pi 0 var'] = df['Pi var'].transform(lambda x: x[0])
    df2['N'] = df['N']
    df2['r'] = df['r']
    df2['T'] = df['T']
    df2['conn'] = df['conn'].transform(lambda x: x/conn_scale)
    df2['default scale'] = df['default scale']
    df2['col sum'] = df['col sum']
    df2['S0'] = df['S0']
    df2['sigma'] = df['sigma']
    df2['Network Type'] = df['Network Type']
    #df2['p'] = df['p']
    
    return (df2, df2_debug)


def run_sim(N, row_val, col_val, p, T, r, S0, sigma, default_scale, netType):
    import numpy as np
    import PyVal
    mul = max(1,col_val+S0)
    N_MC = 20000#int(15000*(1+sigma))#int(400*mul)    # 600
    N_nets = 1#000#int(800*mul)  # 1100
    nw = PyVal.BS_Network()
    print("Runing N=" +str(N)+", col sum="+str(col_val)+", p="+str(p)+", T="+str(T)+", r="+str(r)+\
          ", S0="+str(S0)+", sigma="+str(sigma)+", default_scale="+str(default_scale)+", netType="+str(netType))
    nw.run(N, p, row_val, col_val, 2, T, r, S0, sigma, N_MC,  N_nets, default_scale, netType)
    k_list = nw.k_vals()[0]
    res = []
    print(k_list)
    for k in k_list:
        res.append({'Network Type': netType, 'N': N, 'Number Of Samples': nw.get_N_samples(k)[0], 'default scale': default_scale,\
               'conn': k, 'col sum': col_val, 'T':T, 'r': r , 'sigma': sigma, 'p': p, 'S0': S0, \
               'M': nw.get_M(k), 'M var': nw.get_M_var(k),\
               'Assets': np.array(nw.get_assets(k))[0], 'Assets var': np.array(nw.get_assets_var(k))[0],\
               'RS': np.array(nw.get_rs(k))[0],  'RS var': np.array(nw.get_rs_var(k))[0],\
               'Delta': nw.get_delta_jacobians(k),  'Delta var': nw.get_delta_jacobians_var(k),\
               'Vega': np.array(nw.get_vega(k)),  'Vega var': np.array(nw.get_vega_var(k)),\
               'Theta': np.array(nw.get_theta(k)),  'Theta var': np.array(nw.get_theta_var(k)),\
               'Rho': np.array(nw.get_rho(k)),  'Rho var': np.array(nw.get_rho_var(k)),\
               'Solvent': np.array(nw.get_solvent(k))[0], 'Solvent var': np.array(nw.get_solvent_var(k))[0],\
               'Pi': np.array(nw.get_pi(k))[0],  'Pi var': np.array(nw.get_pi_var(k))[0]
            #'IO Degree Distribution': np.array(nw.get_io_deg_dist())\
                   })
    return res

        
            
    
def flatten_input(cell):
    if type(cell) is list:
        return cell[0]
    else:
        return cell

def pList(N, pts):
    import numpy as np
    #    return np.union1d(np.union1d(np.linspace(0.0, 10.0/N, pts), np.linspace(0.6/N,1.4/N,pts)), np.linspace((np.log(N)-np.log(N)/7.)/N,(np.log(N)+np.log(N)/7.)/N,pts))
    res = np.union1d(np.linspace(0.0, 2.0/N, 2*pts), np.union1d(np.linspace(0.0, 8.0/N, pts), np.linspace((np.log(N)-np.log(N)/7.)/N, (np.log(N)+np.log(N)/7.)/N, pts)))
    res = res[res <= 5.0/N]
    return res

def pFixedList(N, pts):
    import numpy as np
    return np.linspace(0.0, (pts-1)/N, pts)


