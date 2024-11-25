# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This is a Function code for estimating of 2D ETAS parameters with uniform horizontal 
background. This code will use LLrecipes to call most of the functions. 
'''

########################## Importing Required Modules #########################
import sys
import numpy as np
import LLrecipes
from scipy.optimize import minimize

############################# Functions & Classes #############################
def fR(r, d, q):
    cnorm = (q / np.pi) * np.power(d, 2*q)
    fr = cnorm * np.power(np.square(d) + np.square(r), -(1+q))
    return fr

def calculate_ETASrate(time, t, r, mu, Na, c, p, d, q):
    R = mu + np.sum(Na * np.power(c + time - t, -p) * fR(r, d, q))
    return R

def LL1value(t, ti, Ri, mu, muA, Na, c, p, d, q, Nind, Nij, bDT):
    NN = len(ti)
    fac = np.zeros(NN)
    R0 = np.zeros(NN)
    for i in range(len(ti)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        fac[i] = LLrecipes.calculate_detectionfactor_2mainshocks(0, 0, 0, 0, ti[i], t[nij], muA, Na[nij], c, p, bDT)
        R0[i] = calculate_ETASrate(ti[i], t[nij], Ri[i, nij], mu, Na[nij], c, p, d[nij], q)
        LL1 = np.sum(np.log(fac*R0))
    return LL1

def nLLETAS(arguments, tsteps, t, m, ti, mi, Ri, Mc, T1, T2, Nind, Nij, A, TmaxTrig):
    mu = np.square(arguments[0])
    K = np.square(arguments[1])
    alpha = np.square(arguments[2])
    c = np.square(arguments[3])
    p = np.square(arguments[4])
    d0 = np.square(arguments[5])
    gamma = np.square(arguments[6])
    q = np.square(arguments[7])
    b = np.square(arguments[8])
    bDT = np.square(arguments[9])
    
    d  = d0 * np.power(10.0, gamma * m)
    # cnorm = LLrecipes.determine_normalization_factor_2D(d, q)
    # Zinteg = LLrecipes.integration_zrange(d, q, z, Z1, Z2, cnorm)
    Na = K * np.power(10.0, alpha*(m-Mc))
    muA = mu * A
    
    LLM = LLrecipes.LLGR_2mainshocks(0, 0, 0, 0, t, ti, mi, Mc, muA, Na, c, p, Nind, Nij, b, bDT)
    LL1 = LL1value(t, ti, Ri, mu, muA, Na, c, p, d, q, Nind, Nij, bDT)
    LL2 = LLrecipes.LL2valueETASI_2mainshocks(tsteps, 0, 0, 0, 0, t, muA, Na, c, p, TmaxTrig, bDT)
    LL = LLM + LL1 - LL2
    nLL = -LL
    if np.isnan(nLL):
        nLL = 1e10
    tfac = 24*60.0
    sys.stdout.write('\r'+str('search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.2f[min]  p=%.2f d0=%.4f  gamma=%.2f  q=%.2f b=%.2f bDT=%.2f[min]--> Nback=%.1f Ntot=%.1f nLL=%f\r' % (mu, K, alpha, c*tfac, p, d0, gamma, q, b, bDT*tfac, muA, len(ti), nLL)))
    sys.stdout.flush()
    return nLL

def LLoptimize(x0, tsteps, t, m, ti, mi, Ri, Mcut, T1, T2, Nind, Nij, A, TmaxTrig):
    res = minimize(nLLETAS, np.sqrt(x0), args=(tsteps, t, m, ti, mi, Ri, Mcut, T1, T2, Nind, Nij, A, TmaxTrig), method='BFGS', options={'gtol': 1e-01})
    mu = np.square(res.x[0])
    K = np.square(res.x[1])
    alpha = np.square(res.x[2])
    c = np.square(res.x[3])
    p = np.square(res.x[4])
    d0 = np.square(res.x[5])
    gamma = np.square(res.x[6])
    q = np.square(res.x[7])
    b = np.square(res.x[8])
    bDT = np.square(res.x[9])
    return mu, K, alpha, c, p, d0, gamma, q, b, bDT, -res.fun

def setstartparameter(N, A, T1, T2):
    mu = 0.5*N/(A*(T2-T1))
    K     = 0.05   
    alpha = 1.0                 
    c     = 0.01    # [days]
    p     = 1.3
    d0    = 0.013   # ... with d = d0 * 10^(gamma*M) 
    gamma = 0.5     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    q     = 0.5     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    b     = 1.0
    bDT   = 100.0 / (24 * 60 * 60.0)   #  blind time
    
    # mu = 0.104
    # K     = 0.04   
    # alpha = 0.4                 
    # c     = 0.005    # [days]
    # p     = 0.9
    # d0    = 0.015   # ... with d = d0 * 10^(gamma*M) 
    # gamma = 0.4     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    # q     = 0.45     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    # b     = 0.8
    # bDT   = 30
    return np.asarray([mu, K, alpha, c, p, d0, gamma, q, b, bDT])

def determine_ETASparameter(tall, latall, lonall, mall, Mcut, T1, T2, A, TmaxTrig):
    '''
    Search of the minimum -LL value for the ETAS-model

    Input: t, lat, lon, m     sequence of earthquake times, location and magnitudes
           Mcut               cutoff magnitude
           T1, T2             [days] start and end time of the LL-fit
           stdmin             [km] minimum standard deviation (std) for kernel smoothing of background activity
           Nnearest           value used for kernel smoothing if std(Nnearest) > stdmin
           TmaxTrig           [days] maximum length of aftershock triggering
    Return estimated parameters and LL-value
    '''
    ind = ((mall>= Mcut) & (tall>=T1-TmaxTrig) & (tall<=T2))
    t = tall[ind]
    lat = latall[ind]
    lon = lonall[ind]
    m = mall[ind]
    ind = (t>=T1)
    ti = t[ind]
    lati = lat[ind]
    loni = lon[ind]
    mi = m[ind]
    print('\n\t total events (M>=%.1f): N=%d  (fitted: N=%d)\n' % (Mcut, len(t), len(ti)))
    
    tsteps = LLrecipes.define_tsteps(T1, T2, ti)
    Nind, Nij = LLrecipes.select_triggerevents(t, ti, TmaxTrig)
    Ri = np.zeros((len(lati), len(lat)))
    for i in range(len(lati)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        Ri[i, nij] = LLrecipes.dist2D(lati[i], loni[i], lat[nij], lon[nij])

    print(" --> RESULT for time interval [%.1f  %.1f]:\n" % (T1, T2))
    x0 = setstartparameter(len(ti), A, T1, T2)   
    mu, K, alpha, c, p, d0, gamma, q, b, bDT, LL = LLoptimize(x0, tsteps, t, m, ti, mi, Ri, Mcut, T1, T2, Nind, Nij, A, TmaxTrig)
    print("RESULT:  mu=%f  K=%f  alpha=%f  c=%f   p=%f  d0=%f  gamma=%f   q=%f   LL=%f" % (mu*A, K, alpha, c, p, d0, gamma, q, LL))
    return mu*A, K, alpha, c, p, d0, gamma, q, b, bDT, LL










