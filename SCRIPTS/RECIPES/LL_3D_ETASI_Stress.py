# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This is a Function code for estimating of 3D ETASI parameters by considering 
anisotropic kernel using stress patern for mainshock with uniform horizontal 
background which is depth dependent. This code will use LLrecipes to call most of 
the functions. 
'''

########################## Importing Required Modules #########################
import sys
import numpy as np
import LLrecipes
from scipy.optimize import minimize
eps = 1e-6
day2sec = 24 * 60 * 60.0

############################# Functions & Classes #############################
def calculate_ETASrate(tmain, Namain, pmaini, time, t, r, mu, Na, c, p, d, q, cnorm):
    R = mu + np.sum(Na * np.power(c + time - t, -p) * LLrecipes.fR(r, d, q, cnorm))
    if time > tmain:
        R += pmaini * Namain * np.power(c + time - tmain, -p)
    return R

def update_prob(tmain, Mmain, pmaini, t, z, m, ti, Ri, mui, Mc, c, p, K, alpha, d0, gamma, q, cnorm, Nind, Nij):
    Na = K * np.power(10.0, alpha*(m-Mc))
    Namain = K * np.power(10.0, alpha*(Mmain-Mc))
    d  = d0 * np.power(10., gamma * m)
    prob = np.zeros(len(ti))
    for i in range(len(ti)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        rate = calculate_ETASrate(tmain, Namain, pmaini[i], ti[i], t[nij], Ri[i, nij], mui[i], Na[nij], c, p, d[nij], q, cnorm[nij])
        prob[i] = mui[i] / rate
    return prob

def LL1value(tmain, Namain, t, ti, pmaini, Ri, mui, mufac, muV, Na, Zinteg, c, p, d, q, cnorm, Nind, Nij, bDT):
    NN = len(ti)
    fac = np.zeros(NN)
    R0 = np.zeros(NN)
    for i in range(NN):
        NI = Nind[i]
        nij = Nij[i,:NI]
        fac[i] = LLrecipes.calculate_detectionfactor_2mainshocks(tmain, Namain, 0, 0, ti[i], t[nij], mufac*muV, Zinteg[nij]*Na[nij], c, p, bDT)
        R0[i] = calculate_ETASrate(tmain, Namain, pmaini[i], ti[i], t[nij], Ri[i, nij], mufac*mui[i], Na[nij], c, p, d[nij], q, cnorm[nij])
    LL1 = np.sum(np.log(fac * R0))
    return LL1

def nLLETAS(arguments, tsteps, tmain, Mmain, t, z, m, ti, mi, pmaini, Ri, mui, Nbacktot, Mc, T1, T2, Z1, Z2, Nind, Nij, TmaxTrig):
    mufac = np.square(arguments[0])
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
    cnorm = LLrecipes.determine_normalization_factor_3D(d, q)
    Zinteg = LLrecipes.integration_zrange(d, q, z, Z1, Z2, cnorm)
    Na = K * np.power(10.0, alpha*(m-Mc))
    muV = Nbacktot / (T2 - T1)
    Namain = K * np.power(10.0, alpha*(Mmain-Mc))

    LLM = LLrecipes.LLGR_2mainshocks(tmain, Namain, 0, 0, t, ti, mi, Mc, mufac*muV, Na*Zinteg, c, p, Nind, Nij, b, bDT)
    LL1 = LL1value(tmain, Namain, t, ti, pmaini, Ri, mui, mufac, muV, Na, Zinteg, c, p, d, q, cnorm, Nind, Nij, bDT)
    LL2 = LLrecipes.LL2valueETASI_2mainshocks(tsteps, tmain, Namain, 0, 0, t, mufac*muV, Na*Zinteg, c, p, TmaxTrig, bDT)
    LL = LLM + LL1 - LL2
    nLL = -LL
    if np.isnan(nLL):
        nLL = 1e10
    tfac = 24*60.0
    sys.stdout.write('\r'+str('search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.2f[min]  p=%.2f d0=%.4f  gamma=%.2f  q=%.2f b=%.2f  bDT=%.2f[min]--> Ntot=%.1f nLL=%f\r' % (mufac, K, alpha, tfac*c, p, d0, gamma, q, b, tfac*bDT, len(ti), nLL)))
    sys.stdout.flush()
    #print('search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.2f[min]  p=%.2f d0=%.4f  gamma=%.2f  q=%.2f b=%.2f  bDT=%.2f[min] --> Ntot=%.1f (Z=%d) nLL=%f\n' % (mutot, K, alpha, tfac*c, p, d0, gamma, q, b, tfac*bDT, LL2, len(ti), nLL))
    return nLL

def LLoptimize(x0, tsteps, tmain, Mmain, t, z, m, ti, mi, pmaini, Ri, mui, Nbacktot, Mcut, T1, T2, Z1, Z2, Nind, Nij, TmaxTrig):
    res = minimize(nLLETAS, np.sqrt(x0), args=(tsteps, tmain, Mmain, t, z, m, ti, mi, pmaini, Ri, mui, Nbacktot, Mcut, T1, T2, Z1, Z2, Nind, Nij, TmaxTrig), method='BFGS', options={'gtol': 1e-01})
    mufac = np.square(res.x[0])
    K = np.square(res.x[1])
    alpha = np.square(res.x[2])
    c = np.square(res.x[3])
    p = np.square(res.x[4])
    d0 = np.square(res.x[5])
    gamma = np.square(res.x[6])
    q = np.square(res.x[7])
    b = np.square(res.x[8])
    bDT = np.square(res.x[9])
    return mufac, K, alpha, c, p, d0, gamma, q, b, bDT, -res.fun

def setstartparameter(ti, V, T1, T2):
    mu = 0.1 * len(ti) / (V * (T2 - T1))
    K     = 0.05
    alpha = 1.0                
    c     = 0.01    # [days]
    p     = 1.3
    d0    = 0.013   # ... with d = d0 * 10^(gamma*M) 
    gamma = 0.5     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    q     = 0.7     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    b     = 1.0
    bDT   = 100.0 / (24 * 60 * 60.0)   #  blind time
    
    # mu = 0.1
    # K     = 0.008   
    # alpha = 0.97                 
    # c     = 0.017    # [days]
    # p     = 1.09
    # d0    = 0.015   # ... with d = d0 * 10^(gamma*M) 
    # gamma = 0.4     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    # q     = 0.84     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    # b     = 0.94
    # bDT   = 77
    return np.asarray([mu, K, alpha, c, p, d0, gamma, q, b, bDT])

def determine_ETASparameter(dr, lats, lons, zs, probs, tmain, Mmain, tall, latall, lonall, zall, mall, Mcut, T1, T2, A, Z1, Z2, stdmin, Nnearest, TmaxTrig):
    '''
    Search of the minimum -LL value for the ETAS-model

    Input: dr, lats, lons, zs gridresolution [km], gridpoints of mainshock distributions 
           probi1, prob2      spatial-pdf for the two mainshocks
           tmain1/2, Mmain1/2 mainshock times and magnitudes
           t, lat, lon, z, m  sequence of earthquake times, location and magnitudes
           Mcut               cutoff magnitude
           T1, T2             [days] start and end time of the LL-fit
           A                  [km^2] surface area of fit region
           Z1, Z2             depth interval for LL-fit
           stdmin             [km] minimum standard deviation (std) for kernel smoothing of background activity
           Nnearest           value used for kernel smoothing if std(Nnearest) > stdmin
           TmaxTrig           [days] maximum length of aftershock triggering
    Return estimated parameters and LL-value
    '''
    V = A * (Z2 - Z1)   #  [km^3] target volume
    t, lat, lon, z, m, ti, lati, loni, zi, mi = LLrecipes.select_targetevents_3D(tmain, tall, latall, lonall, zall, mall, Mcut, T1, T2, Z1, Z2, TmaxTrig)
    print('\n\t total events (M>=%.2f): N=%d  (fitted: N=%d) Mmax=%.2f (%.2f in fit-period)\n' % (Mcut, len(t), len(ti), np.max(m), np.max(mi)))

    tsteps = LLrecipes.define_tsteps_mainshock(tmain, T1, T2, ti)
    Nind, Nij = LLrecipes.select_triggerevents(t, ti, TmaxTrig)
    Ri = np.zeros((len(lati), len(lat)))
    pmaini = np.zeros(len(lati))
    dzii = np.zeros((len(lati), len(lati)))
    stdzi = stdmin * np.ones(len(lati))
    cznorm = np.ones(len(lati))
    
    for i in range(len(lati)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        Ri[i, nij] = LLrecipes.dist3D(lati[i], loni[i], zi[i], lat[nij], lon[nij], z[nij])
        i0 = np.argmin(LLrecipes.dist3D(lati[i], loni[i], zi[i], lats, lons, zs))
        pmaini[i] = probs[i0] / np.power(dr, 3.0)
        dzii[i, :] = np.abs(zi[i]-zi)
        dzi = np.sort(dzii[i, :])
        if dzi[Nnearest] > stdzi[i]:
            stdzi[i] = dzi[Nnearest]
        cznorm[i] = LLrecipes.calculate_znorm(zi[i], stdzi[i], Z1, Z2)
    pbacki = 0.5 * np.ones(len(lati))
    mui, Nbacktot = LLrecipes.calculate_mu(cznorm, dzii, stdzi, pbacki, T1, T2, A)
    print(" --> RESULT for time interval [%.1f  %.1f]:\n" % (T1, T2))
    dLL = 10.0
    nround = 0
    x0 = setstartparameter(ti, V, T1, T2)
    while dLL > 0.1:
        nround += 1
        mufac, K, alpha, c, p, d0, gamma, q, b, bDT, LL = LLoptimize(x0, tsteps, tmain, Mmain, t, z, m, ti, mi, pmaini, Ri, mui, Nbacktot, Mcut, T1, T2, Z1, Z2, Nind, Nij, TmaxTrig)
        x0 = np.asarray([1.0, K, alpha, c, p, d0, gamma, q, b, bDT])
        mui *= mufac
        d  = d0 * np.power(10.0, gamma * m)
        cnorm = LLrecipes.determine_normalization_factor_3D(d, q)
        for i in range(10):
            pbacki = update_prob(tmain, Mmain, pmaini, t, z, m, ti, Ri, mui, Mcut, c, p, K, alpha, d0, gamma, q, cnorm, Nind, Nij)
            mui, Nbacktot = LLrecipes.calculate_mu(cznorm, dzii, stdzi, pbacki, T1, T2, A)
        if nround == 1:
            dLL = 10.0
        else:
            dLL = np.abs(LL - LL0)
        LL0 = LL
        tfac = 24*60.0
        print(" nround=%d:  mufac=%f  c=%.2f[min]   p=%f  K=%f  alpha=%f  d0=%f  gamma=%f   q=%f   b=%.2f  bDT=%.2f[min] LL=%f  dLL=%f" % (nround, mufac, tfac*c, p, K, alpha, d0, gamma, q, b, tfac*bDT, LL, dLL))
    mu = Nbacktot/(T2-T1)
    return mu, K, alpha, c, p, d0, gamma, q, b, bDT, LL







