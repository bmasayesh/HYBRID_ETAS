# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This is a Function code for estimating of 3D ETAS parameters by considering 
anisotropic kernel using distance to the fault plane of mainshock with uniform 
horizontal background which is depth dependent. This code will use LLrecipes 
to call most of the functions. 
'''
########################## Importing Required Modules #########################
import sys
import LLrecipes
import numpy as np
from scipy.optimize import minimize
eps = 1e-6

############################# Functions & Classes #############################
def calculate_ETASrate(tmain, Namain, time, rf, df, cf, t, r, mu, Na, c, p, d, q, cnorm):
    R = mu + np.sum(Na * np.power(c + time - t, -p) * LLrecipes.fR(r, d, q, cnorm))
    if time > tmain:
        R += Namain * np.power(c + time - tmain, -p) * LLrecipes.fR(rf, df, q, cf)
    return R

def update_prob(tmain, Mmain, t, z, m, ti, rfi, dfault, rfault, Vgrid, Ri, mui, Mc, c, p, K, alpha, d0, gamma, q, cnorm, Nind, Nij):
    Na = K * np.power(10.0, alpha*(m-Mc))
    Namain = K * np.power(10.0, alpha*(Mmain-Mc))
    d  = d0 * np.power(10., gamma * m)
    prob = np.zeros(len(ti))
    cfault = 1.0 / np.sum(LLrecipes.fR(rfault, dfault, q, 1.0) * Vgrid)
    for i in range(len(ti)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        rate = calculate_ETASrate(tmain, Namain, ti[i], rfi[i], dfault, cfault, t[nij], Ri[i, nij], mui[i], Na[nij], c, p, d[nij], q, cnorm[nij])
        prob[i] = mui[i] / rate
    return prob

def LL1value(tmain, Namain, t, z, m, ti, Ri, rfi, mui, mufac, Na, c, p, d, q, dfault, cnorm, cfault, Nind, Nij):
    LL1 = 0
    NN = len(ti)
    R = np.zeros(NN)
    for i in range(NN):
        NI = Nind[i]
        nij = Nij[i,:NI]
        R[i] = calculate_ETASrate(tmain, Namain, ti[i], rfi[i], dfault, cfault, t[nij], Ri[i, nij], mufac*mui[i], Na[nij], c, p, d[nij], q, cnorm[nij])
    LL1 = np.sum(np.log(R))
    return LL1

def LL2valueETAS_2mainshocks(tmain, Namain, t, z, m, ti, Mc, mu, K, alpha, c, p, d0, gamma, q, T1, T2, Z1, Z2, TmaxTrig):
    d  = d0 * np.power(10.0, gamma * m)
    cnorm = LLrecipes.determine_normalization_factor_3D(d, q)
    Zinteg = LLrecipes.integration_zrange(d, q, z, Z1, Z2, cnorm)
    rho = K * np.power(10.0, alpha*(m-Mc)) * Zinteg
    I = mu + LLrecipes.integrate(t, rho, c, p, T1, T2, TmaxTrig)
    if tmain < T2:
        I += LLrecipes.integratemain(tmain, Namain, c, p, T1, T2, TmaxTrig)
    return I

def nLLETAS(arguments, LL_GR, rfault, Vgrid, V, tmain, Mmain, t, z, m, ti, Ri, rfi, mui, Nbacktot, Mc, T1, T2, Z1, Z2, Nind, Nij, TmaxTrig):
    mufac = np.square(arguments[0])
    K = np.square(arguments[1])
    alpha = np.square(arguments[2])
    c = np.square(arguments[3])
    p = np.square(arguments[4])
    d0 = np.square(arguments[5])
    gamma = np.square(arguments[6])
    q = np.square(arguments[7])
    dfault = np.square(arguments[8])
    d  = d0 * np.power(10.0, gamma * m)
    cnorm = LLrecipes.determine_normalization_factor_3D(d, q)
    Na = K * np.power(10.0, alpha*(m-Mc))
    Namain = K * np.power(10.0, alpha*(Mmain-Mc))
    cfault = 1.0 / np.sum(LLrecipes.fR(rfault, dfault, q, 1.0) * Vgrid)

    LL1 = LL1value(tmain, Namain, t, z, m, ti, Ri, rfi, mui, mufac, Na, c, p, d, q, dfault, cnorm, cfault, Nind, Nij)
    LL2 = LL2valueETAS_2mainshocks(tmain, Namain, t, z, m, ti, Mc, mufac*Nbacktot, K, alpha, c, p, d0, gamma, q, T1, T2, Z1, Z2, TmaxTrig)
    
    LL = LL_GR + LL1 -LL2
    nLL = -LL
    if np.isnan(nLL):
        nLL = 1e10
    sys.stdout.write('\r'+str('search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.4f  p=%.2f d0=%.4f  gamma=%.2f  q=%.2f dfault=%.2f --> Ntot=%.1f nLL=%f\r' % (mufac, K, alpha, c, p, d0, gamma, q, dfault, len(ti), nLL)))
    sys.stdout.flush()
    return nLL

def LLoptimize(LL_GR, x0, rfault, Vgrid, tmain, Mmain, t, z, m, ti, Ri, rfi, mui, Nbacktot, Mcut, T1, T2, V, Z1, Z2, Nind, Nij, TmaxTrig):
    res = minimize(nLLETAS, np.sqrt(x0), args=(LL_GR, rfault, Vgrid, V, tmain, Mmain, t, z, m, ti, Ri, rfi, mui, Nbacktot, Mcut, T1, T2, Z1, Z2, Nind, Nij, TmaxTrig), method='BFGS', options={'gtol': 1e-01})
    mu = np.square(res.x[0])
    K = np.square(res.x[1])
    alpha = np.square(res.x[2])
    c = np.square(res.x[3])
    p = np.square(res.x[4])
    d0 = np.square(res.x[5])
    gamma = np.square(res.x[6])
    q = np.square(res.x[7])
    dfault = np.square(res.x[8])
    return mu, K, alpha, c, p, d0, gamma, q, dfault, -res.fun

def setstartparameter(ti, V, T1, T2):
    # mu = 0.1 * len(ti) / (V * (T2 - T1))
    # K     = 0.05   
    # alpha = 1.0                
    # c     = 0.01    # [days]
    # p     = 1.3
    # d0    = 0.013   # ... with d = d0 * 10^(gamma*M) 
    # gamma = 0.5     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    # q     = 0.7     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    # dfault = 1.0    # for distance-to-fault decay of mainshock-triggered events
    
    mu = 0.24
    K     = 0.059   
    alpha = 0.52                
    c     = 0.039    # [days]
    p     = 0.99
    d0    = 0.015   # ... with d = d0 * 10^(gamma*M) 
    gamma = 0.37     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    q     = 0.87     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    dfault = 1.0    # for distance-to-fault decay of mainshock-triggered events
    
    return np.asarray([mu, K, alpha, c, p, d0, gamma, q, dfault])

def determine_ETASparameter(latf, lonf, zf, rfault, tmain, Mmain, tall, latall, lonall, zall, mall, Mcut, T1, T2, A, Z1, Z2, stdzmin, Nnearest, TmaxTrig):
    '''
    Search of the minimum -LL value for the ETAS-model

    Input: latf, lonf, zf, rfault1/2 gridpoints and distance to faults
           tmain1/2, Mmain1/2        mainshock times and magnitudes
           t, lat, lon, z, m  sequence of earthquake times, location and magnitudes
           Mcut               cutoff magnitude
           T1, T2             [days] start and end time of the LL-fit
           A                  [km^2] surface area of fit region
           Z1, Z2             depth interval for LL-fit
           TmaxTrig           [days] maximum length of aftershock triggering
    Return estimated parameters and LL-value
    '''
    ind = ((mall>Mcut) & (tall>=T1) & (zall>=Z1) & (zall<=Z2))
    mm = mall[ind]
    ## Log-Likelihood of GR by maximum likelihood estimation of b-value
    N, b, bstd, LL_GR = LLrecipes.calculate_N_and_bvalue(mm, Mcut)
    print("\n\t GR Maximum Likelihood fit: b=%.2f +- %.2f  LL_GR = %s\n" % (b, bstd, LL_GR))
    
    # cell area of the fault-distance grid:
    llat = np.unique(latf)
    llon = np.unique(lonf)
    zz = np.unique(zf)
    Agrid = LLrecipes.dist3D(llat[1], np.mean(llon), 0, llat[0], np.mean(llon), 0)
    Agrid *= LLrecipes.dist3D(np.mean(llat), llon[1], 0, np.mean(llat), llon[0], 0)
    Vgrid = Agrid * (zz[1] - zz[0])
    V = A * (Z2 - Z1)   #  [km^3] target volume
    t, lat, lon, z, m, ti, lati, loni, zi, mi = LLrecipes.select_targetevents_3D(tmain, tall, latall, lonall, zall, mall, Mcut, T1, T2, Z1, Z2, TmaxTrig)
    print('\n\t total events (M>=%.2f): N=%d  (fitted: N=%d) Mmax=%.2f (%.2f in fit-period)\n' % (Mcut, len(t), len(ti), np.max(m), np.max(mi)))
    Nind, Nij = LLrecipes.select_triggerevents(t, ti, TmaxTrig)
    rfi = np.zeros(len(lati))
    Ri = np.zeros((len(lati), len(lat)))
    dzii = np.zeros((len(lati), len(lati)))
    stdzi = stdzmin * np.ones(len(lati))
    cznorm = np.ones(len(lati))
    for i in range(len(lati)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        Ri[i, nij] = LLrecipes.dist3D(lati[i], loni[i], zi[i], lat[nij], lon[nij], z[nij])
        dzii[i, :] = np.abs(zi[i]-zi)
        dzi = np.sort(dzii[i, :])
        if dzi[Nnearest] > stdzi[i]:
            stdzi[i] = dzi[Nnearest]
        cznorm[i] = LLrecipes.calculate_znorm(zi[i], stdzi[i], Z1, Z2)
        imin = np.argmin(LLrecipes.dist3D(lati[i], loni[i], zi[i], latf, lonf, zf))
        rfi[i] = rfault[imin]
    pbacki = 0.5 * np.ones(len(lati))
    mui, Nbacktot = LLrecipes.calculate_mu(cznorm, dzii, stdzi, pbacki, T1, T2, A)
    print(" --> RESULT for time interval [%.1f  %.1f]:\n" % (T1, T2))
    dLL = 10.0
    nround = 0
    x0 = setstartparameter(ti, V, T1, T2)
    while dLL > 0.1:
        nround += 1
        mufac, K, alpha, c, p, d0, gamma, q, dfault, LL = LLoptimize(LL_GR, x0, rfault, Vgrid, tmain, Mmain, t, z, m, ti, Ri, rfi, mui, Nbacktot, Mcut, T1, T2, V, Z1, Z2, Nind, Nij, TmaxTrig)  
        x0 = np.asarray([1.0, K, alpha, c, p, d0, gamma, q, dfault])
        mui *= mufac
        d  = d0 * np.power(10.0, gamma * m)
        cnorm = LLrecipes.determine_normalization_factor_3D(d, q)
        for i in range(10):
            pbacki = update_prob(tmain, Mmain, t, z, m, ti, rfi, dfault, rfault, Vgrid, Ri, mui, Mcut, c, p, K, alpha, d0, gamma, q, cnorm, Nind, Nij) 
            mui, Nbacktot = LLrecipes.calculate_mu(cznorm, dzii, stdzi, pbacki, T1, T2, A)
        if nround == 1:
            dLL = 10.0
        else:
            dLL = np.abs(LL - LL0)
        LL0 = LL 
        print(" nround=%d:  mufac=%f  c=%f   p=%f  K=%f  alpha=%f  d0=%f  gamma=%f   q=%f   LL=%f  dLL=%f" % (nround, mufac, c, p, K, alpha, d0, gamma, q, LL, dLL))
        
    mu = Nbacktot/(T2-T1)
    return mu, K, alpha, c, p, d0, gamma, q, dfault, b, LL


