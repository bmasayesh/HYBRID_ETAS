# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This is a Function code for estimating of 2D ETASI parameters by considering 
anisotropic kernel using distance to the fault plane of mainshock (34%) and 
and stress scalar from mainshock (66%) with uniform horizontal background. 
This code will use LLrecipes to call most of the functions. 
'''
########################## Importing Required Modules #########################
import sys
import LLrecipes
import numpy as np
from scipy.optimize import minimize

eps = 1e-6
day2sec = 24 * 60 * 60.0

############################# Functions & Classes #############################
def fR(r, d, q):
    cnorm = (q / np.pi) * np.power(d, 2*q)
    fr = cnorm * np.power(np.square(d) + np.square(r), -(1+q))
    return fr

def calculate_ETASrate(tmain, Namain, time, rf, fri, df, cf, t, r, mu, Na, c, p, d, q):    
    R = mu + np.sum(Na * np.power(c + time - t, -p) * fR(r, d, q))
    if time > tmain:
        R += Namain * np.power(c + time - tmain, -p) * ((0.34 * cf * fR(rf, df, q)) + (0.66 * fri))
    return R

def LL1value(tmain, Namain, t, ti, Ri, rfi, pmaini, mu, muA, Na, c, p, d, q, dfault, cfault, Nind, Nij, bDT):
    NN = len(ti)
    fac = np.zeros(NN)
    R0 = np.zeros(NN)
    for i in range(NN):
        NI = Nind[i]
        nij = Nij[i,:NI]
        fac[i] = LLrecipes.calculate_detectionfactor_2mainshocks(tmain, Namain, 0, 0, ti[i], t[nij], muA, Na[nij], c, p, bDT)
        R0[i] = calculate_ETASrate(tmain, Namain, ti[i], rfi[i], pmaini[i], dfault, cfault, t[nij], Ri[i, nij], mu, Na[nij], c, p, d[nij], q)
    LL1 = np.sum(np.log(fac * R0))
    return LL1

def nLLETAS(arguments, rfault, A, tsteps, tmain, Mmain, t, m, ti, mi, Ri, rfi, pmaini, Mc, T1, T2, Nind, Nij, TmaxTrig):
    mu = np.square(arguments[0])
    K = np.square(arguments[1])
    alpha = np.square(arguments[2])
    c = np.square(arguments[3])
    p = np.square(arguments[4])
    d0 = np.square(arguments[5])
    gamma = np.square(arguments[6])
    q = np.square(arguments[7])
    dfault = np.square(arguments[8])
    b = np.square(arguments[9])
    bDT = np.square(arguments[10])
    
    d  = d0 * np.power(10.0, gamma * m)
    Na = K * np.power(10.0, alpha*(m-Mc))
    muA = mu * A  
    
    Namain = K * np.power(10.0, alpha*(Mmain-Mc))
    Agrid = A / len(rfault)  # Test that this is really dr^2 = 1
    cfault = 1.0 / np.sum(Agrid * fR(rfault, dfault, q))

    LLM = LLrecipes.LLGR_2mainshocks(tmain, Namain, 0, 0, t, ti, mi, Mc, muA, Na, c, p, Nind, Nij, b, bDT)
    LL1 = LL1value(tmain, Namain, t, ti, Ri, rfi, pmaini, mu, muA, Na, c, p, d, q, dfault, cfault, Nind, Nij, bDT)
    LL2 = LLrecipes.LL2valueETASI_2mainshocks(tsteps, tmain, Namain, 0, 0, t, muA, Na, c, p, TmaxTrig, bDT)
    LL = LLM + LL1 - LL2
    nLL = -LL
    if np.isnan(nLL):
        nLL = 1e10
    tfac = 24*60.0
    sys.stdout.write('\r'+str('search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.2f[min]  p=%.2f d0=%.4f  gamma=%.2f  q=%.2f  dfault=%.2f b=%.2f  bDT=%.2f[min]--> Ntot=%.1f nLL=%f\r' % (mu, K, alpha, tfac*c, p, d0, gamma, q, dfault, b, tfac*bDT, len(ti), nLL)))
    sys.stdout.flush()
    return nLL

def LLoptimize(x0, rfault, tsteps, tmain, Mmain, t, m, ti, mi, Ri, rfi, pmaini, Mcut, T1, T2, A, Nind, Nij, TmaxTrig):  
    res = minimize(nLLETAS, np.sqrt(x0), args=(rfault, A, tsteps, tmain, Mmain, t, m, ti, mi, Ri, rfi, pmaini, Mcut, T1, T2, Nind, Nij, TmaxTrig), method='BFGS', options={'gtol': 1e-01})
    mu = np.square(res.x[0])
    K = np.square(res.x[1])
    alpha = np.square(res.x[2])
    c = np.square(res.x[3])
    p = np.square(res.x[4])
    d0 = np.square(res.x[5])
    gamma = np.square(res.x[6])
    q = np.square(res.x[7])
    dfault = np.square(res.x[8])
    b = np.square(res.x[9])
    bDT = np.square(res.x[10])
    return mu, K, alpha, c, p, d0, gamma, q, dfault, b, bDT, -res.fun

def setstartparameter(A):
    mu = 1.0 / A 
    K     = 0.05
    alpha = 1.0                
    c     = 0.01    # [days]
    p     = 1.3
    d0    = 0.013   # ... with d = d0 * 10^(gamma*M) 
    gamma = 0.5     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    q     = 0.7     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    dfault = 1.0    # for distance-to-fault decay of mainshock-triggered events
    b     = 1.0
    bDT   = 100.0 / (24 * 60 * 60.0)   #  blind time
    
    # mu = 0.097 
    # K     = 0.049
    # alpha = 0.43               
    # c     = 0.002    # [days]
    # p     = 0.96
    # d0    = 0.018   # ... with d = d0 * 10^(gamma*M) 
    # gamma = 0.4     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    # q     = 0.46     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    # dfault = 5    # for distance-to-fault decay of mainshock-triggered events
    # b     = 0.8
    # bDT   = 10   #  blind time
    return np.asarray([mu, K, alpha, c, p, d0, gamma, q, dfault, b, bDT])

def determine_ETASparameter(latf, lonf, rfault, probs, tmain, Mmain, tall, latall, lonall, mall, Mcut, T1, T2, A, TmaxTrig):
    '''
    Search of the minimum -LL value for the ETAS-model

    Input: latf, lonf, rfault1/2   gridpoints and distance to fault
           tmain1/2, Mmain1/2          mainshock times and magnitudes
           t, lat, lon, m    sequence of earthquake times, location and magnitudes
           Mcut                 cutoff magnitude
           T1, T2               [days] start and end time of the LL-fit
           A                    [km^2] surface area of fit region
           TmaxTrig             [days] maximum length of aftershock triggering
    Return estimated parameters and LL-value
    '''
    t, lat, lon, m, ti, lati, loni, mi = LLrecipes.select_targetevents_2D(tmain, tall, latall, lonall, mall, Mcut, T1, T2, TmaxTrig)
    print('\n\t total events (M>=%.2f): N=%d  (fitted: N=%d) Mmax=%.2f (%.2f in fit-period)\n' % (Mcut, len(t), len(ti), np.max(m), np.max(mi)))
    
    print(" --> RESULT for time interval [%.1f  %.1f]:\n" % (T1, T2))
    tsteps = LLrecipes.define_tsteps_mainshock(tmain, T1, T2, ti)
    Nind, Nij = LLrecipes.select_triggerevents(t, ti, TmaxTrig)
    Ri = np.zeros((len(lati), len(lat)))
    pmaini = np.zeros(len(lati))
    rfi = np.zeros(len(lati)) 
    for i in range(len(lati)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        Ri[i, nij] = LLrecipes.dist2D(lati[i], loni[i], lat[nij], lon[nij])
        imin = np.argmin(LLrecipes.dist2D(lati[i], loni[i], latf, lonf))
        pmaini[i] = probs[imin]
        rfi[i] = rfault[imin]
        
    x0 = setstartparameter(A)
    mu, K, alpha, c, p, d0, gamma, q, dfault, b, bDT, LL = LLoptimize(x0, rfault, tsteps, tmain, Mmain, t, m, ti, mi, Ri, rfi, pmaini, Mcut, T1, T2, A, Nind, Nij, TmaxTrig)
    tfac = 24*60.0
    print("RESULTS:  mu=%f  c=%f   p=%f  K=%f  alpha=%f  d0=%f  gamma=%f   q=%f   b=%.2f  bDT=%.2f[min] LL=%f" % (mu, tfac*c, p, K, alpha, d0, gamma, q, b, tfac*bDT, LL))
    return mu*A, K, alpha, c, p, d0, gamma, q, dfault, b, bDT, LL


