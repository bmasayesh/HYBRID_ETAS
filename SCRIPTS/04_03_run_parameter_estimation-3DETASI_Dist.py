# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
We estimate 3D ETASI parameters by considering distance to the fault plane of the 
mainshock insteed of distance to the hypocenter of mainshocks for 6 large earthquakes
in California.
'''

########################## Importing Required Modules #########################
import sys
import numpy as np
sys.path.append('RECIPES')
import LL_3D_ETASI_Distance as LLETASI_R

day2sec = 24 * 60 * 60.0

############################# Functions & Classes #############################
def read_slipmodelnames(name):
    # slip models:
    if name == 'SuperstitionHill':
        slipmodels = ['s1987SUPERS01LARS', 's1987SUPERS01WALD']
    elif name == 'Landers':    
        slipmodels = ['s1992LANDER01WALD', 's1992LANDER01COHE',  's1992LANDER01COTT', 's1992LANDER01HERN', 's1992LANDER01ZENG']
    elif name == 'Northridge':
        slipmodels = ['s1994NORTHR01DREG', 's1994NORTHR01HART', 's1994NORTHR01HUDN', 's1994NORTHR01SHEN', 's1994NORTHR01WALD', 's1994NORTHR01ZENG']
    elif name == 'HectorMine':
        slipmodels = ['s1999HECTOR01JIxx', 's1999HECTOR01JONS', 's1999HECTOR01KAVE', 's1999HECTOR01SALI']
    elif name == 'BajaCalifornia':
        slipmodels = ['s2010ELMAYO01MEND', 's2010ELMAYO01WEIx']  
    elif name == 'Ridgecrest':
        slipmodels = ['s2019RIDGEC02JINx', 's2019RIDGEC02ROSS', 's2019RIDGEC04XUxx']
    return slipmodels

def read_distance2fault(Distdir, name, slipmodels, dr):
    datname = '%s/distance2fault-%s-dr%.1fkm.out' % (Distdir, slipmodels[0], dr)
    data = np.loadtxt(datname, skiprows=1)
    lati = data[:, 0]
    loni = data[:, 1]
    zi = data[:, 2]
    dist = data[:, 3]
    for ns in range(1, len(slipmodels)):
        datname = '%s/distance2fault-%s-dr%.1fkm.out' % (Distdir, slipmodels[ns], dr)
        data = np.loadtxt(datname, skiprows=1)
        dist += data[:, 3]
            
    norm = 1.0 * len(slipmodels)
    dist /= norm

    return lati, loni, zi, dist

def estimate_parameters_ETASI_3D_distance_to_fault(Paradir, name, latf, lonf, zf, rfault, tmain, Mmain, t, lat, lon, z, m, Mcut, T1, T2, A, Z1, Z2, stdmin, Nnearest, TmaxTrig):
    mutot, K, alpha, c, p, d0, gamma, q, dfault, b, bDT, LLvalue = LLETASI_R.determine_ETASparameter(latf, lonf, zf, rfault, tmain, Mmain, t, lat, lon, z, m, Mcut, T1, T2, A, Z1, Z2, stdmin, Nnearest, TmaxTrig)
    suff = '3D_ETASI-distance-to-fault-%s-Mcut%.2f-T%.1f-%.2f' % (name, Mcut, T1, T2)
    # OUTPUTS:
    outname = '%s/parameter-%s.out' % (Paradir, suff)
    f = open(outname, 'w')
    f.write('#     mutot            K             alpha         c_[days]            p            d0_[km]          gamma             q              dfault          b              Tb_[s]           LL\n')
    f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (mutot, K, alpha, c, p, d0, gamma, q, dfault, b, day2sec*bDT, LLvalue))
    f.close()
    print('\n\t OUTPUT: %s' % (outname))
    return

####################### Inputs and Outputs  directories #######################
Seqdir  = '../INPUTS/CATALOGS/SEQUENCES'               ## directory of sequences
Paradir = '../OUTPUTS/PARAMETERS/3D ETAS/ETASI_R'      ## directory of parameters 
Distdir = '../OUTPUTS/DISTANCE-RESULTS/3D'             ## directory of stress results

########################### Declaring of Parameters ###########################
Mcut = 1.95
Z1 = 0.0
Z2 = 30
stdmin = 0.5                         # [km] minimum smoothing kernel
Nnearest = 5

R = 100.0                            # [km]
T0 = -300.0                          # days befor mainshock
T1 = -100.0                          # [days] start time of the LL-fit
T2 = 100.0                           # [days] end time of the LL-fit
TmaxTrig = 1000.0                    # [days] maximum length of triggering

A = np.pi * np.square(R)
dr = 1.0                             # [km] spatial grid spacing in x, y, z direction
tmain = 0.0                          # [days] time of the mainshock

### ===========================================================================
'''
First we introduce the name of the large events or sequences and then choose the 
sequence for calculating ETAS parameters
'''
names = ['SuperstitionHill', 'Landers', 'Northridge', 'HectorMine', 'BajaCalifornia', 'Ridgecrest']

##=========== 3D ETASI+Distance to fault plane parameter estimation ===========
for i in range(4, 5):         ## Here we just chose Baja California      
    name = names[i]
    slipmodels = read_slipmodelnames(name)
    lati, loni, zi, dist = read_distance2fault(Distdir, name, slipmodels, dr)
    print('\n\t Estimation of 3D ETASI + R parameter for %s sequence' % (name))
    data = np.loadtxt('%s/california_1981_2022-Mcut%.2f-R100.0km-T%.0f-%.0fdays-%s.kat' % (Seqdir, Mcut, T0, T2, name), skiprows=1)
    Mmain = data[0, 5]
    t = data[1:, 1]
    lat = data[1:, 2]
    lon = data[1:, 3]
    z = data[1:, 4]
    m = data[1:, 5]
    estimate_parameters_ETASI_3D_distance_to_fault(Paradir, name, lati, loni, zi, dist, tmain, Mmain, t, lat, lon, z, m, Mcut, T1, T2, A, Z1, Z2, stdmin, Nnearest, TmaxTrig)
    
    
    
