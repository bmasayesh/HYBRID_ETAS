# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This code estimates standard 2D ETASI parameters for 6 large earthquakes in California. 
Background rate is horizontally  uniform.
'''

########################## Importing Required Modules #########################
import sys
import numpy as np
sys.path.append('RECIPES')
import LL_2D_ETASI

day2sec = 24 * 60 * 60.0

############################# Functions & Classes #############################
def estimate_parameters_2D_ETASI(Paradir, name, t, lat, lon, m, Mcut, T1, T2, TmaxTrig, A):
    mu, K, alpha, c, p, d0, gamma, q, b, bDT, LLvalue = LL_2D_ETASI.determine_ETASparameter(t, lat, lon, m, Mcut, T1, T2, A, TmaxTrig)
    suff = '2DstandardETASI-%s-Mcut%.2f-T%.1f-%.2f' % (name, Mcut, T1, T2)
    # OUTPUTS:
    outname = '%s/parameter-%s.out' % (Paradir, suff)
    f = open(outname, 'w')
    f.write('#     mutot            K             alpha         c_[days]            p            d0_[km]          gamma             q              b              Tb_[s]            LL\n')
    f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (mu, K, alpha, c, p, d0, gamma, q, b, day2sec*bDT, LLvalue))
    f.close()
    print('\n\t OUTPUT: %s' % (outname))
    return

####################### Inputs and Outputs  directories #######################
Seqdir  = '../INPUTS/CATALOGS/SEQUENCES'              ## directory of sequences
Paradir = '../OUTPUTS/PARAMETERS/2D ETAS/ETASI'       ## directory of parameters 

########################### Declaring of Parameters ###########################
Mcut = 1.95
Z1 = 0.0
Z2 = 30
stdmin = 0.5                # [km] minimum smoothing kernel
Nnearest = 5

R = 100.0                   # Radius [km]
T0 = -300.0                 # days befor mainshock
T1 = -100.0                 # [days] start time of the LL-fit
T2 = 100.0                  # [days] end time of the LL-fit
TmaxTrig = 1000.0           # [days] maximum length of triggering
A = np.pi * np.square(R)    # [km^2] surface area of the target area

### ===========================================================================
'''
First we introduce the name of the large events or sequences and then choose the 
sequence for calculating ETAS parameters
'''
names = ['SuperstitionHill', 'Landers', 'Northridge', 'HectorMine', 'BajaCalifornia', 'Ridgecrest']

### ===================== 2D ETAS parameter esimation =========================
for i in range(4, 5):         ## Here we just chose Baja California       
    name = names[i]
    print('\n\t Estimation of 2D ETASI parameters for %s sequence' % (name))
    data = np.loadtxt('%s/california_1981_2022-Mcut%.2f-R100.0km-T%.0f-%.0fdays-%s.kat' % (Seqdir, Mcut, T0, T2, name), skiprows=1)
    Mmain = data[0, 5]
    data = data[1:, :]
    ind = ((data[:, 4]>= Z1) & (data[:, 4]<=Z2))
    t = data[ind, 1]
    lat = data[ind, 2]
    lon = data[ind, 3]
    m = data[ind, 5]
    estimate_parameters_2D_ETASI(Paradir, name, t, lat, lon, m, Mcut, T1, T2, TmaxTrig, A)
