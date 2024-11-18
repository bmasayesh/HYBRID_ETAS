# GFZ
'''
We estimate 2D ETASI parameters by considering distance to the fault plane of the 
mainshock insteed of distance to the hypocenter of the mainshocks for 6 large 
earthquakes in California.
'''

########################## Importing Required Modules #########################
import sys
import numpy as np
sys.path.append('RECIPES')
import LL_2D_ETASI_Distance_Stress as LLETASI_R_S

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

def read_stress_dist(Stressdir, Distdir, name, slipmodels, stressvalue, meanstressmodel, stressmax, dr):
    datname = '%s/stressmetrics-%s-dr%.1fkm.out' % (Stressdir, slipmodels[0], dr)
    data = np.loadtxt(datname, skiprows=1)
    data1 = data[data[:, 2]==0.5]
    lati = data1[:, 0]
    loni = data1[:, 1]
    CFS = data[:, 3]
    VM = data[:, 4]
    MS = data[:, 5]
    VMS = data[:, 6]
    if meanstressmodel:
        for ns in range(1, len(slipmodels)):
            datname = '%s/stressmetrics-%s-dr%.1fkm.out' % (Stressdir, slipmodels[ns], dr)
            data = np.loadtxt(datname, skiprows=1)
            CFS += data[:, 3]
            VM += data[:, 4]
            MS += data[:, 5]
            VMS += data[:, 6]
    norm = 1.0 * len(slipmodels)
    CFS /= norm
    CFS1 = CFS.reshape(-1, 30)
    CFS1[CFS1 < 0] = 0
    CFS_mean = np.clip(np.mean(CFS1, axis=1), 0, stressmax)
    VM /= norm
    VM1 = VM.reshape(-1, 30)
    VM1[VM1 < 0] = 0
    VM_mean = np.clip(np.mean(VM1, axis=1), 0, stressmax)
    MS /= norm
    MS1 = MS.reshape(-1, 30)
    MS1[MS1 < 0] = 0
    MS_mean = np.clip(np.mean(MS1, axis=1), 0, stressmax)
    VMS /= norm
    VMS1 = VMS.reshape(-1, 30)
    VMS1[VMS1 < 0] = 0 
    VMS_mean = np.clip(np.mean(VMS1, axis=1), 0, stressmax)
    if stressvalue == 'CFS':
        stress = CFS_mean
    elif stressvalue == 'VM':
        stress = VM_mean
    elif stressvalue == 'MS':
        stress = MS_mean
    elif stressvalue == 'VMS':
        stress = VMS_mean
    pstress = stress * np.heaviside(stress, np.zeros(len(stress)))
    # probi = np.cumsum(pstress) / np.sum(pstress)
    probi = pstress / (np.sum(pstress) * np.power(dr, 2.0))
        
    datname = '%s/distance2fault-%s-dr%.1fkm.out' % (Distdir, slipmodels[0], dr)
    data = np.loadtxt(datname, skiprows=1)
    lati = data[:, 0]
    loni = data[:, 1]
    # zi = data[:, 2]
    dist = data[:, 2]
    for ns in range(1, len(slipmodels)):
        datname = '%s/distance2fault-%s-dr%.1fkm.out' % (Distdir, slipmodels[ns], dr)
        data = np.loadtxt(datname, skiprows=1)
        dist += data[:, 2]
            
    norm = 1.0 * len(slipmodels)
    dist /= norm

    return lati, loni, probi, dist

def estimate_parameters_2D_ETASI_DIST_Stress(Paradir, name, latf, lonf, rfault, probi, tmain, Mmain, t, lat, lon, m, Mcut, T1, T2, A, TmaxTrig):
    mutot, K, alpha, c, p, d0, gamma, q, dfault, b, bDT, LLvalue = LLETASI_R_S.determine_ETASparameter(latf, lonf, rfault, probi, tmain, Mmain, t, lat, lon, m, Mcut, T1, T2, A, TmaxTrig)
    suff = '2D_ETASI-distance-%s-%s-Mcut%.2f-T%.1f-%.2f' % (stressvalue, name, Mcut, T1, T2)
    # OUTPUTS:
    outname = '%s/ETASI_DIST_%s/parameter-%s.out' % (Paradir, stressvalue, suff)
    f = open(outname, 'w')
    f.write('#     mutot            K             alpha         c_[days]            p            d0_[km]          gamma           q           dfault           b              Tb_[s]           LL\n')
    f.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (mutot, K, alpha, c, p, d0, gamma, q, dfault, b, day2sec*bDT, LLvalue))
    f.close()
    print('\n\t OUTPUT: %s' % (outname))
    return

####################### Inputs and Outputs  directories #######################
Seqdir  = '../INPUTS/CATALOGS/SEQUENCES'                      ## directory of sequences
Paradir = '../OUTPUTS/PARAMETERS/2D ETAS'                     ## directory of parameters 
Stressdir = '../OUTPUTS/STRESS-RESULTS'                       ## directory of stress results
Distdir = '../OUTPUTS/DISTANCE-RESULTS/ALL-SUBFAULTS/2D'      ## directory of stress results

########################### Declaring of Parameters ###########################
Mcut = 1.95
Z1 = 0.0
Z2 = 30
stdmin = 0.5        # [km] minimum smoothing kernel
Nnearest = 5

R = 100.0           # [km]
T0 = -300.0         # days befor mainshock
T1 = -100.0         # [days] start time of the LL-fit
T2 = 100.0          # [days] end time of the LL-fit
TmaxTrig = 1000.0   # [days] maximum length of triggering

A = np.pi * np.square(R)
dr = 1.0           # [km] spatial grid spacing in x, y, z direction
stressmax = 1e7    # [Pa] ... 10 MPa
tmain = 0.0        # [days] time of the mainshock

### ===========================================================================
'''
First we introduce the name of the large events or sequences and then choose the 
sequence for calculating ETAS parameters
'''
names = ['SuperstitionHill', 'Landers', 'Northridge', 'HectorMine', 'BajaCalifornia', 'Ridgecrest']

### ===========================================================================
'''
Then we select the stress scalar and consider mean of that stress scalar
'''
#stressvalues = ['CFS', 'VM', 'MS', 'VMS']
stressvalue='MS'
meanstressmodel = True

##################### 2D ETASI + Stress estimation (using mainshock stress):
'''
For this part we need stress scalar due to each mainshock in each sequence then
estimate ETAS parameters.
'''
for i in range(0, 6):  
    name = names[i]
    slipmodels = read_slipmodelnames(name)
    lati, loni, probi, dist = read_stress_dist(Stressdir, Distdir, name, slipmodels, stressvalue, meanstressmodel, stressmax, dr)
    print('\n\t Estimation of 2D_ETASI_DIST + %s parameter for %s sequence\n' % (stressvalue, name))
    data = np.loadtxt('%s/california_1981_2022-Mcut%.2f-R100.0km-T%.0f-%.0fdays-%s.kat' % (Seqdir, Mcut, T0, T2, name), skiprows=1)
    Mmain = data[0, 5]
    data = data[1:, :]
    ind = ((data[:, 4]>= Z1) & (data[:, 4]<=Z2))
    t = data[ind, 1]
    lat = data[ind, 2]
    lon = data[ind, 3]
    m = data[ind, 5]
    estimate_parameters_2D_ETASI_DIST_Stress(Paradir, name, lati, loni, dist, probi, tmain, Mmain, t, lat, lon, m, Mcut, T1, T2, A, TmaxTrig)





