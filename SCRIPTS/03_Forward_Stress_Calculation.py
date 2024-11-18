## GFZ February 2024
'''
'''

########################## Importing Required Modules #########################
import sys
sys.path.append('RECIPES')
import read_fsp_file
import stressmetrics
import stress_calculation
import numpy as np
import matplotlib.pyplot as plt

####################### Inputs and Outputs  directories #######################
Seqdir  = '../INPUTS/CATALOGS/SEQUENCES'    ## directory of sequences
Slipdir  = '../INPUTS/SLIP-MODELS'          ## directory of slip models
Stressdir = '../OUTPUTS/STRESS-RESULTS'     ## directory of stress results

########################### Declaring of Parameters ###########################
# Rock parameters
poisson_ratio = 0.25
mu = 3e10            # [Pa]
lmda = 2 * mu * poisson_ratio / (1 - 2 * poisson_ratio)
alpha = (lmda + mu) / (lmda + 2 * mu)
skempton = 0.75
f = 0.7

############################# Functions & Classes #############################
def dist(lat1, lon1, lat2, lon2):
    """
    Distance (in [km]) between points given as [lat,lon]
    """
    R0 = 6367.3
    R = R0 * np.arccos(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2)))
    return R

def define_grid(Seqdir, dr, name):
    Mcut = 1.95
    Z1 = 0.0
    Z2 = 30.0
    T1 = -300.0   # [days] start time of the sequence
    T2 = 100.0    # [days] end time of the sequence
    data = np.loadtxt('%s/california_1981_2022-Mcut%.2f-R100.0km-T%.0f-%.0fdays-%s.kat' % (Seqdir, Mcut, T1, T2, name), skiprows=1)
    latmain = data[0, 2]
    lonmain = data[0, 3]
    lat2km = dist(latmain-0.5, lonmain, latmain+0.5, lonmain)
    lon2km = dist(latmain, lonmain-0.5, latmain, lonmain+0.5)
    gridr = np.arange(0.5*dr, 100, dr)
    gridlat = np.sort(np.append(latmain-gridr/lat2km, latmain+gridr/lat2km))
    gridlon = np.sort(np.append(lonmain-gridr/lon2km, lonmain+gridr/lon2km))
    gridz = np.arange(Z1+0.5*dr, Z2, dr)
    Nlat = len(gridlat)
    Nlon = len(gridlon)
    Nz = len(gridz)
    lati = np.zeros(Nlat*Nlon*Nz)
    loni = np.zeros(Nlat*Nlon*Nz)
    zi = np.zeros(Nlat*Nlon*Nz)
    n = 0
    for i in range(Nlat):
        for j in range(Nlon):
            for k in range(Nz):
                lati[n] = gridlat[i]
                loni[n] = gridlon[j]
                zi[n] = gridz[k]
                n += 1
    return gridlat, gridlon, gridz, lati, loni, zi

def define_grid_circle(Seqdir, dr, name):
    Mcut = 1.95
    Z1 = 0.0
    Z2 = 30.0
    R = 100.0     # [km]
    T1 = -300.0   # [days] start time of the sequence
    T2 = 100.0    # [days] end time of the sequence
    data = np.loadtxt('%s/california_1981_2022-Mcut%.2f-R100.0km-T%.0f-%.0fdays-%s.kat' % (Seqdir, Mcut, T1, T2, name), skiprows=1)
    latmain = data[0, 2]
    lonmain = data[0, 3]
    lat2km = dist(latmain-0.5, lonmain, latmain+0.5, lonmain)
    lon2km = dist(latmain, lonmain-0.5, latmain, lonmain+0.5)
    gridr = np.arange(0.5*dr, 100, dr)
    gridlat = np.sort(np.append(latmain-gridr/lat2km, latmain+gridr/lat2km))
    gridlon = np.sort(np.append(lonmain-gridr/lon2km, lonmain+gridr/lon2km))
    gridz = np.arange(Z1+0.5*dr, Z2, dr)
    Nlat = len(gridlat)
    Nlon = len(gridlon)
    Nz = len(gridz)
    lats = np.zeros(Nlat*Nlon)
    lons = np.zeros(Nlat*Nlon)
    n = 0
    for i in range(Nlat):
        for j in range(Nlon):
            lats[n] = gridlat[i]
            lons[n] = gridlon[j]
            n += 1
            
    r  = dist(lats, lons, latmain, lonmain)
    ind = (r<=R)  
    lonc = lons[ind]
    latc = lats[ind]
    NSurf = len(lonc)
    lati = np.zeros(NSurf*Nz)
    loni = np.zeros(NSurf*Nz)
    zi = np.zeros(NSurf*Nz)
    m = 0
    for i in range(NSurf):
        for j in range(Nz):
            lati[m] = latc[i]
            loni[m] = lonc[i]
            zi[m] = gridz[j]
            m += 1   
    return gridlat, gridlon, gridz, lati, loni, zi
    
def plot_results(gridlon, gridlat, CFS, VM, MS, VMS):
    plt.contourf(gridlon, gridlat, np.log(CFS), levels=30)
    plt.title('CFS')
    plt.show()
    plt.contourf(gridlon, gridlat, np.log(VM), levels=30)
    plt.title('VM')
    plt.show()
    plt.contourf(gridlon, gridlat, np.log(MS), levels=30)
    plt.title('MS')
    plt.show()
    plt.contourf(gridlon, gridlat, np.log(VMS), levels=30)
    plt.title('VMS')
    plt.show()

def check_dipvalues():
    Ntot = 0
    names = ['SuperstitionHill', 'Landers', 'Northridge', 'HectorMine', 'BajaCalifornia', 'Ridgecrest']
    for nname in range(len(names)):
        name = names[nname]
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
        for ns in range(len(slipmodels)):        
            datname = 'SLIP-MODELS/%s.fsp' % (slipmodels[ns])
            lat_hypo, lon_hypo, z_hypo, yy, mm, dd, M, Nfault, strike, dip, rake, rake_all, L, W, lats, lons, zs, slip = read_fsp_file.read_fsp_file(datname)
            dip = np.asarray(dip)
            if len(dip[(dip>90)]) > 0:
                Ntot += len(dip[(dip>90)])
                print('%s  %s  %d patches with dip>90' % (name, slipmodels[ns], len(dip[(dip>90)])))
            if len(dip[(dip<0)]) > 0:
                Ntot += len(dip[(dip<0)])
                print('%s  %s  %d patches with dip<0' % (name, slipmodels[ns], len(dip[(dip<0)])))
    if Ntot == 0:
        print('\n\t All patches of all slip models have 0 <= dip <= 90\n')
    sys.exit()

###############################################################################
#check_dipvalues()      
names = ['SuperstitionHill', 'Landers', 'Northridge', 'HectorMine', 'BajaCalifornia', 'Ridgecrest']
dr = 1.0   # [km] spatial grid spacing in x, y, z direction

for nname in range(len(names)):
# for nname in range(5, 6):
    name = names[nname]

    # Observation points:
    gridlat, gridlon, gridz, lati, loni, zi = define_grid_circle(Seqdir, dr, name)
    Ntot = len(lati)
    Nlat = len(gridlat)
    Nlon = len(gridlon)
    Nz = len(gridz)
        
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

    for ns in range(len(slipmodels)):
        
        print('\n%s:\n' % (name))
        print('%s:\n' % (slipmodels[ns]))
        
        datname = '%s/%s.fsp' % (Slipdir, slipmodels[ns])
        lat_hypo, lon_hypo, z_hypo, yy, mm, dd, M, Nfault, strike, dip, rake, rake_all, L, W, lats, lons, zs, slip = read_fsp_file.read_fsp_file(datname)
        
        ux = 0.0
        uy = 0.0
        uz = 0.0
        sxx = 0.0
        syy = 0.0
        szz = 0.0
        sxy = 0.0
        syz = 0.0
        szx = 0.0
        for i in range(len(lats)):
            sys.stdout.write('\r'+str('%s: slippatch=%d/%d: lat=%f lon=%f  z=%f\r' % (name, i, len(lats), lats[i], lons[i], zs[i]))); sys.stdout.flush()
            uxi, uyi, uzi, sxxi, syyi, szzi, sxyi, syzi, szxi = stress_calculation.stresstensor_pscokada(alpha, mu, lmda, lats[i], lons[i], zs[i], lati, loni, zi, strike[i], dip[i], rake[i], L[i], W[i], slip[i])
            ux += uxi 
            uy += uyi
            uz += uzi
            sxx += sxxi
            syy += syyi
            szz += szzi
            sxy += sxyi
            syz += syzi
            szx += szxi

        CFSi = np.zeros(Ntot)
        VMi = np.zeros(Ntot)
        MSi = np.zeros(Ntot)
        VMSi = np.zeros(Ntot)
        for i in range(Ntot):
            CFSi[i], VMi[i], MSi[i], VMSi[i] = stressmetrics.stressmetrics(np.mean(strike), np.mean(dip), np.mean(rake), sxx[i], syy[i], szz[i], sxy[i], syz[i], szx[i], f, skempton)


        # CFSi = np.zeros(Nlat*Nlon*Nz)
        # VMi = np.zeros(Nlat*Nlon*Nz)
        # MSi = np.zeros(Nlat*Nlon*Nz)
        # VMSi = np.zeros(Nlat*Nlon*Nz)
        # CFS = np.zeros((Nlat, Nlon))
        # VM = np.zeros((Nlat, Nlon))
        # MS = np.zeros((Nlat, Nlon))
        # VMS = np.zeros((Nlat, Nlon))
        # n = 0
        # for i in range(Nlat):
        #     for j in range(Nlon):
        #         for k in range(Nz):
        #             CFSi[n], VMi[n], MSi[n], VMSi[n] = stressmetrics.stressmetrics(np.mean(strike), np.mean(dip), np.mean(rake), sxx[n], syy[n], szz[n], sxy[n], syz[n], szx[n], f, skempton)
        #             CFS[i,j] += CFSi[n] / (1.0*Nz)
        #             VM[i,j] += VMi[n] / (1.0*Nz)
        #             MS[i,j] += MSi[n] / (1.0*Nz)
        #             VMS[i,j] += VMSi[n] / (1.0*Nz)
        #             n += 1

        #plot_results(gridlon, gridlat, CFS, VM, MS, VMS)         
        
        outname = '%s/stressmetrics-%s-dr%.1fkm.out' % (Stressdir, slipmodels[ns], dr)
        np.savetxt(outname, np.column_stack((lati, loni, zi, CFSi, VMi, MSi, VMSi)), fmt='%f\t%f\t%f\t%e\t%e\t%e\t%e', header='lat\tlon\tz\tCFS\tVM\tMS\tVMS')
        print('\n\t OUTPUT: %s\n' % (outname))







    
