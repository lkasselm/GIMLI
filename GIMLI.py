import numpy as np
from scipy.integrate import quad
from random import uniform, gauss
import h5py
import warnings
warnings.filterwarnings("ignore")
import configparser

#############################################################################################################
########################################### FUNCTION DEFINITIONS ############################################
#############################################################################################################

def write_IC_hdf5(filename, data):
 def ifset(d,key):
  if key in d.keys():
   result=d[key]
  else:
   result=None
   if key in ['lt','fmt']:  #for plot
    result=''
  return result

 def ifset2(d,key,value):
  if value is None:
   result=ifset(d,key)
  else:
   result=value
  return result

 if isinstance(data, dict):
  data=[data]

 BoxSize = None
 NumPartType = 6
 MassTable = np.zeros(NumPartType, dtype = float)
 NumPart = np.zeros(NumPartType, dtype = int)
 
 for d in data:
  BoxSize = ifset2(d, 'BoxSize', BoxSize)
  i = d['PartType']
  MassTable[i] = d['PartMass']
  NumPart[i] = d['count']
  
 file = h5py.File(filename+'.hdf5','w')

 group = file.create_group("Header")
 if BoxSize is not None:
  group.attrs.create("BoxSize", BoxSize, shape=None, dtype=h5py.h5t.IEEE_F64LE)
 else:
  group.attrs.create("BoxSize", 0, shape=None, dtype=h5py.h5t.IEEE_F64LE)
 group.attrs.create("Flag_Cooling", 0, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("Flag_DoublePrecision", 0, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("Flag_Feedback", 0, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("Flag_IC_Info", 0, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("Flag_Metals", 0, shape=None, dtype=h5py.h5t.STD_I32LE) #in makegal ics is 1
 group.attrs.create("Flag_Sfr", 0, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("Flag_StellarAge", 0, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("HubbleParam", 1, shape=None, dtype=h5py.h5t.IEEE_F64LE)
 group.attrs.create("MassTable", MassTable, shape=None,
                                                     dtype=h5py.h5t.IEEE_F64LE)
 group.attrs.create("NumFilesPerSnapshot", 1, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("NumPart_ThisFile", NumPart, shape=None, dtype=h5py.h5t.STD_I32LE)
 group.attrs.create("NumPart_Total", NumPart, shape=None, dtype=h5py.h5t.STD_U32LE)
 group.attrs.create("NumPart_Total_HighWord", (0,0,0,0,0,0), shape=None, dtype=h5py.h5t.STD_U32LE)
 group.attrs.create("Omega0", 0, shape=None, dtype=h5py.h5t.IEEE_F64LE)
 group.attrs.create("OmegaLambda", 0, shape=None, dtype=h5py.h5t.IEEE_F64LE)
 group.attrs.create("Redshift", 0, shape=None, dtype=h5py.h5t.IEEE_F64LE)
 group.attrs.create("Time", 0, shape=None, dtype=h5py.h5t.IEEE_F64LE)

 ID_offset = 0
 for d in data:
  group = file.create_group("PartType"+str(d['PartType']))
  dataset = group.create_dataset("Coordinates", (d['count'],3), data=d['Coordinates'],
                                                            dtype=h5py.h5t.IEEE_F32LE)
  dataset = group.create_dataset("ParticleIDs", (d['count'],),
                   data=np.array(range(ID_offset,ID_offset+d['count'])), dtype=h5py.h5t.STD_I32BE)
  ID_offset += d['count']
  dataset = group.create_dataset("Velocities", (d['count'],3), data=d['Velocities'],
                                                            dtype=h5py.h5t.IEEE_F32LE)
  if d["PartType"] == 0:
   dataset = group.create_dataset("InternalEnergy", (d['count'],),
                   data=d["InternalEnergy"], dtype=h5py.h5t.IEEE_F32LE)
 file.close()

def find_closest(array, value):
    # find index of 'array' entry which is closest to 'value'
    diff = array - value
    idx = np.abs(diff).argmin()
    return idx, diff[idx]

def Burkert(r, r_s, rho_s):
    x = r/r_s
    return rho_s/((1 + x)*(1 + x**2))

def mass_Burkert(r):
    def dMdr(r):
        return 4 * np.pi * Burkert(r) * r**2
    return quad(dMdr, 0, r)[0]

def tNFW(r, r_s, rho_s, t):
    # t is the truncation radius in units of r_s
    x = r/r_s
    return (rho_s/(x * (1+x)**2)) * (t**2/(x**2 + t**2)) 

def dMdr_tNFW(r, r_s, rho_s, t):
    return 4 * np.pi * tNFW(r, r_s, rho_s, t) * r**2

def mass_tNFW(r, r_s, rho_s, t):
    return quad(dMdr_tNFW, 0, r, args=(r_s, rho_s, t))[0]

def NFW(r, r_s, rho_s):
    x = r/r_s
    return rho_s/(x * (1+x)**2)

def mass_NFW(r, r_s, rho_s):
    # return mass enclosed in r
    return 4 * np.pi * rho_s * r_s**3 * ( np.log( (r_s + r) /r_s ) - r/(r_s + r))

def Hernquist(r, r_s, rho_s):
    x = r/r_s
    return rho_s/(x * (1+x)**3)

def mass_Hernquist(r, r_s, rho_s):
    # return mass enclosed in r
    x = r/r_s
    return 2 * np.pi * rho_s * r_s**3 * x**2/(x+1)**2

def Beta_model(r, rho_0, r_c, beta):
    return rho_0 * (1 + (r/r_c)**2 )**(-3*beta/2)

def mass_Beta_model(r, rho_0, r_c, beta):
    return quad(lambda r_:4*np.pi*r_**2*Beta_model(r_, rho_0, r_c, beta), 0, r)[0]

def inverse_CPD_Hernquist(x, r_s, rho_s, r_max):
    # assumes r_max in kpc
    x *= (r_max/r_s)**2/( (r_max/r_s) +1)**2
    return - r_s * np.sqrt(x)/(np.sqrt(x)-1)
    
def mass_stars(r):
    return mass_Hernquist(r, r_s_stars, rho_s_stars)
def density_stars(r):
    return Hernquist(r, r_s_stars, rho_s_stars)

def mean_particle_separation_gas(r):
    N = mass_gas(r)/part_mass_gas # number of particles enclosed in r
    n = 3*N/(4 * np.pi * r**3) # average number density
    return n**(-1/3) # mean seperation

def mean_particle_separation_DM(r):
    N = mass_dm(r)/part_mass_DM # number of particles enclosed in r
    n = 3*N/(4 * np.pi * r**3) # average number density
    return n**(-1/3) # mean seperation

def mean_particle_separation_stars(r):
    N = mass_stars(r)/part_mass_stars
    n = 3 *N/(4 * np.pi * r**3)
    return n**(-1/3)
    
def dpdr_gas(r):
    return  43000 * density_gas(r) * total_mass(r)/r**2

def temp_gas(r):
    p = quad(dpdr_gas, r, np.inf)[0]
    u = (3/2) * p / density_gas(r)
    return u

def sigma_2_halo(r):
    p = quad(dpdr_halo, r, np.inf)[0]
    sigma_2 = p / density_dm(r)
    return sigma_2

def sigma_2_stars(r):
    def dpdr(r):
        return 43000 * density_stars(r) * total_mass(r)/r**2
    p = quad(dpdr, r, np.inf)[0]
    sigma_2 = p / density_stars(r)
    return sigma_2
        
# this is just for printing out the progress in a nice way
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, 
                      fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def main():
    #############################################################################################################
    ############################################## PARSE INI FILE ###############################################
    #############################################################################################################

    # initialize variables 
    have_gas = have_DM = have_stars = sample_DM = sample_gas = mirrored_halo = let_halo_contract = filename = N_bins = gas_data_path = N_part_gas =   gas_profile = beta = rho_s_gas = r_s_gas = N_part_DM = dm_profile = r_max_DM = truncation_radius_dm = r_max_gas = r_s_DM = rho_s_DM = None
    
    config = configparser.ConfigParser()
    config.read('settings.ini')
    
    # switches
    for key, value in config['SWITCHES'].items():
        print(key)
        globals()[key] = bool(value)
    
    # params
    for key, value in config['PARAMS'].items():
        print(key, value)
        if '.' in value:
            try:
                globals()[key] = float(value)
            except ValueError:
                globals()[key] = value # store as string
        else:
            try:
                globals()[key] = int(value)
            except ValueError:
                globals()[key] = value # store as string

    if gas_profile == 'Burkert':
        def mass_gas(r):
            return mass_Burkert(r)
        def density_gas(r):
            return Burkert(r)
    
    elif gas_profile == 'Hernquist':
        def mass_gas(r):    
            return mass_Hernquist(r, r_s_gas, rho_s_gas)
        def density_gas(r):
            return Hernquist(r, r_s_gas, rho_s_gas)
    
    elif gas_profile == 'Beta_model':
        def mass_gas(r):
            return mass_Beta_model(r, rho_s_gas, r_s_gas, beta)
        def density_gas(r):
            return Beta_model(r, rho_s_gas, r_s_gas, beta)

    if dm_profile == 'NFW':
        def mass_dm(r):
            return mass_NFW(r, r_s_DM, rho_s_DM)
        def density_dm(r):
            return NFW(r, r_s_DM, rho_s_DM)
    elif dm_profile == 'tNFW':
        def density_dm(r):
            return tNFW(r, r_s_DM, rho_s_DM, truncation_radius_dm)
        def mass_dm(r):
            return mass_tNFW(r, r_s_DM, rho_s_DM, truncation_radius_dm)

    def total_mass(r):
        return have_DM * mass_dm(r) + have_gas * mass_gas(r) + have_stars * mass_stars(r)

    if let_halo_contract:
        def dpdr_halo(r):
            return  43000 * density_dm(r) * mass_dm(r)/r**2
    else:
        def dpdr_halo(r):
            return 43000 * density_dm(r) * total_mass(r)/r**2
            
    ##############################################################################################################
    ########################################### CALCULATE INTEGRATION TABLES #####################################
    ##############################################################################################################
    print(read_gas_data)
    if not read_gas_data:
        r_max_gas = r_max_gas * r_s_gas
    else:
        coords_gas = np.load(gas_data_path) * r_s_gas
        c = np.transpose(coords_gas)
        rs = np.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
        r_max_gas = np.amax(rs)
    
    r_max_DM = r_max_DM * r_s_DM
    r_max_stars = r_max_stars * r_s_stars
    
    r_arr_gas = np.logspace(-9+np.log10(r_max_gas), np.log10(r_max_gas), N_sampling)
    r_arr_DM = np.logspace(-9+np.log10(r_max_DM), np.log10(r_max_DM), N_sampling)
    r_arr_stars = np.logspace(-9+np.log10(r_max_stars), np.log10(r_max_stars), N_sampling)
        
    if have_gas and sample_gas:
        M_arr_gas = [mass_gas(r) for r in r_arr_gas]
        CPD_arr_gas = M_arr_gas/np.amax(M_arr_gas)
        
        print('Calculating gas temperature profile... ')
    
        temp_arr = np.zeros(N_sampling)
    
        for i, r in enumerate(r_arr_gas):
            printProgressBar(i, len(r_arr_gas), length=50)
            temp_arr[i] = temp_gas(r)
                
    if have_stars:
        print('Calculating star velocity dispersion profile... ')
    
        sigma2_arr_stars = np.zeros(N_sampling)
    
        for i, r in enumerate(r_arr_stars):
            printProgressBar(i, len(r_arr_stars), length=50)
            sigma2_arr_stars[i] = sigma_2_stars(r)
    
    if have_DM and sample_DM:
        M_arr_DM = [mass_dm(r) for r in r_arr_DM]
        CPD_arr_DM = M_arr_DM/np.amax(M_arr_DM)
            
        print('Calculating DM velocity dispersion profile...')
    
        sigma2_arr_dm = np.zeros(N_sampling)
    
        for i, r in enumerate(r_arr_DM):
            printProgressBar(i, len(r_arr_DM), length=50)
            sigma2_arr_dm[i] = sigma_2_halo(r)
    
    #############################################################################################################
    ############################################# SAMPLE PARTICLES ##############################################
    #############################################################################################################
    
    if have_gas and sample_gas:
        
        print('')
        print('Sampling gas ...')
    
        if not read_gas_data:
            coords_gas = np.zeros([N_part_gas, 3])
        vels_gas = np.zeros([N_part_gas, 3])
        temps_gas = np.zeros(N_part_gas)
        
        if mirrored_halo and not read_gas_data:
            # N_part must be an even number:
            if N_part_gas % 2 != 0:
                N_part_gas += 1
            N_sample = int(N_part_gas/2)
        else:
            N_sample = N_part_gas
    
        for i in range(N_sample):
            if i%1000 == 0:
                printProgressBar(i, N_part_gas, length=50)
            
            if not read_gas_data:
                # sample radius
                if gas_profile == 'Hernquist':
                    x = uniform(0, 1)
                    r = inverse_CPD_Hernquist(x, r_s_gas, rho_s_gas, r_max_gas)
                else:
                    x = uniform(0, 1)
                    idx, diff = find_closest(CPD_arr_gas, x)
                    r = r_arr_gas[idx]
    
                # sample angular position 
                phi = uniform(0, 1) * 2 * np.pi
                x = uniform(0.0,1.0)-0.5
                theta = np.arccos(-2.0*x)
    
                # set coordinates
                coords_gas[i][0] = r*np.sin(theta)*np.cos(phi)
                coords_gas[i][1] = r*np.sin(theta)*np.sin(phi)
                coords_gas[i][2] = r*np.cos(theta)
            else:
                c = np.transpose(coords_gas[i])
                r = np.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
    
            # set temperature
            idx, diff = find_closest(r_arr_gas, r)
            temps_gas[i] = temp_arr[idx]
    
            if mirrored_halo and not read_gas_data:
                # find positions of mirror particles
                coords_gas[N_sample + i] = -coords_gas[i]
                temps_gas[N_sample +i] = temps_gas[i]
        
    if have_DM and sample_DM:
        print('')
        print('Sampling DM ...')
    
        if mirrored_halo:
            # N_part_DM must be an even number:
            if N_part_DM % 2 != 0:
                N_part_DM += 1
            N_sample = int(N_part_DM/2)
        else:
            N_sample = N_part_DM
            
        coords_DM = np.zeros([N_part_DM,3])
        vels_DM = np.zeros([N_part_DM,3])
    
        for i in range(N_sample):
            if i%1000 == 0:
                printProgressBar(i, N_sample, length=50)
    
            # sample radius 
            x = uniform(0, 1)
            idx, diff = find_closest(CPD_arr_DM, x)
            r = r_arr_DM[idx]
    
            # sample angular position 
            phi = uniform(0, 1) * 2 * np.pi
            x = uniform(0.0,1.0)-0.5
            theta = np.arccos(-2.0*x)
    
            # convert to carthesian coordinates
            coords_DM[i][0] = r*np.sin(theta)*np.cos(phi)
            coords_DM[i][1] = r*np.sin(theta)*np.sin(phi)
            coords_DM[i][2] = r*np.cos(theta)
    
            # find velocity dispersion by matching to v_disp array
            idx, diff = find_closest(r_arr_DM, r)
            sigma2 = sigma2_arr_dm[idx]
            sigma = np.sqrt(sigma2)
    
            vels_DM[i][0] = gauss(0, sigma)
            vels_DM[i][1] = gauss(0, sigma)
            vels_DM[i][2] = gauss(0, sigma)
            
            if mirrored_halo:
                # find positions of mirror particles
                coords_DM[N_sample + i] = -coords_DM[i]
                vels_DM[N_sample + i] = -vels_DM[i]
                
    if have_stars:
        print('')
        print('Sampling stars ...')
    
        if mirrored_halo:
            # N_part must be an even number:
            if N_part_stars % 2 != 0:
                N_part_stars += 1
            N_sample = int(N_part_stars/2)
        else:
            N_sample = N_part_stars
            
        coords_stars = np.zeros([N_part_stars,3])
        vels_stars = np.zeros([N_part_stars,3])
    
        for i in range(N_sample):
            if i%1000 == 0:
                printProgressBar(i, N_sample, length=50)
    
            # sample radius 
            x = uniform(0, 1)
            r = inverse_CPD_Hernquist(x, r_s_stars, rho_s_stars, r_max_stars)
    
            # sample angular position 
            phi = uniform(0, 1) * 2 * np.pi
            x = uniform(0.0,1.0)-0.5
            theta = np.arccos(-2.0*x)
    
            # find coordinates
            coords_stars[i][0] = r*np.sin(theta)*np.cos(phi)
            coords_stars[i][1] = r*np.sin(theta)*np.sin(phi)
            coords_stars[i][2] = r*np.cos(theta)
    
            # find square velocity dispersion
            idx, diff = find_closest(r_arr_DM, r)
            sigma2 = sigma2_arr_stars[idx]
            sigma = np.sqrt(sigma2)
    
            vels_stars[i][0] = gauss(0, sigma)
            vels_stars[i][1] = gauss(0, sigma)
            vels_stars[i][2] = gauss(0, sigma)
            
            if mirrored_halo:
                # find positions of mirror particles
                coords_stars[N_sample + i] = -coords_stars[i]
                vels_stars[N_sample + i] = -vels_stars[i]
    
    # get total masses:
    total_mass_gas = mass_gas(r_max_gas)
    total_mass_DM = mass_dm(r_max_DM)
    total_mass_stars = mass_stars(r_max_stars)
    
    #############################################################################################################
    ############################################# SAVE TO HDF5 FILE #############################################
    #############################################################################################################
    
    print(' ')
    print('Saving ...',end='')
    
    data = []
    
    if have_gas and sample_gas:
        part_mass_gas = total_mass_gas/N_part_gas
        data_gas = {}
        data_gas['count'] = N_part_gas
        data_gas['PartMass'] = part_mass_gas
        data_gas['PartType'] = 0
        data_gas['Coordinates'] = coords_gas
        data_gas['Velocities'] = vels_gas
        data_gas['InternalEnergy'] = temps_gas
        data.append(data_gas)
    
    if have_DM and sample_DM:
        part_mass_DM = total_mass_DM/N_part_DM
        data_DM = {}
        data_DM['count'] = N_part_DM
        data_DM['PartMass'] = part_mass_DM
        data_DM['PartType'] = 1
        data_DM['Coordinates'] = coords_DM
        data_DM['Velocities'] = vels_DM
        data.append(data_DM)
        
    if have_stars:
        part_mass_stars = total_mass_stars/N_part_stars
        data_stars = {}
        data_stars['count'] = N_part_stars
        data_stars['PartMass'] = part_mass_stars
        data_stars['PartType'] = 4
        data_stars['Coordinates'] = coords_stars
        data_stars['Velocities'] = vels_stars
        data.append(data_stars)
    
    write_IC_hdf5(filename, data)
    
    print('done!')
    print('Suggested softening lengths based on mean central interparticle spacing:')
    if have_gas and sample_gas:
        print('Gas: ', np.round(mean_particle_separation_gas(r_s_gas/5)*2, 3), ' kpc')
    if have_DM and sample_DM:
        print('DM: ', np.round(mean_particle_separation_DM(r_s_gas/5)*2, 3), ' kpc')
    if have_stars:
        print('Stars: ', np.round(mean_particle_separation_stars(r_s_gas/5)*2, 3), ' kpc')

if __name__ == '__main__':
    main()