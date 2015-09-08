#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  halo.py
#
#  Copyright 2014 Andrej Dvornik <dvornik@dommel.strw.leidenuniv.nl>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

# Halo profile code
# Andrej Dvornik, 2014/2015

import time
import multiprocessing as multi
from progressbar import *
import numpy as np
import mpmath as mp
import longdouble_utils as ld
import matplotlib.pyplot as pl
import scipy
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp
import sys
sys.path.insert(0, '/home/dvornik/MajorProject/pylib/lib/python2.7/site-packages/')
import hmf.tools as ht
from hmf import MassFunction

import baryons
from tools import Integrate, Integrate1, int_gauleg, extrap1d, extrap2d, fill_nan, gas_concentration, star_concentration
from lens import power_to_corr, power_to_corr_multi, sigma, d_sigma
from dark_matter import NFW, NFW_Dc, NFW_f, Con, DM_mm_spectrum, GM_cen_spectrum, GM_sat_spectrum, delta_NFW, GM_cen_analy, GM_sat_analy
from cmf import *


"""
#-------- Declaring functions ----------
"""
	
	
"""
# --------------- Actual halo functions and stuff ------------------
"""


"""
# Mass function from HMFcalc.
"""

def memoize(function):
    memo = {}
    def wrapper(*args, **kwargs):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args, **kwargs)
            memo[args] = rv
        return rv
    return wrapper

@memoize
def Mass_Function(M_min, M_max, step, k_min, k_max, k_step, name, **cosmology_params):
	
	m = MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=step, mf_fit=name, delta_h=200.0, delta_wrt='mean', cut_fit=False, z2=None, nz=None, delta_c=1.686, **cosmology_params)
	
	return m


"""
# Components of density profile from Mohammed and Seljak 2014
"""

def T_n(n, rho_mean, z, M, R, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	
	np.seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')
    
	"""
	Takes some global variables! Be carefull if you remove or split some stuff to different container! 
	"""
	n = np.float64(n)
	
	if len(M.shape) == 0:
		T = np.ones(1)
		M = np.array([M])
		R = np.array([R])
	else:
		T = np.ones(len(M), dtype=np.longdouble)
		
	if profile == 'dm':
		for i in range(len(M)):
		
			i_range = np.linspace(0,R[i],100)

			Ti = (mp.mpf(4.0 * np.pi)/(M[i] * (mp.factorial(2.0*n + 1.0)))) * mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * (NFW(rho_mean, Con(z, M[i]), i_range)), i_range))
			
			T[i] = ld.string2longdouble(str(Ti))
	
	elif profile == 'gas':
		for i in range(len(M)):
		
			i_range = np.linspace(0,R[i],100)
	
			Ti = (mp.mpf(4.0 * np.pi)/(M[i] * (mp.factorial(2.0*n + 1.0)))) * mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * (baryons.u_g(np.float64(i_range), slope, r_char[i], omegab, omegac, M[i])), i_range))
		
			T[i] = ld.string2longdouble(str(Ti))
		
	elif profile == 'stars':
		for i in range(len(M)):
		
			i_range = np.linspace(0,R[i],100)
	
			Ti = (mp.mpf(4.0 * np.pi)/(M[i] * (mp.factorial(2.0*n + 1.0)))) * mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * (baryons.u_s(np.float64(i_range), slope, r_char[i], h_mass, M[i], sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)), i_range))
			
			T[i] = ld.string2longdouble(str(Ti))

	return T
	

"""
# Integrals of mass functions with density profiles and population functions.
"""

def f_k(k_x):

	F = sp.erf(k_x/0.1) #0.05!
	
	return F

	
def multi_proc_T(a,b, que, n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	outdict = {}
	
	r = np.arange(a, b, 1)

	T = np.ones((len(r), len(m_x)), dtype=np.longdouble)
		
	for i in range(len(r)):
		T[i,:] = T_n(r[i], rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)
	
	# Write in dictionary, so the result can be read outside of function.
		
	outdict = np.column_stack((r, T))
	que.put(outdict)
	
	return


def T_table_multi(n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	
	n = (n+4)/2
	
	nprocs = multi.cpu_count() # Match the number of cores!
	q1 = multi.Queue()
	procs = []
	chunk = int(np.ceil(n/float(nprocs)))
	
    #widgets = ['Calculating T: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=nprocs).start()
	
	for j in range(nprocs):
			
		work = multi.Process(target=multi_proc_T, args=((j*chunk), ((j+1)*chunk), q1, n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2))
		procs.append(work)
		work.start()

	result = np.array([]).reshape(0, len(m_x)+1)
	
	for j in range(nprocs):
		result = np.vstack([result, np.array(q1.get())])
	
    #pbar.finish()
	result = result[np.argsort(result[:, 0])]

	return np.delete(result, 0, 1)


def T_table(n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	
	"""
	Calculates all the T integrals and saves them into a array, so that the calling of them is fast for all other purposes.
	"""
	
	n = n + 2
	
	T = np.ones((n/2, len(m_x)))
	
	widgets = ['Calculating T: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ', ETA()]
	pbar = ProgressBar(widgets=widgets, maxval=n/2).start()
	
	for i in range(0, n/2, 1):
		T[i,:] = T_n(i, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)
		pbar.update(i+1)
	
	pbar.finish()
		
	return T
	
	
def n_gal(z, mass_function, population, x, r_x): # Calculates average number of galaxies!

	integrand = mass_function.dndlnm*population/x
	
	n = Integrate(integrand, x)

	return n
	
	
def eff_mass(z, mass_func, population, m_x):
	
	integ1 = mass_func.dndlnm*population
	integ2 = mass_func.dndm*population
	
	mass = Integrate(integ1, m_x)/Integrate(integ2, m_x)
	
	return mass
	

	
"""	
# Some bias functions
"""

def Bias(hmf, r_x):
	# PS bias - analytic
		
	bias = 1+(hmf.nu-1)/(hmf.growth*hmf.delta_c)
	
    #print ("Bias OK.")
	return bias
	
	
def Bias_Tinker10(hmf, r_x):
	# Tinker 2010 bias - empirical
	
    nu = np.sqrt(hmf.nu)
    y = np.log10(hmf.delta_halo)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4

	# print y, A, a, B, b, C, c
    return 1 - A * nu ** a / (nu ** a + hmf.delta_c ** a) + B * nu ** b + C * nu ** c	
	
	
"""
# Two halo term for matter-galaxy specta! For matter-matter it is only P_lin!
"""

def TwoHalo(mass_func, norm, population, k_x, r_x, m_x): # This is ok!
	
	P2 = (np.exp(mass_func.power)/norm)*(Integrate((mass_func.dndlnm*population*Bias_Tinker10(mass_func,r_x)/m_x),m_x))
	
    #print ("Two halo term calculated.")
	
	return P2

def model(theta, R, h=1, Om=0.315, Ol=0.685):
    
    from itertools import count, izip
    
    
    # HMF set up parameters - fixed and not setable from config file.
    
    
    expansion = 100
    expansion_stars = 160
    
    n_bins = 10000
    
    M_min = 8.5
    M_max = 15.5
    step = (M_max-M_min)/100 # or n_bins
    
    k_min = -6.0 #ln!!! not log10!
    k_max = 9.0 #ln!!! not log10!
    k_step = (k_max-k_min)/n_bins
    
    k_range = np.arange(k_min, k_max, k_step)
    k_range_lin = np.exp(k_range)
    
    M_star_min = 8.5 # Halo mass bin range
    M_star_max = 15.5
    step_star = (M_star_max-M_star_min)/100
    
    mass_range = np.arange(M_min,M_max,step)
    mass_range_lin = 10.0 ** (mass_range)
    
    mass_star_log = np.arange(M_star_min,M_star_max,step_star, dtype=np.longdouble)
    mass_star = 10.0 ** (mass_star_log)
    
    
    # Setting parameters from config file
    
    
    omegab = 0.0455 # To be replaced by theta
    omegac = 0.226 # To be replaced by theta
    omegav = 0.726 # To be replaced by theta
    
    sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, alpha_star, beta_gas, r_t0, r_c0, z, M_bin_min1, M_bin_min2, M_bin_max1, M_bin_max2 = theta
    
    M_bin_min = [M_bin_min1, M_bin_min2] # Expanded according to number of bins!
    M_bin_max = [M_bin_max1, M_bin_max2] # Expanded according to number of bins!
    
    hod_mass = [(10.0 ** (np.arange(Mi, Mx, (Mx - Mi)/100, dtype=np.longdouble))) for Mi, Mx in izip(M_bin_min, M_bin_max)]
    
    r_t0 = r_t0*np.ones(100)
    r_c0 = r_c0*np.ones(100)
    
    H0 = 70.4 # To be replaced by theta
    rho_crit = 2.7763458 * (10.0**11.0) # in M_sun*h^2/Mpc^3 # To be called from nfw_utils!
    
    cosmology_params = {"sigma_8": 0.81, "H0": 70.4,"omegab": 0.0455, "omegac": 0.226, "omegav": 0.728, "n": 0.967, "lnk_min": k_min ,"lnk_max": k_max, "dlnk": k_step, "transfer_fit": "CAMB", "z":z} # Values to be read in from theta
    
    
    # ---------------------------------------
    
    
    hmf = Mass_Function(M_min, M_max, step, k_min, k_max, k_step, "Tinker10", **cosmology_params) # Tinker10 should also be read from theta!
    
    mass_func = hmf.dndlnm
    #power = hmf.power
    rho_mean_int = rho_crit*(hmf.omegac+hmf.omegab)
    rho_mean = Integrate(mass_func, mass_range_lin)
    
    
    
    radius_range_lin = ht.mass_to_radius(mass_range_lin, rho_mean)/((200)**(1.0/3.0))
    radius_range = np.log10(radius_range_lin)
    radius_range_3d = 10.0 ** np.arange(-3.0, 2.9, (2.9 - (-3.0))/(100))
    
    radius_range_3d_i = 10.0 ** np.arange(-2.9, 1.0, (1.0 - (-2.9))/(50))
    radius_range_2d_i = 10.0 ** np.arange(-2.5, 1.0, (1.0 - (-2.5))/(50)) # THIS IS R!
    
    
    # Calculating halo model
    
    
    ngal = np.array([n_gal(z, hmf, ngm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) , mass_range_lin, radius_range_lin) for i in izip(hod_mass)])
    rho_dm = baryons.rhoDM(hmf, mass_range_lin, omegab, omegac)
    rho_stars = np.array([baryons.rhoSTARS(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    rho_gas = np.array([baryons.rhoGAS(hmf, rho_crit, omegab, omegac, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])[:,0]
    F = np.array([baryons.rhoGAS(hmf, rho_crit, omegab, omegac, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])[:,1]
    
    norm2 = rho_mean_int/rho_mean
    
    effective_mass = np.array([eff_mass(z, hmf, ngm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2), mass_range_lin) for i in izip(hod_mass)])
    effective_mass_dm = np.array([eff_mass(z, hmf, ncm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2), mass_range_lin)*baryons.f_dm(omegab, omegac) for i in izip(hod_mass)])
    effective_mass2 = effective_mass*(omegab/(omegac+omegab))
    effective_mass_bar = np.array([effective_mass2*(baryons.f_stars(i[0], effective_mass2, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)) for i in izip(hod_mass)])
    
    T_dm = np.array([T_table_multi(expansion, rho_dm, z, mass_range_lin, radius_range_lin, i[0], "dm", omegab, omegac, 0, 0, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    T_stars = np.array([T_table_multi(expansion_stars, rho_mean, z, mass_range_lin, radius_range_lin, i[0], "stars", omegab, omegac, alpha_star, r_t0, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    T_gas = np.array([T_table_multi(expansion, rho_mean, z, mass_range_lin, radius_range_lin, i[0], "gas", omegab, omegac, beta_gas, r_c0, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    T_tot = np.array([T_dm[i][0:1:1,:] + T_stars[i][0:1:1,:] + T_gas[i][0:1:1,:] for i in range(len(M_bin_min))])
    
    F_k1 = f_k(k_range_lin)
    
    pop_c = np.array([ncm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    pop_s = np.array([nsm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    pop_g = np.array([ngm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    
    # Galaxy - dark matter spectra
    
    Pg_2h = np.array([TwoHalo(hmf, ngal_i, pop_g_i, k_range_lin, radius_range_lin, mass_range_lin) for ngal_i, pop_g_i in izip(ngal, pop_g)])
    
    Pg_c = np.array([F_k1 * GM_cen_spectrum(hmf, z, rho_dm, rho_mean, expansion, pop_c_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_tot_i) for pop_c_i, ngal_i, T_dm_i, T_tot_i in izip(pop_c, ngal, T_dm, T_tot)])
    Pg_s = np.array([F_k1 * GM_sat_spectrum(hmf, z, rho_dm, rho_mean, expansion, pop_s_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_tot_i) for pop_s_i, ngal_i, T_dm_i, T_tot_i in izip(pop_s, ngal, T_dm, T_tot)])
    
    # Galaxy - stars spectra
    
    Ps_c = np.array([F_k1 * baryons.GS_cen_spectrum(hmf, z, rho_stars_i, rho_mean, expansion_stars, pop_c_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_stars_i, T_tot_i) for rho_stars_i, pop_c_i, ngal_i, T_stars_i, T_tot_i in izip(rho_stars, pop_c, ngal, T_stars, T_tot)])
    Ps_s = np.array([F_k1 * baryons.GS_sat_spectrum(hmf, z, rho_stars_i, rho_mean, expansion, pop_s_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_stars_i, T_tot_i) for rho_stars_i, pop_s_i, ngal_i, T_dm_i, T_stars_i, T_tot_i in izip(rho_stars, pop_s, ngal, T_dm, T_stars, T_tot)])
    
    # Galaxy - gas spectra
    
    Pgas_c = np.array([F_k1 * baryons.GGas_cen_spectrum(hmf, z, F_i, rho_gas_i, rho_mean, expansion, pop_c_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_gas_i, T_tot_i) for F_i, rho_gas_i, pop_c_i, ngal_i, T_gas_i, T_tot_i in izip(F, rho_gas, pop_c, ngal, T_gas, T_tot)])
    Pgas_s = np.array([F_k1 * baryons.GGas_sat_spectrum(hmf, z, F_i, rho_gas_i, rho_mean, expansion, pop_s_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_gas_i, T_tot_i) for F_i, rho_gas_i, pop_s_i, ngal_i, T_dm_i, T_gas_i, T_tot_i in izip(F, rho_gas, pop_s, ngal, T_dm, T_gas, T_tot)])
    
    # Combined (all) by type
    
    Pg_k_dm = np.array([(ngal_i*rho_dm*(Pg_c_i + Pg_s_i + Pg_2h_i*rho_mean/rho_dm))/(rho_mean*ngal_i) for ngal_i, Pg_c_i, Pg_s_i, Pg_2h_i in izip(ngal, Pg_c, Pg_s, Pg_2h)])
    Pg_k_s = np.array([(ngal_i*rho_stars_i*(Ps_c_i + Ps_s_i))/(rho_mean*ngal_i) for ngal_i, rho_stars_i, Ps_c_i, Ps_s_i in izip(ngal, rho_stars, Ps_c, Ps_s)])
    Pg_k_g = np.array([(ngal_i*rho_gas_i*(Pgas_c_i + Pgas_s_i))/(rho_mean*ngal_i) for ngal_i, rho_gas_i, Pgas_c_i, Pgas_s_i in izip(ngal, rho_gas, Pgas_c, Pgas_s)])
    
    Pg_k = np.array([(ngal_i*rho_dm*(Pg_c_i + Pg_s_i + Pg_2h_i*rho_mean/rho_dm) + ngal_i*rho_stars_i*(Ps_c_i + Ps_s_i) + ngal_i*rho_gas_i*(Pgas_c_i + Pgas_s_i))/(rho_mean*ngal_i) for ngal_i, Pg_c_i, Pg_s_i, Pg_2h_i, rho_stars_i, Ps_c_i, Ps_s_i, rho_gas_i, Pgas_c_i, Pgas_s_i in izip(ngal, Pg_c, Pg_s, Pg_2h, rho_stars, Ps_c, Ps_s, rho_gas, Pgas_c, Pgas_s)])  # all components
    
    # Normalized sattelites and centrals for sigma and d_sigma
    
    Pg_c2 = np.array([(rho_dm/rho_mean)*Pg_c_i for Pg_c_i in izip(Pg_c)])
    Pg_s2 = np.array([(rho_dm/rho_mean)*Pg_s_i for Pg_s_i in izip(Pg_s)])
     
    Ps_c2 = np.array([(rho_stars_i/rho_mean)*Ps_c_i for rho_stars_i, Ps_c_i in izip(rho_stars, Ps_c)])
    Ps_s2 = np.array([(rho_stars_i/rho_mean)*Ps_s_i for rho_stars_i, Ps_s_i in izip(rho_stars, Ps_s)])
     
    Pgas_c2 =  np.array([(rho_gas_i/rho_mean)*Pgas_c_i for rho_gas_i, Pgas_c_i in izip(rho_gas, Pgas_c)])
    Pgas_s2 =  np.array([(rho_gas_i/rho_mean)*Pgas_s_i for rho_gas_i, Pgas_s_i in izip(rho_gas, Pgas_s)])
    
    lnPg_k = np.array([np.log(Pg_k_i) for Pg_k_i in izip(Pg_k)]) # Total
    
    P_inter2 = [scipy.interpolate.UnivariateSpline(k_range, np.nan_to_num(lnPg_k_i), s=0) for lnPg_k_i in izip(lnPg_k)]
    
    """
    # Correlation functions
    """
    
    xi2 = np.zeros((len(M_bin_min), len(radius_range_3d)))
    for i in range(len(M_bin_min)):
        xi2[i,:] = power_to_corr_multi(P_inter2[i], radius_range_3d)
        xi2[xi2 <= 0.0] = np.nan
        xi2[i,:] = fill_nan(xi2[i,:])
    
    """
    # Projected surface density
    """
    
    sur_den2 = np.array([np.nan_to_num(sigma(xi2_i, rho_mean, radius_range_3d, radius_range_3d_i)) for xi2_i in izip(xi2)])
    sur_den2[sur_den2 >= 10.0**16.0] = np.nan
    sur_den2[sur_den2 <= 0.0] = np.nan
    for i in range(len(M_bin_min)):
        sur_den2[i,:] = fill_nan(sur_den2[i,:])
    
    """
    # Excess surface density
    """
    
    d_sur_den2 = np.array([np.nan_to_num(d_sigma(sur_den2_i, radius_range_3d_i, radius_range_2d_i)) for sur_den2_i in izip(sur_den2)])
    
    #print d_sur_den2/10**12.0

    return [d_sur_den2/10**12.0, xi2, lnPg_k] # Add other outputs as needed. Total ESD should always be first!


	
if __name__ == '__main__':
	
    #------------ Reading sys arguments ------------
	
    #task = str(sys.argv[1]) # Reads first argument after the script name - calculate, read or None!
    #task = int(task)
    #task = "calculate"
    #task = "read"
	
    argdict = dict(task="calculate", expansion=100, expansion_stars=160, combination=2)
	

    #############################################################################
    # "Combination" is used for combining different power spectra together		#
    # to represent some combined parts! Check the code later for which parts! 	#
    # Options are 1, 2 or 3!													#
    #############################################################################
    
    
    #theta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 2.0/3.0, 0.03, 0.05, 0.0, 11.2, 11.5, 11.5, 11.8]# = theta
    R = 0
    #model(theta, R, 1, 0.315, 0.685)

	
    print ('Task selected:'), argdict['task']
	
    #------------ Setting up all the parameters: precision, steps, cosmology, ...
	
    # Redshift
    z = 0.0
	
    print ('Calculating halo model at z ='), z
	
    expansion = argdict['expansion'] # Last power should be with + (-1)**((n+2)/2) !!! # Power of k to which we do Taylor expansion of power spectra (sinx/x), as definded in Mohammed!
    expansion_stars = argdict['expansion_stars']
	
    n_bins = 10000
            
    M_min = 8.5
    M_max = 15.5
    step = (M_max-M_min)/100 # or n_bins
       
    k_min = -6.0 #ln!!! not log10!
    k_max = 9.0
    k_step = (k_max-k_min)/n_bins
    
    k_range = np.arange(k_min, k_max, k_step)
    k_range_lin = np.exp(k_range)
    
    M_star_min = 8.5 # Halo mass bin range
    M_star_max = 15.5
    step_star = (M_star_max-M_star_min)/100
    
    mass_range = np.arange(M_min,M_max,step)
    mass_range_lin = 10.0 ** (mass_range)
	
    mass_star_log = np.arange(M_star_min,M_star_max,step_star, dtype=np.longdouble)
    mass_star = 10.0 ** (mass_star_log)
	
    hod_mass = 10.0 ** (np.arange(11.5, 11.8, (11.8 - 11.5)/100, dtype=np.longdouble))
	
    # Cosmology parameters for HMFcalc.
    
    cosmology_params = {"sigma_8": 0.81, "H0": 70.4,"omegab": 0.0455, "omegac": 0.226, "omegav": 0.728, "n": 0.967, "lnk_min": k_min ,"lnk_max": k_max, "dlnk": k_step, "transfer_fit": "CAMB", "z":z}
	
    
    # Densities.
            
    H0 = 70.4
    rho_crit = 2.7763458 * (10.0**11.0) # in M_sun*h^2/Mpc^3
	
    
    hmf = Mass_Function(M_min, M_max, step, k_min, k_max, k_step, "Tinker10", **cosmology_params)

    mass_func = hmf.dndlnm
    power = hmf.power
    rho_mean_int = rho_crit*(hmf.omegac+hmf.omegab)
    rho_mean = Integrate(mass_func, mass_range_lin)
	
    print ("Mass function calculated.")
	
	
	
    # Global constants for baryons: sigma and m_0 for profiles, ... Fedeli fiducial.
	
    omegab = hmf.omegab
    omegac = hmf.omegac
    alpha = 1.0
    beta = 2.0/3.0
    b_d = 0.85 # Bias from Fedeli for diffuse gas component!
    r_t0 = 0.03*np.ones(100)# 0.03 - fiducial parameters, in general shoudl be fitted for
    r_c0 = 0.05*np.ones(100)# 0.05
	
    # Various ranges of quantities.

    radius_range_lin = ht.mass_to_radius(mass_range_lin, rho_mean)/((200)**(1.0/3.0))
    radius_range = np.log10(radius_range_lin)
    radius_range_3d = 10.0 ** np.arange(-3.0, 2.9, (2.9 - (-3.0))/(100))
	
    radius_range_3d_i = 10.0 ** np.arange(-2.9, 1.0, (1.0 - (-2.9))/(50))
    radius_range_2d_i = 10.0 ** np.arange(-2.5, 1.0, (1.0 - (-2.5))/(50))
    
    k_range = np.arange(k_min, k_max, k_step)
    k_range_lin = np.exp(k_range)
	
	
    #-------------------------------------
    
    print ('\nSanity checks. \n')
    print ('Min, max mass:'), mass_range[0], mass_range[-1]
    print ('Ranges:'), len(k_range), len(k_range_lin), len(mass_range_lin)
    print ('Min, max virial radii:'), radius_range_lin[0], radius_range_lin[-1]
    print ('Min, max k:'), k_range_lin[0], k_range_lin[-1]
    
            
    #---------- End parameters -----------
    
    
    #-------------------------------------
    
    
    #---------- Begin calculations -------
    
    nphi = phi_i(hmf, mass_star, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    ngal = n_gal(z, hmf, ngm(hmf, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0) , mass_range_lin, radius_range_lin) # That is ok!
	
    rho_dm = baryons.rhoDM(hmf, mass_range_lin, omegab, omegac)
    rho_stars = baryons.rhoSTARS(hmf, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    rho_gas, F = baryons.rhoGAS(hmf, rho_crit, omegab, omegac, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    norm2 = rho_mean_int/rho_mean
	
    print ('F:'), F
    print ('DM density:'), rho_dm
    print ('Stars density:'), rho_stars
    print ('Gas density:'), rho_gas
    print ('Sum of densities:'), rho_dm + rho_stars + rho_gas
    print ('Mean density int:'), rho_mean_int
    print ('Mean density theory:'), rho_mean
    print ('Critical density:'), rho_crit
    print ('Ratio of MDI/MDT:'), norm2
    print ('Ratio of MDS/MDT:'), (rho_dm + rho_stars + rho_gas)/rho_mean
	
    effective_mass = eff_mass(z, hmf, ngm(hmf, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0), mass_range_lin)
    effective_mass_dm = eff_mass(z, hmf, ncm(hmf, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0), mass_range_lin)*baryons.f_dm(omegab, omegac)
    effective_mass2 = effective_mass*(omegab/(omegac+omegab))
    effective_mass_bar = effective_mass2*(baryons.f_stars(mass_star, effective_mass2, 0, 0, 0, 0, 0, 0, 0, 0, 0))
	
    #~ r_t0 = star_concentration(mass_range_lin, 10**11.8, 3.0)
    #~ r_c0 = gas_concentration(mass_range_lin, 10**11.8, 3.0)
		
    print ngal
    print effective_mass_dm, np.log10(effective_mass_bar)
 	     
    concentration = Con(z, effective_mass_dm)
    NFW_d_sigma = delta_NFW(z, rho_dm, effective_mass_dm, radius_range_2d_i)
    print concentration
	
    # Calculating the Taylor expansion coefficients!
    
    if argdict['task'] == "calculate":
        print ('\nCalculating Taylor series coefficients.')

        T_dm = T_table_multi(expansion, rho_dm, z, mass_range_lin, radius_range_lin, hod_mass, "dm", omegab, omegac, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        T_stars = T_table_multi(expansion_stars, rho_mean, z, mass_range_lin, radius_range_lin, hod_mass, "stars", omegab, omegac, alpha, r_t0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        T_gas = T_table_multi(expansion, rho_mean, z, mass_range_lin, radius_range_lin, hod_mass, "gas", omegab, omegac, beta, r_c0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        T_tot = T_dm[0:1:1,:] + T_stars[0:1:1,:] + T_gas[0:1:1,:]
		
        #np.savetxt("output_data/11_50_11_80/T_dm_hydro.dat", T_dm, delimiter='\t')
        #np.savetxt("output_data/11_50_11_80/T_stars_hydro.dat", T_stars, delimiter='\t')
        #np.savetxt("output_data/11_50_11_80/T_gas_hydro.dat", T_gas, delimiter='\t')
		
		
    elif argdict['task'] == "read":
        print ('\nImporting precalculated Taylor series coefficients.')
		
        #T_dm = np.genfromtxt(open("output_data/11_50_11_80/T_dm_hydro.dat", "rb"), delimiter='\t', dtype=np.longdouble)
        #T_stars = np.genfromtxt(open("output_data/11_50_11_80/T_stars_hydro.dat", "rb"), delimiter='\t', dtype=np.longdouble)
        #T_gas = np.genfromtxt(open("output_data/11_50_11_80/T_gas_hydro.dat", "rb"), delimiter='\t', dtype=np.longdouble)
        #T_tot = T_dm[0:1:1,:] + T_stars[0:1:1,:] + T_gas[0:1:1,:]
		
		
    else:
        print ('\nTask shoudl be "calculate" or "read"! Ending.')
        sys.exit()

    #----------------------------------
	
    print ('Starting power spectra and lensing calculations. \n')
	
    F_k1 = f_k(k_range_lin)
	
    pop_c = ncm(hmf, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    pop_s = nsm(hmf, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    pop_g = ngm(hmf, hod_mass, mass_range_lin, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	
	# Matter-matter spectra -- Not the cross spectrum!
	
    P2_k = np.exp(hmf.power)
	
    P1_k = DM_mm_spectrum(hmf, z, rho_dm, rho_mean, expansion, k_range_lin, radius_range_lin, mass_range_lin, T_dm)*F_k1
    P_k = P1_k + P2_k
	
	
	# Galaxy - dark matter spectra
	
    Pg_2h = TwoHalo(hmf, ngal, pop_g, k_range_lin, radius_range_lin, mass_range_lin)
	
    Pg_c = F_k1 * GM_cen_spectrum(hmf, z, rho_dm, rho_mean, expansion, pop_c, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_dm, T_tot)
    Pg_s = F_k1 * GM_sat_spectrum(hmf, z, rho_dm, rho_mean, expansion, pop_s, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_dm, T_tot)
	
    Pg_c_a = F_k1 * GM_cen_analy(hmf, z, rho_dm, rho_mean, expansion, pop_c, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_dm, T_tot)
    Pg_s_a = F_k1 * GM_sat_analy(hmf, z, rho_dm, rho_mean, expansion, pop_s, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_dm, T_tot)
	
	
	# Galaxy - stars spectra
	
    Ps_c = F_k1 * baryons.GS_cen_spectrum(hmf, z, rho_stars, rho_mean, expansion_stars, pop_c, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_stars, T_tot)
    Ps_s = F_k1 * baryons.GS_sat_spectrum(hmf, z, rho_stars, rho_mean, expansion, pop_s, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_dm, T_stars, T_tot)
    Ps_2h = np.zeros(n_bins)#Pg_2h/rho_stars
	
	#~ Ps_c = F_k1 * baryons.GS_cen_analy(hmf, z, rho_stars, rho_mean, expansion, pop_c, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_stars, T_tot, r_t0)
    #Ps_c_a = (rho_stars/rho_mean_int)* F_k1 * baryons.GS_cen_analy(hmf, z, rho_stars, rho_mean, expansion, pop_c, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_stars, T_tot, r_t0)
	
	# Galaxy - gas spectra
	
    Pgas_c = F_k1 * baryons.GGas_cen_spectrum(hmf, z, F, rho_gas, rho_mean, expansion, pop_c, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_gas, T_tot)
    Pgas_s = F_k1 * baryons.GGas_sat_spectrum(hmf, z, F, rho_gas, rho_mean, expansion, pop_s, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_dm, T_gas, T_tot)
	
    Pgas_2h = np.zeros(n_bins)#Pg_2h/rho_gas
    Pgas_dif = b_d * Pg_2h#TwoHalo(hmf, ngal, pop_g, k_range_lin, radius_range_lin, mass_range_lin)
	
    P_gas = (1.0-F) * Pgas_dif + F * (Pgas_c + Pgas_s + Pgas_2h)
	
    Pgas_c_a = np.zeros(n_bins)#(rho_gas/rho_dm) * (F_k1 * baryons.Ggas_cen_analy(hmf, z, F, rho_gas, rho_mean, expansion, pop_c, ngal, k_range_lin, radius_range_lin, mass_range_lin, T_gas, T_tot, r_c0))
	
	# Combined (all) by type
	
    Pg_k_dm = (ngal*rho_dm*(Pg_c + Pg_s + Pg_2h*rho_mean/rho_dm))/(rho_mean*ngal) # galaxy-dm only
    Pg_k_s = (ngal*rho_stars*(Ps_c + Ps_s + Ps_2h))/(rho_mean*ngal)
    Pg_k_g = (ngal*rho_gas*(P_gas))/(rho_mean*ngal)
	
    Pg_k = (ngal*rho_dm*(Pg_c + Pg_s + Pg_2h*rho_mean/rho_dm) + ngal*rho_stars*(Ps_c + Ps_s + Ps_2h) + ngal*rho_gas*P_gas)/(rho_mean*ngal) # all components
	
    Pg_k_dm_a = (ngal*rho_dm*(Pg_c_a + Pg_s_a + Pg_2h*rho_mean/rho_dm))/(rho_mean*ngal)
	
	# Normalized sattelites and centrals for sigma and d_sigma
	
    Pg_c2 = (rho_dm/rho_mean)*Pg_c
    Pg_s2 = (rho_dm/rho_mean)*Pg_s
	
    Ps_c2 = (rho_stars/rho_mean)*Ps_c
    Ps_s2 = (rho_stars/rho_mean)*Ps_s
	
    Pgas_c2 = (rho_gas/rho_mean)*Pgas_c
    Pgas_s2 = (rho_gas/rho_mean)*Pgas_s
	
    # ----------------------------------------
	
    lnPg_k = np.log(Pg_k) # Total

    P_inter2 = scipy.interpolate.UnivariateSpline(k_range, np.nan_to_num(lnPg_k), s=0)
    
    """
    # Correlation functions
    """
    
    xi2 = power_to_corr_multi(P_inter2, radius_range_3d)
    xi2[xi2 <= 0.0] = np.nan
    xi2 = fill_nan(xi2)
	
    """
    # Projected surface density
    """
    
    sur_den2 = np.nan_to_num(sigma(xi2, rho_mean, radius_range_3d, radius_range_3d_i))
    sur_den2[sur_den2 >= 10.0**16.0] = np.nan
    sur_den2[sur_den2 <= 0.0] = np.nan
    sur_den2 = fill_nan(sur_den2)
    
    """
    # Excess surface density
    """
    
    d_sur_den2 = np.nan_to_num(d_sigma(sur_den2, radius_range_3d_i, radius_range_2d_i))
    
    print d_sur_den2/10**12.0
	
    """
    # Point mass excess surface density
    """
	
    d_sur_den_point = effective_mass_bar/(np.pi*(radius_range_2d_i**2.0))

