#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  dark.py
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
import time
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp
from tools import Integrate, Integrate1, int_gauleg, extrap1d, extrap2d, fill_nan


"""
# NFW profile and corresponding parameters.
"""

def NFW(rho_mean, c, R_array):
    
    r_max = R_array[-1]
    d_c = NFW_Dc(200, c)
    r_s = NFW_RS(c, r_max)
    
    # When using baryons, rho_mean gets is rho_dm!
    
    profile = (rho_mean*d_c)/((R_array/r_s)*((1+R_array/r_s)**2))
    
    #print ("NFW.")
    #print ("%s %s %s", d_c, r_s, r_max)
    
    return profile


def NFW_RS(c, r_max):
    #print ("RS.")
    return r_max/c
    

def NFW_Dc(delta_h, c):
    #print ("DC.")
    return (delta_h*(c**3))/(3*(np.log(1+c)-(c/(1+c))))
    
    
# Fourier transform of NFW profile - analytic!

def NFW_f(z, rho_mean, m_x, r_x, k_x):
	
	if len(m_x.shape) == 0:
		m_x = np.array([m_x])
		r_x = np.array([r_x])
	
	u_k = np.zeros((len(k_x), len(m_x)))
	
	
	for i in range(len(m_x)):
		c = Con(z, m_x[i])
		r_s = NFW_RS(c, r_x[i])
		
		K = k_x*r_s
		
		bs, bc = sp.sici(K)
		asi, ac = sp.sici((1 + c) * K)
		
		u_k[:,i] = 4*np.pi*rho_mean*NFW_Dc(200, c)*(r_s**3) * (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) + np.cos(K) * (ac - bc))/m_x[i]
	
	return u_k
    

def Con(z, M):
	
	#duffy rho_crit
	#c = 6.71 * (M / (2.0 * 10 ** 12)) ** -0.091 * (1 + z) ** -0.44
	
	#duffy rho_mean
	c = 10.14 * (M / (2.0 * 10 ** 12)) ** -0.081 * (1.0 + z) ** -1.01
	
	#zehavi
	#c = ((M / 1.5e13) ** -0.13) * 9.0 / (1 + z)

	#bullock_rescaled
	#c = (M / 10 ** 12.47) ** (-0.13) * 11 / (1 + z)
	
	#c = c0 * (M/M0) ** b
	
	return c
	
	
def delta_NFW(z, rho_mean, M, r):
	
	import hmf.tools as ht
	
	c = Con(z, M)
	r_vir = ht.mass_to_radius(M, rho_mean)/((200.0)**(1.0/3.0))
	r_s = NFW_RS(c, r_vir)
	d_c = NFW_Dc(200.0, c)
   
	x = r/r_s
	
	g = np.ones(len(x))
	
	for i in range(len(x)):
		if x[i]<1.0:
			g[i] = (8.0*np.arctanh(np.sqrt((1.0 - x[i])/(1.0 + x[i]))))/((x[i]**2.0)*np.sqrt(1.0 - x[i]**2.0)) + (4.0*np.log(x[i]/2.0))/(x[i]**2.0) - 2.0/(x[i]**2.0 - 1.0) + (4.0*np.arctanh(np.sqrt((1.0 - x[i])/(1.0 + x[i]))))/((x[i]**2.0 - 1.0)*np.sqrt(1.0 - x[i]**2.0))
			
		elif x[i]==1.0:
			g[i] = 10.0/3.0 + 4.0*np.log(0.5)
			
		elif x[i]>=1.0:
			g[i] = (8.0*np.arctan(np.sqrt((x[i] - 1.0)/(1.0 + x[i]))))/((x[i]**2.0)*np.sqrt(x[i]**2.0 - 1.0)) + (4.0*np.log(x[i]/2.0))/(x[i]**2.0) - 2.0/(x[i]**2.0 - 1.0) + (4.0*np.arctan(np.sqrt((x[i] - 1.0)/(1.0 + x[i]))))/((x[i]**2.0 - 1.0)**(3.0/2.0))
		
	
	return r_s * d_c * rho_mean * g
	
	
def av_delta_NFW(mass_func, z, rho_mean, hod, M, r):
	
	integ = np.ones((len(M), len(r)))
	average = np.ones(len(r))
	prob = hod*mass_func
	
	for i in range(len(M)):

		integ[i,:] = delta_NFW(z, rho_mean, M[i], r)
	
	for i in range(len(r)):
		
		average[i] = Integrate(prob*integ[:,i], M)
		
	av = average/Integrate(prob, M) # Needs to be normalized!
	
	return av
	

"""
# Spectrum components for dark matter.
"""
	
def GM_cen_analy(mass_func, z, rho_dm, rho_mean, n, population, ngal, k_x, r_x, m_x, T, T_tot):
	
	spec = np.ones(len(k_x))
	integ = np.ones((len(k_x), len(m_x)))
	
	from baryons import f_dm
	
	u_k = NFW_f(z, rho_mean*f_dm(0.0455, 0.226), m_x, r_x, k_x)
	
	for i in range(len(k_x)):
		integ[i,:] = mass_func.dndlnm * population * u_k[i,:]
		spec[i] = Integrate(integ[i,:], m_x)
	
	spec = spec/(rho_dm*ngal)
	return spec
	
	
def GM_sat_analy(mass_func, z, rho_dm, rho_mean, n, population, ngal, k_x, r_x, m_x, T, T_tot):
	
	spec = np.ones(len(k_x))
	integ = np.ones((len(k_x), len(m_x)))
	
	from baryons import f_dm
	
	u_k = NFW_f(z, rho_mean*f_dm(0.0455, 0.226), m_x, r_x, k_x)
	
	for i in range(len(k_x)):
		integ[i,:] = mass_func.dndlnm * population * u_k[i,:]**2.0
		spec[i] = Integrate(integ[i,:], m_x)
	
	spec = spec/(rho_dm*ngal)
	
	return spec
	
	
def DM_mm_spectrum(mass_func, z, rho_dm, rho_mean, n, k_x, r_x, m_x, T):
	
	"""
	Calculates the power spectrum for the component given in the name. Following the construction from Mohammed, but to general power of k!
	In practice the contributions from k > 50 are so small it is not worth doing it.
	Extrapolates the power spectrum to get rid of the knee, which is a Taylor series artifact.
	"""
	
	n = n + 2
	
	k_x = np.longdouble(k_x)
	
	#T = np.ones((n/2, len(m_x)))
	spec = np.ones(len(k_x))
	integ = np.ones((n/2, len(m_x)))
	T_comb = np.ones((n/2, len(m_x)))
	comp = np.ones((n/2, len(k_x)), dtype=np.longdouble)
	
	# Calculating all the needed T's! 
	"""
	for i in range(0, n/2, 1):
		T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
	"""		
	norm = 1.0/((T[0,:])**2.0)
	
	for k in range(0, n/2, 1):
		T_combined = np.ones((k+1, len(m_x)))
		
		for j in range(0, k+1, 1):

			T_combined[j,:] = T[j,:] * T[k-j,:]
			 
		T_comb[k,:] = np.sum(T_combined, axis=0)
		
		#print T_comb[k,:]
		
		integ[k,:] = norm*(m_x*mass_func.dndlnm*T_comb[k,:])/((rho_dm**2.0))
		comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2.0)) * (-1.0)**(k)
	
	spec = np.sum(comp, axis=0)
	spec[spec >= 10.0**10.0] = np.nan
	spec[spec <= 0.0] = np.nan

	spec_ext = extrap1d(np.float64(k_x), np.float64(spec), 0.001, 2)
	
	return spec_ext
	

def GM_cen_spectrum(mass_func, z, rho_dm, rho_mean, n, population, ngal, k_x, r_x, m_x, T, T_tot):
	
	"""
	Calculates the power spectrum for the component given in the name. Following the construction from Mohammed, but to general power of k!
	In practice the contributions from k > 50 are so small it is not worth doing it.
	Extrapolates the power spectrum to get rid of the knee, which is a Taylor series artifact.
	"""
	
	n = n + 2
	
	k_x = np.longdouble(k_x)
	
	#T = np.ones((n/2, len(m_x)))
	spec = np.ones(len(k_x))
	integ = np.ones((n/2, len(m_x)))
	T_comb = np.ones((n/2, len(m_x)))
	comp = np.ones((n/2, len(k_x)), dtype=np.longdouble)
	
	# Calculating all the needed T's! 
	"""
	for i in range(0, n/2, 1):
		T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
	"""		
	norm = 1.0/(T_tot[0,:])
	
	for k in range(0, n/2, 1):
		
		T_comb[k,:] = T[k,:]
		
		integ[k,:] = norm*(population*mass_func.dndlnm*T_comb[k,:])/(rho_dm*ngal)
		comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2.0)) * (-1.0)**(k)
	
	spec = np.sum(comp, axis=0)
	spec[spec >= 10.0**10.0] = np.nan
	spec[spec <= 0.0] = np.nan

	spec_ext = extrap1d(np.float64(k_x), np.float64(spec), 0.001, 3)
	
	
	return spec_ext
	
	
def GM_sat_spectrum(mass_func, z, rho_dm, rho_mean, n, population, ngal, k_x, r_x, m_x, T, T_tot):
	
    """
    Calculates the power spectrum for the component given in the name. Following the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small it is not worth doing it.
    Extrapolates the power spectrum to get rid of the knee, which is a Taylor series artifact.
    """
	
    n = n + 2
	
    k_x = np.longdouble(k_x)
	
    #T = np.ones((n/2, len(m_x)))
    spec = np.ones(len(k_x))
    integ = np.ones((n/2, len(m_x)))
    T_comb = np.ones((n/2, len(m_x)))
    comp = np.ones((n/2, len(k_x)), dtype=np.longdouble)
	
    # Calculating all the needed T's!
    """
    for i in range(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/((T_tot[0,:])**2.0)
	
    for k in range(0, n/2, 1):
        T_combined = np.ones((k+1, len(m_x)))
		
        for j in range(0, k+1, 1):

            T_combined[j,:] = T[j,:] * T[k-j,:]
        
        T_comb[k,:] = np.sum(T_combined, axis=0)
		
        integ[k,:] = norm*(population*mass_func.dndlnm*T_comb[k,:])/(rho_dm*ngal)
        comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2.0)) * (-1.0)**(k)
	
    spec = np.sum(comp, axis=0)
    spec[spec >= 10.0**10.0] = np.nan
    spec[spec <= 0.0] = np.nan
	
    spec_ext = extrap1d(np.float64(k_x), np.float64(spec), 0.001, 2)
	
	
    return spec_ext

if __name__ == '__main__':
	main()

