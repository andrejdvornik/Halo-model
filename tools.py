#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  tools.py
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
import scipy
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp
import gauleg



"""
# Mathematical tools.
"""

def Integrate(func_in, x_array): # Simpson integration on fixed spaced data!
	
	func_in = np.nan_to_num(func_in)
	result = simps(func_in, x_array)
    #result = trapz(func_in, x_array)
	
	#print ("Integrating over given function.")
	return result


def Integrate1(func_in, x_array): # Gauss - Legendre quadrature!
	
	import scipy.integrate as intg
	
	func_in = np.nan_to_num(func_in)
	
	c = interp1d(np.log(x_array), np.log(func_in), kind='slinear', bounds_error=False, fill_value=0.0)
	
	integ = lambda x: np.exp(c(np.log(x)))
	#result = intg.fixed_quad(integ, x_array[0], x_array[-1], n=2000)
	result = intg.quad(integ, x_array[0], x_array[-1])
	
	#print ("Integrating over given function.")
	return result[0]


def int_gauleg(func_in, a, b, n):
	
	"""
		Instead of this, use scipy.fixed_quad! Same stuff, but better implemented.
	"""
	
	#a = x_array[0]
	#b = x_array[-1]
	#n = len(x_array)
	
	#func_in = np.nan_to_num(func_in)
	#func_in = interp1d(np.log(x_array), np.log(func_in), kind='slinear', bounds_error=False, fill_value=0.0)
	
	w, x = gauleg.gaulegf(a,b,n)
	result = 0.0
	for i in range(1, n+1):
		result += w[i] * np.nan_to_num(func_in(x[i]))
	
	return result


def extrap1d(x, y, step_size, method):
	
	y_out = np.ones(len(x))
	
	
	# Step for stars at n = 76 = 0.006, for DM 0.003 and for gas same as for stars
	#~ step = len(x)*0.005#len(x)*0.0035#0.005!
	step = len(x)*step_size
	
	xi = np.log(x)
	yi = np.log(y)
	
	xs = xi
	ys = yi
	
	minarg = np.argmin(np.nan_to_num(np.gradient(yi)))
	#~ step = minarg - np.where(xi == min(xi, key=lambda x:abs(x - (xi[minarg] - 0.9) )))[0]
	
	if step < 3:
		step = 3

	#~ print np.exp(xi[minarg])
	#~ print np.exp(xi[minarg] - 0.04)
	
	if method == 1:
		yslice = ys[(minarg-step):(minarg):1]
		xslice = np.array([xs[(minarg-step):(minarg):1], np.ones(step)]) 
			
		w = np.linalg.lstsq(xslice.T, yslice)[0]
			
	elif method == 2:
		yslice = ys[(minarg-step):(minarg):1]
		xslice = xs[(minarg-step):(minarg):1]
		
		w = np.polyfit(xslice, yslice, 2)
				
	elif method == 3:	
		#~ yslice = y[(minarg-20):(minarg):1] #60
		#~ xslice = x[(minarg-20):(minarg):1]
		yslice = y[(minarg-len(x)/500.0):(minarg):1] #60
		xslice = x[(minarg-len(x)/500.0):(minarg):1]
		
		from scipy.optimize import curve_fit
		def func(x, a, b, c):
			return a * (1.0+(x/b))**(-2.0)
				
		popt, pcov = curve_fit(func, xslice, yslice, p0=(y[0], x[minarg], 0.0))
    #print popt
	
	for i in range(len(x)):
		
		if i > minarg: #100! #65, 125
		#~ if i+start > minarg: #100! #65, 125
		#if i+50 > (np.where(grad < grad_val)[0][0]):
			
			"""
			# Both procedures work via same reasoning, second one actually fits the slope from all the points, so more precise!
			"""
			
			#~ y_out[i] = np.exp(ys[np.argmin(yi)-50] + (xi[i] - xi[np.argmin(yi)-50])*(ys[np.argmin(yi)-50] - ys[np.argmin(yi)-100])/(xs[np.argmin(yi)-50] - xs[np.argmin(yi)-100])) #ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])     
			
			if method == 1:
				y_out[i] = np.exp(ys[minarg] + (xi[i] - xi[minarg])*(w[0]))
			
			elif method == 2:
				y_out[i] = np.exp(w[2] + (xi[i])*(w[1]) + ((xi[i])**2.0)*(w[0]))
				
			elif method == 3:	
				y_out[i] = (func(x[i], popt[0], popt[1], popt[2]))
				
		else:
			y_out[i] = np.exp(yi[i])
			
	return y_out
	

def extrap2d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike
    

#~ def fill_nan(a):
	#~ 
	#~ # Replaces nan with closest value, so we are not left with 0 or 10^308-something!
	#~ 
	#~ ind = np.where(~np.isnan(a))[0]
	#~ first, last = ind[0], ind[-1]
	#~ a[:first] = a[first]
	#~ a[last + 1:] = a[last]
	#~ 
	#~ return a
	

def fill_nan(a):
	
	not_nan = np.logical_not(np.isnan(a))
	indices = np.arange(len(a))
	
	if a[not_nan].size == 0 or indices[not_nan].size == 0:
		return a
	else:
		b = np.interp(indices, indices[not_nan], a[not_nan])
	
		return b
	
	
def gas_concentration(mass, x_1, power):
	r_c = 0.05 * (mass/(x_1))**(power)
	return r_c
	

def star_concentration(mass, x_1, power):
	r_t = 0.02 * (mass/(x_1))**(power)
	return r_t
	

if __name__ == '__main__':
	main()
	
	


