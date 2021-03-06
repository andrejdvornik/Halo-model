import numpy as np
from tools import Integrate


"""
# Population functions - average number of galaxies (central or satelites or all) in a halo of mass M - from HOD or CLF!
"""


"""
# Zheng 2005
"""

#~ def ncm(mass_func, m, M):
    #~
    #~ M_min = 11.6222
    #~ M_1 = 12.851
    #~ alpha = 1.049
    #~ M_0 = 11.5047
    #~ sig_logm = 0.26
    #~
    #~ nc = 0.5 * (1 + sp.erf((np.log10(M) - M_min) / sig_logm))
    #~
    #~ return nc
    #~
    #~
#~ def nsm(mass_func, m, M):
    #~
    #~ M_min = 11.6222
    #~ M_1 = 12.851
    #~ alpha = 1.049
    #~ M_0 = 11.5047
    #~ sig_logm = 0.26
    #~
    #~ ns = np.zeros_like(M)
    #~ ns[M > 10 ** M_0] = ((M[M > 10 ** M_0] - 10 ** M_0) / 10 ** M_1) ** alpha
    #~
    #~ return ns
    #~
    #~
#~ def ngm(mass_func, m, M):
    #~
    #~ ng = ncm(mass_func, m, M) + nsm(mass_func, m, M)
    #~
    #~ return ng


"""
# Conditional mass function derived
"""

def phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2):
    # Conditional stellar mass function - centrals
    # m - stellar mass, M - halo mass

    # FROM OWLS HOD FIT: sigma = 4.192393813649759049e-01

    sigma = 0.125

    if not np.iterable(M):
        M = np.array([M])
        phi = np.zeros((1, len(m)))
    else:
        phi = np.zeros((len(M), len(m)))

    for i in range(len(M)):

        M_0 = m_0(M[i], A, M_1, gamma_1, gamma_2)
        x = m/M_0
        phi[i,:] = np.log10(np.e)*np.exp(-(np.log10(x)*np.log10(x))/(2.0*(sigma**2.0)))/(((2.0*np.pi)**0.5)*sigma*m)


    return phi


def phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
    # Conditional stellar mass function - satellites
    # m - stellar mass, M - halo mass

    alpha = -2.060096789583814925e+00

    if not np.iterable(M):
        M = np.array([M])
        phi = np.zeros((1, len(m)))
    else:
        phi = np.zeros((len(M), len(m)))

    for i in range(len(M)):

        M_0 = 0.5 * m_0(M[i], A, M_1, gamma_1, gamma_2)
        x = m/M_0
        y = -(x**2.0)
        phi[i,:] = phi_0(M[i], b_0, b_1, b_2) * ((x)**(alpha + 1.0)) * np.exp(y) / m

    return phi


def phi_t(m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
    # Sum of phi_c and phi_s
    # m - stellar mass, M - halo mass

    phi = phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2) + phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)

    return phi


def phi_i(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
    # Integrated phi_t!
    # m - stellar mass, M - halo mass
    
    phi = np.ones(len(m))
    
    phi_int = phi_t(m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2).T

    for i in range(len(m)):
        integ = phi_int[i,:]*mass_func.dndm
        phi[i] = Integrate(integ, M)

    return phi.T


def av_cen(m, M, sigma, A, M_1, gamma_1, gamma_2):

    if not np.iterable(M):
        M = np.array([M])

    phi = np.ones(len(M))

    phi_int = phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2) #Centrals!
    for i in range(len(M)):
        integ = phi_int[i,:]*m
        phi[i] = Integrate(integ, m)

    return phi


def av_sat(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):

    if not np.iterable(M):
        M = np.array([M])

    phi = np.ones(len(M))

    phi_int = phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) #Satelites!
    for i in range(len(M)):
        integ = phi_int[i,:]*m
        phi[i] = Integrate(integ, m)

    return phi


def m_0(M, A, M_1, gamma_1, gamma_2):
    # Stellar mass as a function of halo mass parametrisation!
    # Fit as a result in my thesis!

    A = 10.0**(9.125473164494394496e+00)
    M_1 = 10.0**(1.044861151548183820e+01)
    gamma_1 = 2.619106875140703838e+00
    gamma_2 = 8.256965648888969778e-01
    # Above values taken from Cacciato 2009, may not be ok!


    m_0 = A*(((M/M_1)**(gamma_1))/((1.0 + (M/M_1))**(gamma_1 - gamma_2)))

    return m_0


def phi_0(M, b_0, b_1, b_2):
    # Functional form for phi_0 - taken directly from Cacciato 2009
    # Fit as a result in my thesis!

    b_0 = -5.137787703823422092e-01
    b_1 = 7.478552629547742525e-02
    b_2 = -7.938925982752477462e-02
    M12 = M/(10.0**12.0)

    log_phi = b_0 + b_1*np.log10(M12) + b_2*(np.log10(M12))**2.0

    phi = 10.0**log_phi

    return phi


def ncm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
    
    nc = np.ones(len(M))
        
    phi_int = phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2)
        
    for i in range(len(M)):
        integ = phi_int[i,:]
        nc[i] = Integrate(integ, m)
    return nc


def nsm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
    
    ns = np.ones(len(M))
        
    phi_int = phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)
        
    for i in range(len(M)):
        integ = phi_int[i,:]
        ns[i] = Integrate(integ, m)
    return ns


def ngm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
    
    ng = ncm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) + nsm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)
        
    return ng

if __name__ == '__main__':
    main()
