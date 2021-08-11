import numpy as np
import numba
from numba import jit

@jit(nopython=True)
def coulomb_phi_multipole(xq, q, p, Q):
    
    N = len(xq)
    T2 = np.zeros((3,3))
    eps = 1e-15
    phi = np.zeros(N)
    
    for i in range(N):
        
        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(Rnorm>1e-12)[0]:
            
            T0 = 1/Rnorm[j]
            T1 = Ri[j,:]/Rnorm[j]**3
            T2[:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5
            
            phi[i] += q[j]*T0 + np.sum(T1[:]*p[j,:]) + 0.5*np.sum(np.sum(T2[:,:]*Q[j,:,:], axis = 1), axis = 0)
            
    return phi

@jit(nopython=True)
def coulomb_dphi_multipole(xq, q, p, Q, alpha, thole, polar_group, flag_polar_group):
    
    N = len(xq)
    T1 = np.zeros((3))
    T2 = np.zeros((3,3))
    eps = 1e-15
    
    scale3 = 1.0
    scale5 = 1.0
    scale7 = 1.0
    
    dphi = np.zeros((N,3))
    
    for i in range(N):
        
        aux = np.zeros((3))
        
        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(Rnorm>1e-12)[0]:
            
            R3 = Rnorm[j]**3
            R5 = Rnorm[j]**5
            R7 = Rnorm[j]**7
            
            if flag_polar_group==False:
                
                not_same_polar_group = True
                
            else:
                
                gamma = min(thole[i], thole[j])
                damp = (alpha[i]*alpha[j])**0.16666667
                damp += 1e-12
                damp = -1*gamma * (R3/(damp*damp*damp))
                expdamp = np.exp(damp)
                
                scale3 = 1 - expdamp
                scale5 = 1 - expdamp*(1-damp)
                scale7 = 1 - expdamp*(1-damp+0.6*damp*damp)
                
                if polar_group[i]!=polar_group[j]:
                    
                    not_same_polar_group = True
                    
                else:
                    
                    not_same_polar_group = False
                    
            if not_same_polar_group==True:
                
                for k in range(3):
                    
                    T0 = -Ri[j,k]/R3 * scale3
                    
                    for l in range(3):
                        
                        dkl = (k==l)*1.0
                        
                        T1[l] = dkl/R3 * scale3 - 3*Ri[j,k]*Ri[j,l]/R5 * scale5
                        
                        for m in range(3):
                            
                            dkm = (k==m)*1.0
                            T2[l][m] = (dkm*Ri[j,l]+dkl*Ri[j,m])/R5 * scale5 - 5*Ri[j,l]*Ri[j,m]*Ri[j,k]/R7 * scale7
         
                    
                    aux[k] += T0*q[j] + np.sum(T1*p[j]) + 0.5*np.sum(np.sum(T2[:,:]*Q[j,:,:], axis = 1), axis = 0)
                
        dphi[i,:] += aux[:]
        
    return dphi

@jit(nopython=True)
def coulomb_ddphi_multipole(xq, q, p, Q):
    
    T1 = np.zeros((3))
    T2 = np.zeros((3,3))
    
    eps = 1e-15
    
    N = len(xq)
    
    ddphi = np.zeros((N,3,3))
    
    for i in range(N):
        
        aux = np.zeros((3,3))
        
        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(Rnorm>1e-12)[0]:
            
            R3 = Rnorm[j]**3
            R5 = Rnorm[j]**5
            R7 = Rnorm[j]**7
            R9 = R3**3
            
            for k in range(3):
                
                for l in range(3):
                    
                    dkl = (k==l)*1.0
                    T0 = -dkl/R3 + 3*Ri[j,k]*Ri[j,l]/R5
                    
                    for m in range(3):
                        
                        dkm = (k==m)*1.0
                        dlm = (l==m)*1.0
                        
                        T1[m] = -3*(dkm*Ri[j,l]+dkl*Ri[j,m]+dlm*Ri[j,k])/R5 + 15*Ri[j,l]*Ri[j,m]*Ri[j,k]/R7
                        
                        for n in range(3):
                            
                            dkn = (k==n)*1.0
                            dln = (l==n)*1.0
                            
                            T2[m][n] = 35*Ri[j,k]*Ri[j,l]*Ri[j,m]*Ri[j,n]/R9 - 5*(Ri[j,m]*Ri[j,n]*dkl \
                                                                          + Ri[j,l]*Ri[j,n]*dkm \
                                                                          + Ri[j,m]*Ri[j,l]*dkn \
                                                                          + Ri[j,k]*Ri[j,n]*dlm \
                                                                          + Ri[j,m]*Ri[j,k]*dln)/R7 + (dkm*dln + dlm*dkn)/R5
                            
                    aux[k][l] += T0*q[j] + np.sum(T1[:]*p[j,:]) +  0.5*np.sum(np.sum(T2[:,:]*Q[j,:,:], axis = 1), axis = 0)
                    
        ddphi[i,:,:] += aux[:,:]
        
    return ddphi

@jit(nopython=True)
def coulomb_phi_multipole_Thole(xq, p, alpha, thole, polar_group, connections_12, pointer_connections_12, \
                                connections_13, pointer_connections_13, p12scale, p13scale):
    
    eps = 1e-15
    T1 = np.zeros((3))
    
    N = len(xq)
    
    phi = np.zeros((N))
    
    for i in range(N):
        
        aux = 0.
        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]
        
        Ri = xq[i] - xq
        
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(r<1e12)[0]:
            
            pscale = 1.0
            
            for ii in range(start_12, stop_12):
                
                if connections_12[ii]==j:
                    
                    pscale = p12scale
                    
            for ii in range(start_13, stop_13):
                
                if connections_13[ii]==j:
                    
                    pscale = p13scale
                    
            r3 = r[j]**3
            
            gamma = min(thole[i], thole[j])
            damp = (alpha[i]*alpha[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)
            
            scale3 = 1 - expdamp
            
            for k in range(3):
                
                T1[k] = Ri[j,k]*r3*scale3*pscale
                
            aux += np.sum(T1[:]*p[j,:])
            
        phi[i] += aux
    
    return phi

@jit(nopython=True)
def coulomb_dphi_multipole_Thole(xq, p, alpha, thole, polar_group, connections_12, pointer_connections_12, \
                                connections_13, pointer_connections_13, p12scale, p13scale):
    
    eps = 1e-15
    T1 = np.zeros((3))
    
    N = len(xq)
    
    dphi = np.zeros((N,3))
    
    for i in range(N):
        
        aux = np.zeros((3))
        
        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]
        
        Ri = xq[i] - xq
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(r<1e12)[0]:
            
            pscale = 1.0
            
            for ii in range(start_12, stop_12):
                
                if connections_12[ii]==j:
                    
                    pscale = p12scale
                    
            for ii in range(start_13, stop_13):
                
                if connections_13[ii]==j:
                    
                    pscale = p13scale
                    
            r3 = r[j]**3
            r5 = r[j]**5
            
            gamma = min(thole[i], thole[j])
            damp = (alpha[i]*alpha[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)
            
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1 - damp)
            
            for k in range(3):
                
                for l in range(3):
                    
                    dkl = (k==l)*1.0
                    T1[l] = scale3*dkl*r3*pscale - scale5*3*Ri[j,k]*Ri[j,l]*r5*pscale
                    
                aux[k] += np.sum(T1[:] * p[j,:])
                
        dphi[i,:] += aux[:]
        
    return dphi

@jit(nopython=True)
def coulomb_ddphi_multipole_Thole(xq, p, alpha, thole, polar_group, connections_12, pointer_connections_12, \
                                 connections_13, pointer_connections_13, p12scale, p13scale):
    
    eps = 1e-15
    T1 = np.zeros((3))
    
    N = len(xq)
    
    ddphi = np.zeros((N,3,3))
    
    for i in range(N):
        
        aux = np.zeros((3,3))
        
        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]
        
        Ri = xq[i] - xq
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(r<1e12)[0]:
            
            pscale = 1.0
            
            for ii in range(start_12, stop_12):
                
                if connections_12[ii]==j:
                    
                    pscale = p12scale
                    
            for ii in range(start_13, stop_13):
                
                if connections_13[ii]==j:
                    
                    pscale = p13scale
                    
            r3 = r[j]**3
            r5 = r[j]**5
            r7 = r[j]**7
            
            gamma = min(thole[i], thole[j])
            damp = (alpha[i]*alpha[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)
            
            scale5 = 1 - expdamp*(1 - damp)
            scale7 = 1 - expdamp*(1 - damp + 0.6*damp**2)
            
            for k in range(3):
                
                for l in range(3):
                    
                    dkl = (k==l)*1.0
                    
                    for m in range(3):
                        
                        dkm = (k==m)*1.0
                        dlm = (l==m)*1.0
                        
                        T1[m] = -3*(dkm*Ri[j,l] + dkl*Ri[j,m] + dlm*Ri[j,k])*r5*scale5*pscale \
                        + 15*Ri[j,l]*Ri[j,m]*Ri[j,k]*r7*scale7*pscale
                        
                    aux[k][l] += np.sum(T1[:]*p[j,:])
                    
        ddphi[i,:,:] += aux[:,:]
        
    return ddphi

def coulomb_energy_multipole(xq, q, p, p_pol, Q, alphaxx, thole, polar_group, \
                             connections_12, pointer_connections_12, \
                             connections_13, pointer_connections_13, \
                             p12scale, p13scale):
    
    N = len(xq)
    
    point_energy = np.zeros((N))
    
    flag_polar_group = False
    
    dummy = np.zeros((N))
    
    #phi, dphi and ddphi from permanent multipoles
    
    phi = coulomb_phi_multipole(xq, q, p, Q)
    
    dphi = coulomb_dphi_multipole(xq, q, p, Q, dummy, dummy, dummy, flag_polar_group)
    
    ddphi = coulomb_ddphi_multipole(xq, q, p, Q)
    
    #phi, dphi and ddphi from induced dipoles
    
    phi_thole = coulomb_phi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                            connections_12, pointer_connections_12, \
                                            connections_13, pointer_connections_13, \
                                            p12scale, p13scale)
    
    dphi_thole = coulomb_dphi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                              connections_12, pointer_connections_12, \
                                              connections_13, pointer_connections_13, \
                                              p12scale, p13scale)
    
    ddphi_thole = coulomb_ddphi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                                connections_12, pointer_connections_12, \
                                                connections_13, pointer_connections_13, \
                                                p12scale, p13scale)
    
    phi += phi_thole
    dphi += dphi_thole
    ddphi += ddphi_thole
    
    point_energy[:] = q[:]*phi[:] + np.sum(p[:] * dphi[:], axis = 1) + (np.sum(np.sum(Q[:]*ddphi[:], axis = 1), axis = 1))/6.
    
    return point_energy

def compute_induced_dipole(xq, q, p, p_pol, Q, alpha, thole, polar_group, \
                           connections_12, pointer_connections_12, \
                           connections_13, pointer_connections_13, \
                           dphi_reac, E):
    
    N = len(xq)
    
    u12scale = 1.0
    u13scale = 1.0
    
    flag_polar_group = True
    
    alphaxx = alpha[:,0,0]
    
    dphi_coul = coulomb_dphi_multipole(xq, q, p, Q, alphaxx, thole, polar_group, flag_polar_group)
    
    dphi_coul_thole = coulomb_dphi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                                   connections_12, pointer_connections_12, \
                                                   connections_13, pointer_connections_13, \
                                                   u12scale, u13scale)
    
    dphi_coul += dphi_coul_thole
    
    SOR = 0.7
    
    for i in range(N):
        
        E_total = (dphi_coul[i]/E + 4*np.pi*dphi_reac[i])*-1
        p_pol[i] = p_pol[i]*(1 - SOR) + np.dot(alpha[i], E_total)*SOR
    
    return p_pol