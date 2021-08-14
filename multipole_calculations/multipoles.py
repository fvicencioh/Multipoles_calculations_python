import bempp.api
import numpy as np
import os
import time
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import inspect
from scipy.sparse.linalg import gmres


from .direct.direct import compute_induced_dipole, coulomb_energy_multipole
from .util.getData import *
from .util.getMesh import *


def generate_nanoshaper_grid(filename):
    
    """
    filename: Name of mesh files, whitout .face or .vert extension
    
    Returns:
    
    grid: Bempp Grid
        
    """
    
    faces = np.loadtxt(filename+".face", dtype= int) - 1
    verts = np.loadtxt(filename+".vert", dtype= float)
    
    grid = bempp.api.Grid(verts.transpose(), faces.transpose())
    
    return grid

def getLHS(dirichl_space, neumann_space, ep_in, ep_ex, kappa, assembler="dense"):
    
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space);
    VL       = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=assembler);
    KL       = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=assembler);
    VY       = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=assembler);
    KY       = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=assembler);
    
    blocked = bempp.api.BlockedOperator(2, 2);
    blocked[0, 0] = 0.5*identity + KL;
    blocked[0, 1] = -VL;
    blocked[1, 0] = 0.5*identity - KY;
    blocked[1, 1] = (ep_in/ep_ex)*VY;
    LHS = blocked.strong_form();
    
    return LHS

def getRHS(x_q, q, p, Q, ep_in, ep_ex, dirichl_space, neumann_space):
    
    @bempp.api.real_callable
    def multipolar_quadrupoles_charges_fun(x, n, i, result):
        T2 = np.zeros((len(x_q),3,3))
        phi = 0
        dist = x - x_q
        norm = np.sqrt(np.sum((dist*dist), axis = 1))
        T0 = 1/norm[:]
        T1 = np.transpose(dist.transpose()/norm**3)
        T2[:,:,:] = np.ones((len(x_q),3,3))[:]* dist.reshape((len(x_q),1,3))* \
        np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5
        phi = np.sum(q[:]*T0[:]) + np.sum(T1[:]*p[:]) + 0.5*np.sum(np.sum(T2[:]*Q[:],axis=1))
        result[0] = (phi/(4*np.pi*ep_in))
        
    charged_grid_fun = bempp.api.GridFunction(dirichl_space, fun = multipolar_quadrupoles_charges_fun)
    RHS = np.concatenate([charged_grid_fun.coefficients, np.zeros(neumann_space.global_dof_count)])
    
    return RHS

def iteration_counter(x):
    global array_it, array_frame, it_count
    it_count += 1
    frame = inspect.currentframe().f_back
    array_it = np.append(array_it, it_count)
    array_frame = np.append(array_frame, frame.f_locals["resid"])
    print(F"Gmres iteration {it_count}, residual: {x}")

def solvation_energy_solvent(q, p, Q, phi, dphi, ddphi):
    
    cal2J = 4.184
    qe = 1.60217646e-19
    Na = 6.0221415e+23
    ep_vacc = 8.854187818e-12
    C0 = qe**2*Na*1e-3*1e10/(cal2J*ep_vacc)
    
    q_aux = 0
    p_aux = 0
    Q_aux = 0
    
    for i in range(len(q)):
        q_aux += q[i]*phi[i]
        
        for j in range(3):
            p_aux += p[i,j]*dphi[i,j]
            
            for k in range(3):
                Q_aux += Q[i,j,k]*ddphi[i,j,k]/6.
                
    solvent_energy = 0.5 * C0 * (q_aux + p_aux + Q_aux)
    
    return solvent_energy

def solvent_potential_first_derivate(xq, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl):
    
    """
    Compute the first derivate of potential due to solvent
    in the position of the points
    Inputs:
    -------
        xq: Array size (Nx3) whit positions to calculate the derivate.
        h: Float number, distance for the central difference.

    Return:

        dpdr: Derivate of the potential in the positions of points.
    """

    dpdr = np.zeros([len(xq), 3])
    dist = np.array(([h,0,0],[0,h,0],[0,0,h]))
    # x axis derivate
    dx = xq[:] + dist[0]
    dx = np.concatenate((dx, xq[:] - dist[0]))
    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dx.transpose())
    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dx.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdx = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,0] = dpdx

    #y axis derivate
    dy = xq[:] + dist[1]
    dy = np.concatenate((dy, xq[:] - dist[1]))
    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dy.transpose())
    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dy.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdy = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,1] = dpdy

    #z axis derivate
    dz = xq[:] + dist[2]
    dz = np.concatenate((dz, xq[:] - dist[2]))
    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dz.transpose())
    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dz.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdz = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,2] = dpdz

    return dpdr

def solvent_potential_second_derivate(x_q, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl):
    
    """
    Compute the second derivate of potential due to solvent
    in the position of the points

    xq: Array size (Nx3) whit positions to calculate the derivate.
    h: Float number, distance for the central difference.

    Return:

    ddphi: Second derivate of the potential in the positions of points.
    """
    ddphi = np.zeros((len(x_q),3,3))
    dist = np.array(([h,0,0],[0,h,0],[0,0,h]))
    for i in range(3):
        for j in np.where(np.array([0, 1, 2]) >= i)[0]:
            if i==j:
                dp = np.concatenate((x_q[:] + dist[i], x_q[:], x_q[:] - dist[i]))
                slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dp.transpose())
                dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dp.transpose())
                phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
                ddphi[:,i,j] = (phi[0,:len(x_q)] - 2*phi[0,len(x_q):2*len(x_q)] + phi[0, 2*len(x_q):])/(h**2)
      
            else:
                dp = np.concatenate((x_q[:] + dist[i] + dist[j], x_q[:] - dist[i] - dist[j], x_q[:] + \
                                     dist[i] - dist[j], x_q[:] - dist[i] + dist[j]))
                slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dp.transpose())
                dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dp.transpose())
                phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
                ddphi[:,i,j] = (phi[0,:len(x_q)] + phi[0,len(x_q):2*len(x_q)] - \
                                phi[0, 2*len(x_q):3*len(x_q)] - phi[0, 3*len(x_q):])/(4*h**2)
                ddphi[:,j,i] = (phi[0,:len(x_q)] + phi[0,len(x_q):2*len(x_q)] - \
                                phi[0, 2*len(x_q):3*len(x_q)] - phi[0, 3*len(x_q):])/(4*h**2)
  
    return ddphi

def get_induced_dipole(x_q, q, p, p_pol, Q, alpha, ep_in, thole, polar_group, \
                       connections_12, connections_13, pointer_connections_12, \
                       pointer_connections_13, dphi_solvent):

    
    p_pol = compute_induced_dipole(x_q, q, p, p_pol, Q, alpha, thole, polar_group, \
                                  connections_12, pointer_connections_12, \
                                  connections_13, pointer_connections_13, \
                                  dphi_solvent, ep_in)
    
    return p_pol

def get_coulomb_energy(x_q, q, p, p_pol, Q, alpha, ep_in, thole, polar_group, connections_12, connections_13, pointer_connections_12, pointer_connections_13, p12scale, p13scale):
    
    
    cal2J = 4.184
    ep_vacc = 8.854187818e-12
    qe = 1.60217646e-19
    Na = 6.0221415e+23
    C0 = qe**2*Na*1e-3*1e10/(cal2J*ep_vacc)
    
    alphaxx = alpha[:,0,0]
    
    point_energy = coulomb_energy_multipole(x_q, q, p, p_pol, Q, alphaxx, thole, np.int32(polar_group), \
                                            np.int32(connections_12), np.int32(pointer_connections_12), \
                                            np.int32(connections_13), np.int32(pointer_connections_13), \
                                            p12scale, p13scale)
    
    coulomb_energy = np.sum(point_energy) * 0.5*C0/(4*np.pi*ep_in)
    
    return coulomb_energy

def quick_compilation():
    """
    Quick compilation needed to Numba Jit
    """
    time_compilation_init = time.time()
    
    dummy = np.zeros((1))
    dummy2 = np.zeros((1,3))
    dummy3 = np.zeros((1,3,3))
    dummy4 = np.zeros((3))
    dummy5 = 1.
    
    compilation1 = get_induced_dipole(dummy2, dummy, dummy2, dummy2, dummy3, dummy3, \
                                      dummy5, dummy4, dummy4, dummy4, dummy4, dummy4, \
                                      dummy4, dummy2)
    compilation2 = get_coulomb_energy(dummy2, dummy, dummy2, dummy2, dummy3, dummy3, \
                                      dummy5, dummy4, dummy4, dummy4, dummy4, dummy4, \
                                      dummy4, dummy5, dummy5)
    
    #compilation1 = coulomb_phi_multipole(dummy2, dummy, dummy2, dummy3)
    #compilation2 = coulomb_dphi_multipole(dummy2, dummy, dummy2, dummy3, dummy3, dummy4, dummy4, False)
    #compilation3 = coulomb_ddphi_multipole(dummy2, dummy, dummy2, dummy3)
    #compilation4 = coulomb_phi_multipole_thole(dummy2, dummy2, dummy1, dummy4, dummy4, dummy4, dummy4, dummy4, dummy4, dummy5)
    #compilation5 = coulomb_dphi_multipole_thole(dummy2, dummy2, dummy1, dummy4, dummy4, dummy4, dummy4, dummy4, dummy4, dummy5)
    #compilation6 = coulomb_ddphi_multipole_thole(dummy2, dummy2, dummy1, dummy4, dummy4, dummy4, dummy4, dummy4, dummy4, dummy5)
    
    
    time_compilation_final = time.time()
    
    print(F"Compilation time (Needed to Numba Jit): {time_compilation_final - time_compilation_init} seconds")

def multipole_calculations(file_path, ep_in, ep_ex, k, h, maxiter=100, gmres_maxiter=500, mu="None", tol=1e-2, gmres_tol=1e-5, external_mesh=False, grid_scale=1.0, probe_radius=1.4):
    
    global array_it, array_frame, it_count
    
    time_init = time.time()
    
    x_q, q, d, Q, alpha, mass, polar_group, thole, \
           connections_12, connections_13, \
           pointer_connections_12, pointer_connections_13, \
           p12scale, p13scale, N = read_tinker(file_path,float)
    
    Nq = len(q)
    
    if mu=="None":
        mu = np.zeros((Nq,3))
    
    time_init = time.time()
    
    if external_mesh==False:
        from multipole_calculations import multipole_calculations_dir_name
        
        filename = file_path.split("/")[-1]
        mesh_dir = os.path.join(multipole_calculations_dir_name, "ExternalSoftware/NanoShaper/meshs/" + filename + "/")
        
        if not os.path.exists(mesh_dir + filename + ".face") \
        and not os.path.exists(mesh_dir + filename + ".vert"):
            
            print("Creating mesh...")
            
            xyzr_filepath = file_path + ".xyzr"
            get_mesh_files(xyzr_filepath, grid_scale, probe_radius)
            split_mesh(mesh_dir + filename)
            
            grid = generate_nanoshaper_grid(mesh_dir + filename)
            
        else:
            
            print (F"Loading mesh from: {mesh_dir}")
            grid = generate_nanoshaper_grid(mesh_dir + filename)
    
    else:
        
        print(F"Loading mesh from: {external_mesh}")
        grid = generate_nanoshaper_grid(external_mesh)
    
    dirichl_space = bempp.api.function_space(grid, "DP", 0)
    neumann_space = bempp.api.function_space(grid, "DP", 0)
    
    x_b = np.zeros((2*dirichl_space.global_dof_count))
    
    assembler="dense"
    
    if grid.number_of_elements>50000:
        assembler="fmm"
    
    lhs = getLHS(dirichl_space, neumann_space, ep_in, ep_ex, k, assembler=assembler)
    
    for iter_number in range(maxiter): #Iterations to compute the induced dipole component
        
        print(F"------- Iteration {iter_number +1} -------")
        
        array_it, array_frame, it_count = np.array([]), np.array([]), 0
        
        p = d + mu #d is the permanent dipole, mu is the polarizable dipole
        
        rhs = getRHS(x_q, q, p, Q, ep_in, ep_ex, dirichl_space, neumann_space)
        
        x, info = gmres(lhs, rhs, x0=x_b, callback=iteration_counter, \
                        callback_type="pr_norm", tol=gmres_tol, maxiter=gmres_maxiter, restart = 1000)
        x_b = x.copy()
        
        solution_dirichl = bempp.api.GridFunction(dirichl_space, coefficients=x[:dirichl_space.global_dof_count])
        solution_neumann = bempp.api.GridFunction(neumann_space, coefficients=x[dirichl_space.global_dof_count:])
        
        #Calculation dphi due to solvent.
        
        dphi_solvent = solvent_potential_first_derivate(x_q, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl)
        
        #Calculation of induced dipole.
        
        mu_b = mu.copy()
        
        time_mu_init = time.time()
        
        mu = get_induced_dipole(x_q, q, d, mu, Q, alpha, ep_in, thole, polar_group, \
                                connections_12, connections_13, pointer_connections_12, \
                                pointer_connections_13, dphi_solvent)
        
        time_mu_final = time.time()
        
        print(F"Compution time for induced dipole: {time_mu_final - time_mu_init} seconds")
        
        dipole_diff = np.max(np.sqrt(np.sum((np.linalg.norm(mu_b-mu,axis=1))**2)/len(mu)))
        if dipole_diff<tol:
            print(F"The induced dipole in dissolved state has converged in {iter_number+1} iterations")
            break
            
        print(F"Induced dipole residual: {dipole_diff}")
        
    #Calculation of phi and ddphi due to solvent once induced dipole has converged
    
    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose())
    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose())
    
    phi_solvent = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    
    ddphi_solvent = solvent_potential_second_derivate(x_q, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl)
    
    G_diss_solv = solvation_energy_solvent(q, d, Q, phi_solvent[0], dphi_solvent, ddphi_solvent)
    
    time_energy_init = time.time()
    
    G_diss_mult = get_coulomb_energy(x_q, q, d, mu, Q, alpha, ep_in, thole, polar_group, connections_12, \
                                     connections_13, pointer_connections_12, pointer_connections_13, p12scale, p13scale)
    
    time_energy_final = time.time()
    
    #Calcution of induced dipole in vacum:
    
    p_pol_vacc = np.zeros((Nq,3))
    
    dipole_diff_vacc = 1.
    iteration = 0
    
    while dipole_diff_vacc>tol:
        
        iteration += 1 
        
        p_pol_prev = p_pol_vacc.copy()
        
        p_pol_vacc = get_induced_dipole(x_q, q, d, p_pol_vacc, Q, alpha, ep_in, thole, polar_group, \
                                        connections_12, connections_13, pointer_connections_12, pointer_connections_13, np.zeros((Nq,3)))
        
        dipole_diff_vacc = np.max(np.sqrt(np.sum((np.linalg.norm(p_pol_prev-p_pol_vacc,axis=1))**2)/len(p_pol_vacc)))
        
        print(F"Induced dipole residual in vacuum: {dipole_diff_vacc}")
        
    print(F"{iteration} iterations for vacuum induced dipole to converge")
    
    G_vacc = get_coulomb_energy(x_q, q, d, p_pol_vacc, Q, alpha, ep_in, thole, polar_group, connections_12, \
                                     connections_13, pointer_connections_12, pointer_connections_13, p12scale, p13scale)
    
    time_final = time.time()
    
    print(F"Solvent contribution: {G_diss_solv} [kcal/Mol]")
    print(F"Multipoles contribution: {G_diss_mult} [kcal/Mol]")
    print(F"Coulomb vacuum energy: {G_vacc}")
    
    print("These values consider the polarization energy")
    
    total_energy = G_diss_solv + G_diss_mult - G_vacc
    
    print(F"Total solvation energy: {total_energy} [kcal/Mol]")
    print(F"Compution time for Coulomb Energy: {time_energy_final - time_energy_init} seconds")
    print(F"Total time: {time_final - time_init} seconds.")
    
    
    return total_energy
    
    