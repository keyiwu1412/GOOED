# Author: Keyi Wu 
# keyiwu@math.texas.edu
# University of Texas at Austin

from __future__ import absolute_import, division, print_function

import dolfin as dl
import math
import numpy as np
from itertools import combinations 
import scipy
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

import logging
from ReducedHessian_J import ReducedHessian_J
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

np.random.seed(seed=2)

#######compute Hessian eigen value decomp#################################

def ComputeHessianEvalue(model,misfit,Vh,targets,sensor,solver,x_map=None):
    misfit2 = PointwiseStateObservation(Vh[STATE], targets)
    misfit2.noise_variance = noise_std_dev*noise_std_dev
    misfit2.d[0:len(sensor)] = misfit.d[sensor].copy()
    problem = Model(model.problem, model.prior, misfit2)

    m = problem.prior.mean.copy()
    solver = ReducedSpaceNewtonCG(problem)
    solver.parameters["rel_tolerance"] = 1e-6
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"]      = 25
    # solver.parameters["inner_rel_tolerance"] = 1e-15
    solver.parameters["GN_iter"] = 5
    solver.parameters["globalization"] = "LS"
    solver.parameters["LS"]["c_armijo"] = 1e-4
    x = solver.solve([None, m, None])

    if x_map is not None:
        objs = [dl.Function(Vh[PARAMETER],x_map[PARAMETER]), dl.Function(Vh[PARAMETER],x[PARAMETER])]
        mytitles = ["map point", "true map point"]
        filename = 'dif.pdf' 
        nb.multi1_plot(objs, mytitles,filename = filename)
        problem.setPointForHessianEvaluations(x_map, gauss_newton_approx=True)
    else:
        problem.setPointForHessianEvaluations(x, gauss_newton_approx=True)

    Hmisfit = ReducedHessian(problem, misfit_only=True)
    k = len(sensor)
    p = 20

    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)
    lmbda, V = doublePassG(Hmisfit, problem.prior.R, problem.prior.Rsolver, Omega, k)   
    
    return lmbda, V, x[PARAMETER]
#####Generate the true parameter#########################

def true_model(prior):
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise,mtrue)
    return mtrue

if __name__ == "__main__":
    #####Set up the mesh and finite element spaces#########################
    ndim = 2
    nx = 32
    ny = 32
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    para_dim = Vh[PARAMETER].dim()
    print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
        Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()) )

    #####Set up the forward problem#########################
    def u_boundary(x, on_boundary):
        return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)

    f = dl.Constant(0.0)

    def pde_varf(u,m,p):
        return dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx

    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)


    gamma = .04
    delta = .2
    anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
    # anis_diff = dl.Expression(code_AnisTensor2D, degree=1)
    anis_diff.theta0 = 2.
    anis_diff.theta1 = .5
    anis_diff.alpha = math.pi/4

    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True)




    mtrue = true_model(prior)

    print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2))    



    model = Model(pde,prior, misfit)

    #####Set up the misfit functional and generate synthetic observations#########################

    rel_noise = 0.01

    #Targets only on the bottom
    x1 = np.arange(0.1,1.,0.1)
    x2 = np.arange(0.1,1.,0.1)

    X1,X2 = np.meshgrid(x1, x2)
    x_2d = np.array([X1.flatten('F'),X2.flatten('F')])
    targets = x_2d.T
    ntargets = targets.shape[0]

    print( "Number of observation points: {0}".format(ntargets) )



    #####low rank approximation of H_d#########################

    N = 5
    r = 10
    rall = 60
    maxloop = 10
    compute_true_map = False
    fix_initial_guess = False
    compute_true = False
    misfit_all = []
    U_all = []
    Umatrix_all = []
    D_all = []
    x_all = []
    map_all = []
    Umatrix_sum = np.zeros((ntargets,rall))
    noise_all = []
    for n in range(N):
        noise_obs = dl.Vector(model.prior.R.mpi_comm())
        mtrue = true_model(prior)
        misfit = PointwiseStateObservation(Vh[STATE], targets)
        utrue = pde.generate_state()
        ptrue = model.generate_vector(ADJOINT)
        x = [utrue, mtrue, ptrue]
        pde.solveFwd(x[STATE], x)
        misfit.B.mult(x[STATE], misfit.d)
        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX

        misfit.B.init_vector(noise_obs,0)
        parRandom.normal(1., noise_obs)
        misfit.d.axpy(noise_std_dev, noise_obs)
        noise_all.append(noise_obs)

        # parRandom.normal_perturb(noise_std_dev, misfit.d)
        misfit.noise_variance = noise_std_dev*noise_std_dev
        misfit_all.append(misfit)



        model = Model(pde, prior, misfit)
        # ptrue = model.generate_vector(ADJOINT)
        model.solveAdj(x[ADJOINT], x)
        x_all.append(x)
        #####Evaluate Jacobian#########################
        m = prior.mean.copy()
        solver = ReducedSpaceNewtonCG(model)
        solver.parameters["rel_tolerance"] = 1e-6
        solver.parameters["abs_tolerance"] = 1e-12
        solver.parameters["max_iter"]      = 25
        # solver.parameters["inner_rel_tolerance"] = 1e-15
        solver.parameters["GN_iter"] = 5
        solver.parameters["globalization"] = "LS"
        solver.parameters["LS"]["c_armijo"] = 1e-4
        solver.parameters["print_level"] = -1


        k = rall
        p = 20

        obs_vector = dl.Vector()
        misfit.B.init_vector(obs_vector,0)
        Omega_u = MultiVector(obs_vector, k+p)
        parRandom.normal(1., Omega_u)
        model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
        JCJT = ReducedHessian_J(model, misfit_only=True, jacobi = True)

        D, U = doublePass(JCJT, Omega_u, k, s=1) 
        
        U_all.append(U)
        D_all.append(D)


        Umatrix = np.zeros((ntargets,rall))
        for i in range(rall):
            Umatrix[:,i] = U[i].get_local()
            Umatrix_sum[:,i] += U[i].get_local()
        Umatrix_all.append(Umatrix)


    print("ntargets",ntargets,"r:",r,"rall:",rall)
    eig = []#np.zeros(Test)
    det = []#np.zeros(Test)
    sensor_all = []

    u = Umatrix_sum/N

    ev = np.zeros(ntargets)
    for i in range(ntargets):
        if rall > 1:
            ev[i] = np.linalg.norm(u[i,:])
        else:
            ev[i] = u[i]/np.linalg.norm(u)

    ev = np.arccos(ev)
    sort_index = np.argsort(ev)
    print(sort_index[0:r])
    ur_index = sort_index[0:r].copy()
    ur_index_all = sort_index.copy()

    #####Swapping greedy algorithm#########################
    print("Swapping greedy algorithm")
    print("initial:",sort_index[0:r])
    ur_index = sort_index[0:r].copy()
    ur_index_all = sort_index.copy()

    # count = 0
    # K = 0
    # ur_index_old = ur_index.copy()
    converge = False
    # count_inner = np.inf
    loop = 0
    count_all = 0
    while loop < maxloop and not converge:
        print("loop:",loop)
        count = 0
        for i in range(r):
            Di_op = 0.
            sensor_matrix = np.zeros((r,ntargets))
            for k in range(r):
                sensor_matrix[k][ur_index[k]] = 1
            for n in range(N):
                D = D_all[n]
                U = Umatrix_all[n]
                newJ = np.dot(sensor_matrix,U)
                Hmatrix = np.dot(newJ,np.dot(np.diag(D),newJ.T))
                eig_H,_ = np.linalg.eigh(Hmatrix)         

                Di_op += np.sum(np.log(eig_H+1.)-eig_H/(1.+eig_H))
            for j in range(r,ntargets):
                ur_index_all_tmp = ur_index_all.copy()
                ur_index_all_tmp[[i,j]]=ur_index_all_tmp[[j,i]]
                ur_index_tmp = ur_index_all_tmp[0:r]
                Dj_op = 0.
                sensor_matrix = np.zeros((r,ntargets))
                for k in range(r):
                    sensor_matrix[k][ur_index_tmp[k]] = 1
                for n in range(N):
                    D = D_all[n]
                    U = Umatrix_all[n]
                    newJ = np.dot(sensor_matrix,U)
                    Hmatrix = np.dot(newJ,np.dot(np.diag(D),newJ.T))
                    eig_H,_ = np.linalg.eigh(Hmatrix)         

                    Dj_op += np.sum(np.log(eig_H+1.)-eig_H/(1.+eig_H))
      
                if Dj_op > Di_op:
                    count += 1
                    ur_index_all = ur_index_all_tmp.copy()
                    ur_index = ur_index_tmp.copy()
                    Di_op = Dj_op.copy()

        loop += 1
        count_all += count
        if count == 0:
            converge = True
        print("count of swap:",count)


    #####compute LA EIG for greedy result#########################

    ps_tr = 0.
    dkl = 0.
    sensor = np.sort(ur_index)
    for n in range(N):
        misfit = misfit_all[n]
        noise_obs = noise_all[n]
        lmbda, U, mmap = ComputeHessianEvalue(model,misfit,Vh,targets[ur_index],ur_index,solver)

        posterior = GaussianLRPosterior( prior, lmbda, U )
        posterior.mean = mmap
        post_tr, _, _= posterior.trace(method="Randomized", r=300)
        ps_tr += post_tr
        dkl += 0.5*np.sum(np.log(1.+lmbda)-lmbda/(1.+lmbda))+ model.prior.cost(mmap)


    eig.append(dkl/N)
    sensor_all.append(sensor)
    print("#####greedy result######")
    print("sensor:",sensor)
    print("eig:",dkl/N)
    print("count of swap:",count)




    filename = 'eig_num_test_' +str(N) + '_num_obs_' + str(ntargets) + '_'+ str(r) + '_' + str(rall) + '_test'# + '_test_' + str(Test)

    np.savez(filename,sensor=sensor_all,eig=eig)




    ########LA-EIG for random designs###########################
    # choices = list(combinations(np.arange(ntargets),r))
    Test = 100#len(choices)

    print("#####random choice result######")
    #############################################################################


    for test in range(1,Test+1):
        if r > 1:
            sensor = np.random.choice(ntargets, size=r, replace=False, p=None)#np.array(choices[test-1])#
        else:
            sensor = [test-1]
        ps_tr = 0.
        dkl = 0.
        sensor = np.sort(sensor)
        for n in range(N):
            misfit = misfit_all[n]
            noise_obs = noise_all[n]
            lmbda, U, mmap = ComputeHessianEvalue(model,misfit,Vh,targets[sensor],sensor,solver)
            posterior = GaussianLRPosterior( prior, lmbda, U )
            posterior.mean = mmap
            post_tr, _, _= posterior.trace(method="Randomized", r=300)
            ps_tr += post_tr
            dkl += 0.5*np.sum(np.log(1.+lmbda)-lmbda/(1.+lmbda)) + model.prior.cost(mmap)

        

        eig.append(dkl/N)
        sensor_all.append(sensor)

        print(test,"sensor:",sensor)
        print("eig:",dkl/N)

        np.savez(filename,sensor=sensor_all,eig=eig)

    print(max(eig))




