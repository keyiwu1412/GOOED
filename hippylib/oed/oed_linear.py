# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.
from __future__ import absolute_import, division, print_function

import dolfin as dl
import numpy.matlib
import math
import numpy as np
from itertools import combinations 
import scipy
import matplotlib.pyplot as plt

from ReducedHessian_J import ReducedHessian_J

import sys
import os

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../..") + "/applications/ad_diff/" )
from model_ad_diff import TimeDependentAD, SpaceTimePointwiseStateObservation
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

seed = 1
np.random.seed(seed=seed)

#######compute Hessian eigen value decomp#################################

def ComputeHessianEvalue(misfit,Vh,observation_times,targets,noise_std_dev,mesh,prior,simulation_times,wind_velocity,r,x):
	misfit2 = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)
	misfit2.noise_variance = noise_std_dev*noise_std_dev
	problem = TimeDependentAD(mesh, [Vh,Vh,Vh], prior, misfit2, simulation_times, wind_velocity, True)

	for t in range(misfit2.observation_times.shape[0]):
		misfit2.d.data[t][0:r]=misfit.d.data[t][ur_index]

	H2 = ReducedHessian(problem, misfit_only=True) 

	k = r
	p = 2
	# print( "Single Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )
	Omega = MultiVector(x[PARAMETER], k+p)
	parRandom.normal(1., Omega)
	# lmbda, V = singlePassG(H, prior.R, prior.Rsolver, Omega, k)
	d, U = doublePassG(H2, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
	return d,U





#####Construct the velocity field#########################

def v_boundary(x,on_boundary):
	return on_boundary

def q_boundary(x,on_boundary):
	return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS

def computeVelocityField(mesh):
	Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
	Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
	if dlversion() <= (1,6,0):
		XW = dl.MixedFunctionSpace([Xh, Wh])
	else:
		mixed_element = dl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
		XW = dl.FunctionSpace(mesh, mixed_element)

	Re = 50#1e2

	g = dl.Expression(('0.0','(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'), degree=1)
	bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
	bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
	bcs = [bc1, bc2]

	vq = dl.Function(XW)
	(v,q) = dl.split(vq)
	(v_test, q_test) = dl.TestFunctions (XW)

	def strain(v):
		return dl.sym(dl.nabla_grad(v))

	F = ( (2./Re)*dl.inner(strain(v),strain(v_test))+ dl.inner (dl.nabla_grad(v)*v, v_test)
		   - (q * dl.div(v_test)) + ( dl.div(v) * q_test) ) * dl.dx

	dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
										 {"relative_tolerance":1e-4, "maximum_iterations":100}})

	# fig = plt.figure(figsize=(15,5))
	# filename = 'figure/velocity.pdf'
	# vh = dl.project(v,Xh)
	# qh = dl.project(q,Wh)
	# nb.plot(nb.coarsen_v(vh), subplot_loc=121,mytitle="Velocity")
	# nb.plot(qh, subplot_loc=122,mytitle="Pressure")
	# fig.savefig(filename, format='pdf')
	# plt.close()

	return v

if __name__ == "__main__":
	#####Set up the mesh and finite element spaces#########################

	mesh = dl.refine( dl.Mesh("ad_20.xml") )

	wind_velocity = computeVelocityField(mesh)
	Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
	para_dim = Vh.dim()
	print( "Number of dofs: {0}".format( Vh.dim() ) )

	#####Set up model (prior, true/proposed initial condition)#########################
	ic_expr = dl.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=Vh.ufl_element())
	true_initial_condition = dl.interpolate(ic_expr, Vh).vector()


	gamma = 1.
	delta = 8.
	prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
	print( "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2) )
	prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()

	# ##############
	# #test for sqrtC
	ide = dl.Vector()
	prior.sqrtM.init_vector(ide,1)
	mshape = ide.get_local().shape[0]
	Csqrt = np.zeros((para_dim,mshape))
	for i in range(mshape): 
		sqrtMx = dl.Vector()
		prior_help = dl.Vector()
		prior.sqrtM.init_vector(sqrtMx,0)
		prior.sqrtM.init_vector(prior_help,0)

		Idev = np.zeros(mshape)
		Idev[i] = 1
		ide.set_local(Idev)

		prior.sqrtM.mult(ide,sqrtMx)
		prior.Asolver.solve(prior_help,sqrtMx)
		Csqrt[:,i] = prior_help.get_local() 

	t_init         = 0.
	t_final        = 4.
	t_1            = 1.
	dt             = .1
	observation_dt = .2

	simulation_times = np.arange(t_init, t_final+.5*dt, dt)
	observation_times = np.array([t_final])#np.arange(t_1, t_final+.5*dt, observation_dt)

	# targets = np.loadtxt('targets.txt')
	# ntargets = 80
	ndim = 2


	# targets_x = np.random.uniform(0.1,0.5, [ntargets] )
	# targets_y = np.random.uniform(0.4,0.9, [ntargets] )
	x1 = np.array([0.2,0.55,0.8])#np.arange(0.1,1.,0.1)#np.array([0.2,0.55,0.8])
	x2 = np.array([0.25,0.5,0.75])#np.arange(0.1,1.,0.1)#np.array([0.25,0.5,0.75])
	X1,X2 = np.meshgrid(x1, x2)
	x_2d = np.array([X1.flatten('F'),X2.flatten('F')])
	xtargets = x_2d.T
	# ntargets = targets.shape[0]
	targets = []
	for i in range(xtargets.shape[0]):
		pt = xtargets[i]
		if pt[0] > 0.25 and pt[0] < 0.5 and pt[1] > 0.15 and pt[1] < 0.4:
			pass
		elif pt[0] > 0.6 and pt[0] < 0.75 and pt[1] > 0.6 and pt[1] < 0.85:
			pass
		else:
			targets.append(pt)
	ntargets = len(targets)
	targets = np.array(targets)

	print ("Number of observation points: {0}".format(targets.shape[0]) )
	misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

	problem = TimeDependentAD(mesh, [Vh,Vh,Vh], prior, misfit, simulation_times, wind_velocity, True)
	problem.gauss_newton_approx = True

	# objs = [dl.Function(Vh,true_initial_condition),
	# 		dl.Function(Vh,prior.mean)]
	# mytitles = ["True Initial Condition", "Prior mean"]

	# nb.multi1_plot(objs, mytitles)

	#####Generate the synthetic observations#########################
	rel_noise = 0.01
	utrue = problem.generate_vector(STATE)
	x = [utrue, true_initial_condition, None]
	problem.solveFwd(x[STATE], x)
	misfit.observe(x, misfit.d)
	MAX = misfit.d.norm("linf", "linf")
	noise_std_dev = rel_noise * MAX
	parRandom.normal_perturb(noise_std_dev,misfit.d)
	misfit.noise_variance = noise_std_dev*noise_std_dev

	# nb.show_solution(Vh, true_initial_condition, utrue, "Solution",times=[0,1.,t_final])



	# fig = plt.figure(figsize=(15,5))
	# filename = 'observation.pdf'


	# utrue.store(true_initial_condition, 0)

	# title_stamp = " at time {0}s" 

	# myu = dl.Function(Vh)
	# utrue.retrieve(myu.vector(),t_final)


	# vmax = max( utrue.data[0].max(), misfit.d.data[0].max() )
	# vmin = min( utrue.data[0].min(), misfit.d.data[0].min() )
	# nb.plot(myu, subplot_loc=121,  mytitle="True State",vmin=vmin, vmax=vmax)

	# # nb.plot(dl.Function(Vh[STATE], utrue), mytitle="True State", subplot_loc=121, vmin=vmin, vmax=vmax)
	# nb.plot_pts(targets, misfit.d.data[0], mytitle="Observations", subplot_loc=122, vmin=vmin, vmax=vmax)
	# fig.savefig(filename, format='pdf')
	# plt.close()




	#####Evaluate the gradient#########################
	[u,m,p] = problem.generate_vector()
	problem.solveFwd(u, [u,m,p])
	problem.solveAdj(p, [u,m,p])
	mg = problem.generate_vector(PARAMETER)
	grad_norm = problem.evalGradientParameter([u,m,p], mg)

	print( "(g,g) = ", grad_norm)
	H = ReducedHessian(problem, misfit_only=True) 

	k = ntargets
	p = 2
	Omega = MultiVector(x[PARAMETER], k+p)
	parRandom.normal(1., Omega)
	lmbda, V = doublePassG(H, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
	posterior = GaussianLRPosterior( prior, lmbda, V )
	post_tr, _, _= posterior.trace(method="Randomized", r=300)
	print("post trace:", post_tr)

	r = 3
	rall = 3
	max_loop = 10
	# Test = 200

	J = ReducedHessian_J(problem, misfit_only=True, jacobi = True) 

	Jmatrix = np.zeros((targets.shape[0],para_dim))

	Hx = dl.Vector(misfit.B.mpi_comm())
	misfit.B.init_vector(Hx, 0)
	for i in range(para_dim):
		iden_v = np.zeros(para_dim)
		iden_v[i] = 1.
		ide = problem.generate_vector(PARAMETER)
		ide.set_local(iden_v)
		Hx.zero()
		J.mult(ide,Hx)
		Jmatrix[:,i] = Hx.get_local()



	# print(Jmatrix,Csqrt)
	JsqrtCt = np.dot(Jmatrix,Csqrt)
	JCJ = np.dot(JsqrtCt,JsqrtCt.T)
	D,u0 = np.linalg.eigh(JCJ)
	print("eig:",D)

	# J = Jacobian(problem) 
	# k = rall
	# p = 20
	# Omega = MultiVector(x[PARAMETER], k+p)
	# parRandom.normal(1., Omega)
	# U,D,V = accuracyEnhancedSVD(J,Omega,k,s=2,check=True)

	# UDVt = np.zeros((ntargets,para_dim))

	# y = dl.Vector(U[0])
	# for i in range(para_dim):
	#     iden_v = np.zeros(para_dim)
	#     iden_v[i] = 1.
	#     ide = model.generate_vector(PARAMETER)
	#     ide.set_local(iden_v)

	#     Vtx = V.dot_v(ide)
	#     dVtx = D*Vtx   #elementwise mult

	#     y.zero()
	#     U.reduce(y, dVtx)
	#     UDVt[:,i] = y.get_local()


	# fig = plt.figure(figsize=(15,5))
	# filename = 'figure/eigenvalue.pdf'
	# plt.plot(range(0,d1.shape[0]), d1, 'b*')
	# plt.yscale('log')
	# plt.xlabel('number')
	# plt.ylabel('eigenvalue')
	# fig.savefig(filename, format='pdf')
	# plt.close()

	# fig = plt.figure(figsize=(15,5))
	# filename = 'figure/sensor.pdf'

	# r = 30
	# ev = 0.
	# for i in range(r):
	#     # print(np.linalg.norm(u[i],np.inf))
	#     ev += u[:,i]**2 
	# Hx.set_local(ev)
	# nb.plot_pts(targets, Hx, mytitle="norm", subplot_loc=121, vmin=vmin, vmax=vmax)

	print("ntargets",ntargets,"r:",r,"rall:",rall)
	trace = []#np.zeros(Test)
	det = []#np.zeros(Test)
	sensor_all = []
	test = 0

	u = u0[:,0:rall]


	########################################################################
	# print("method one of finding the dominant submatrix")
	# #dominant submatrix
	# #start with arbitrary r*r submatrix ur
	# # ur_index = np.arange(r)#np.random.choice(ntargets, size=r, replace=False, p=None)
	# # ur_index_all = np.arange(0,ntargets)
	# # Q,R,P = scipy.linalg.qr(u.T, pivoting=True)
	# # ur_index = P[0:r]
	# # ur_index_all = P
	ev = np.zeros(ntargets)
	for i in range(ntargets):
		# [ur,dr,vr] = np.linalg.svd(u[i,0:r])
		# print(i,u[i,0:r],np.dot(u[i,0:r],u[i,0:r].T))
		if rall > 1:
			ev[i] = np.linalg.norm(u[i,:])
		else:
			ev[i] = u[i]/np.linalg.norm(u)
		# ev[i] = np.linalg.norm(np.arccos(u[i,:]))
	# ev = ev/np.max(ev)
	# print(u,ev)
	ev = np.arccos(ev)
	sort_index = np.argsort(ev)
	ur_index = sort_index[0:r].copy()
	ur_index_all = sort_index.copy()






	ur = u[ur_index,:]
	ur_all = u[ur_index_all,:]
	converge = False
	loop = 0
	count_all = 0
	while loop < max_loop and not converge:
		count = 0
		for i in range(r):
			sensor_matrix = np.zeros((r,ntargets))
			for k in range(r):
				sensor_matrix[k][ur_index[k]] = 1


			Hmatrix = np.dot(np.dot(sensor_matrix,JCJ),sensor_matrix.T)
			Ai = np.log(np.abs(np.linalg.det(Hmatrix+np.identity(r))))
			# d, U = ComputeHessianEvalue(misfit,Vh,observation_times,targets[ur_index],noise_std_dev,mesh,prior,simulation_times,wind_velocity,r,x)
			# print(np.sum(np.log(1.+d)),Ai)
			# dfdfdf
			# print(ur_index_all)
			for j in range(r,ntargets):
				ur_index_all_tmp = ur_index_all.copy()
				ur_index_all_tmp[[i,j]]=ur_index_all_tmp[[j,i]]
				ur_index_tmp = ur_index_all_tmp[0:r]



				sensor_matrix = np.zeros((r,ntargets))
				for k in range(r):
					sensor_matrix[k][ur_index_tmp[k]] = 1
		 
				Hmatrix = np.dot(np.dot(sensor_matrix,JCJ),sensor_matrix.T)
				Aj = np.log(np.abs(np.linalg.det(Hmatrix+np.identity(r))))

				# print(i,j,ur_index_all_tmp)
				# print(Ai,Aj)
				if Aj>Ai:#np.abs(Aj) > np.abs(Ai):
					count += 1
					ur_index_all = ur_index_all_tmp.copy()
					ur_index = ur_index_tmp.copy()
					Ai = Aj.copy()
		count_all += count
		loop += 1
		if count == 0:
			converge = True
		print("count of swap:",count)
	print("count of swap all:",count_all)
	d, U = ComputeHessianEvalue(misfit,Vh,observation_times,targets[ur_index],noise_std_dev,mesh,prior,simulation_times,wind_velocity,r,x)

	posterior = GaussianLRPosterior( prior, d, U )
	post_tr, _, _= posterior.trace(method="Randomized", r=300)
	# sensor = ur_index
	sensor = np.sort(ur_index)

	trace.append(post_tr)
	det.append(np.sum(np.log(1.+d)))
	# fig = plt.figure(figsize=(7.5,5))
	# filename0 = 'sensor_choice_'+ str(ntargets) + '_'+str(r)+'.pdf'
	# # nb.plot(dl.Function(Vh[STATE], utrue), mytitle="True State", subplot_loc=121, vmin=vmin, vmax=vmax)
	# # nb.plot_pts(targets[sensor], misfit.d[sensor], mytitle="sensor", vmin=vmin, vmax=vmax)
	# nb.plot_pts(targets[sensor], misfit.d.data[0][sensor], mytitle="sensor",vmin=vmin, vmax=vmax)
	# fig.savefig(filename0, format='pdf')
	# plt.close()

	print("sensor:",sensor)
	print("det:",np.sum(np.log(1.+d)),"trace:",post_tr)


	############################################################################
	choices = list(combinations(np.arange(ntargets),r))
	Test = 3#len(choices)


	#############################################################################
	filename = 'adv_num_obs_' + str(ntargets) + '_'+ str(r) + '_' + str(rall) + '_test_' + str(Test)
	np.savez(filename,sensor=sensor,det=det,trace=trace)



	for test in range(1,Test+1):
		if r > 1:
			sensor = np.array(choices[test-1])#np.random.choice(ntargets, size=r, replace=False, p=None)#np.array(choices[test-1])
		else:
			sensor = [test-1]
		d, U = ComputeHessianEvalue(misfit,Vh,observation_times,targets[sensor],noise_std_dev,mesh,prior,simulation_times,wind_velocity,r,x)

		posterior = GaussianLRPosterior( prior, d, U )
		post_tr, _, _= posterior.trace(method="Randomized", r=300)
		sensor = np.sort(sensor)
		sensor_all.append(sensor)
		trace.append(post_tr)
		det.append(np.sum(np.log(1.+d)))

		print(test)
		print("sensor:",sensor)
		print("det:",np.sum(np.log(1.+d)),"trace:",post_tr)

		# filename = 'num_obs_' + str(ntargets) + '_'+ str(r) + '_' + str(rall) + '_test_' + str(Test)
		np.savez(filename,sensor=sensor,det=det,trace=trace)






