# Author: Keyi Wu 
# keyiwu@math.texas.edu
# University of Texas at Austin


from __future__ import absolute_import, division, print_function

# from .variables import STATE, PARAMETER, ADJOINT
import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
import numpy as np
import dolfin as dl

class ReducedHessian_J:
    """
    This class implements matrix free application of the reduced Hessian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    - :code:`innerTol`:            the relative tolerance for the solution of the incremental forward and adjoint problems.
    - :code:`misfit_only`:         a boolean flag that describes whenever the full Hessian or only the misfit component of the Hessian is used.
    
    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """
    def __init__(self, model, misfit_only=False, jacobi = False):
        """
        Construct the reduced Hessian Operator
        """
        self.model = model
        self.gauss_newton_approx = self.model.gauss_newton_approx 
        self.jacobi = jacobi
        self.misfit_only=misfit_only
        self.ncalls = 0
        
        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat    = model.generate_vector(STATE)
        self.phat    = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape.
        - :code:`dim`: if 0 then :code:`x` will be reshaped to be compatible with the range of the reduced Hessian, if 1 then :code:`x` will be reshaped to be compatible with the domain of the reduced Hessian.
               
        .. note:: Since the reduced Hessian is a self adjoint operator, the range and the domain is the same. Either way, we choosed to add the parameter :code:`dim` for consistency with the interface of :code:`Matrix` in dolfin.
        """
        self.model.init_parameter(x)
        
    def mult(self,x,y):
        """
        Apply the reduced Hessian (or the Gauss-Newton approximation) to the vector :code:`x`. Return the result in :code:`y`.
        """
        if self.jacobi:
            self.Jacobian(x,y)
        else:
            if self.gauss_newton_approx:
                self.GNHessian(x,y)
            else:
                self.TrueHessian(x,y)
        
        self.ncalls += 1
    
    def inner(self,x,y):
        """
        Perform the inner product between :code:`x` and :code:`y` in the norm induced by the reduced
        Hessian :math:`H,\\,(x, y)_H = x' H y`.
        """
        Ay = self.model.generate_vector(PARAMETER)
        Ay.zero()
        self.mult(y,Ay)
        return x.inner(Ay)
            
    def GNHessian(self,x,y):
        """
        Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, y)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)

    # def Jacobian_time(self,x,y):
    #     """
    #     Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
    #     Return the result in :code:`y`.        
    #     """
    #     self.model.applyC(x, self.rhs_fwd)
    #     self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
    #     y.zero()
        
    #     try:
    #         # t = self.model.misfit.observation_times[1]
    #         for t in self.model.misfit.observation_times:
    #             self.uhat.retrieve(self.model.misfit.u_snapshot, t)
    #             self.model.misfit.B.mult(self.model.misfit.u_snapshot, y)
    #             # print(np.sqrt(self.model.misfit.noise_variance))
    #             y *= 1./np.sqrt(self.model.misfit.noise_variance)
    #     except:
    #         self.model.misfit.B.mult(self.uhat, y)
    #         y *= 1./np.sqrt(self.model.misfit.noise_variance)

        
    #     # if not self.misfit_only:
    #     #     self.model.applyR(x,self.yhelp)
    #     #     y.axpy(1., self.yhelp)

    def Jacobian(self,x,y):
        """
        Apply the Gauss-Newton approximation of the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        # xdata = x.get_local()
        # x_shape = xdata.shape[0]
        # index = np.arange(1,x_shape+1,2)
        # xdata[index] = 0.
        # x.set_local(xdata)
        
        self.Ctphat = self.model.generate_vector(PARAMETER)
        self.sqrtMx = dl.Vector()
        self.prior_help = dl.Vector()
        self.quad_help = dl.Vector()
        self.model.prior.sqrtM.init_vector(self.sqrtMx,0)
        self.model.prior.sqrtM.init_vector(self.prior_help,0)
        self.model.prior.sqrtM.init_vector(self.quad_help,1)
        try:
            self.model.misfit.B.transpmult(x,self.rhs_adj)
            self.model.solveAdjIncremental(self.phat, self.rhs_adj)
            self.model.applyCt(self.phat, self.Ctphat)
            self.model.prior.Asolver.solve(self.prior_help,self.Ctphat)
            self.model.prior.sqrtM.transpmult(self.prior_help,self.quad_help)
            self.model.prior.sqrtM.mult(self.quad_help,self.sqrtMx)
            self.model.prior.Asolver.solve(self.prior_help,self.sqrtMx)
            self.model.applyC(self.prior_help, self.rhs_fwd)
            self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
            self.model.misfit.B.mult(self.uhat,y)
            # ydata = y.get_local()
            # y_shape = ydata.shape[0]
            # index = np.arange(1,y_shape+1,2)
            # ydata[index] = 0.
            # y.set_local(ydata)
            y *= (1./self.model.misfit.noise_variance)
        except:
            # self.model.prior.sqrtM.mult(self.quad_help,self.sqrtMx)
            # self.model.prior.Asolver.solve(self.prior_help,self.sqrtMx)
            self.model.applyC(x, self.rhs_fwd)
            self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
            y.zero()
            for t in self.model.misfit.observation_times:
                self.uhat.retrieve(self.model.misfit.u_snapshot, t)
                self.model.misfit.B.mult(self.model.misfit.u_snapshot, y)
                y *= 1./np.sqrt(self.model.misfit.noise_variance)

        
    def TrueHessian(self, x, y):
        """
        Apply the the reduced Hessian to the vector :code:`x`.
        Return the result in :code:`y`.        
        """
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWum(x, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWmm(x, y)
        self.model.applyCt(self.phat, self.yhelp)
        y.axpy(1., self.yhelp)
        self.model.applyWmu(self.uhat, self.yhelp)
        y.axpy(-1., self.yhelp)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)
            
            
class FDHessian:
    """
    This class implements matrix free application of the reduced Hessian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    - :code:`m0`:                  the value of the parameter at which the Hessian needs to be evaluated.
    - :code:`h`:                   the mesh size for FD.
    - :code:`innerTol`:            the relative tolerance for the solution of the forward and adjoint problems.
    - :code:`misfit_only`:         a boolean flag that describes whenever the full Hessian or only the misfit component of the Hessian is used.
    
    Type :code:`help(Template)` for more information on which methods model should implement.
    """
    def __init__(self, model, m0, h, innerTol,  misfit_only=False):
        """
        Construct the reduced Hessian Operator
        """
        self.model = model
        self.m0 = m0.copy()
        self.h = h
        self.tol = innerTol
        self.misfit_only=misfit_only
        self.ncalls = 0
        
        self.state_plus  = model.generate_vector(STATE)
        self.adj_plus    = model.generate_vector(ADJOINT)
        self.state_minus = model.generate_vector(STATE)
        self.adj_minus   = model.generate_vector(ADJOINT)
        self.g_plus      = model.generate_vector(PARAMETER)
        self.g_minus     = model.generate_vector(PARAMETER)
        self.yhelp       = model.generate_vector(PARAMETER)
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape
        - :code:`dim`: if 0 then :code:`x` will be reshaped to be compatible with the range of the reduced Hessian, if 1 then :code:`x` will be reshaped to be compatible with the domain of the reduced Hessian
               
        .. note:: Since the reduced Hessian is a self adjoint operator, the range and the domain is the same. Either way, we choosed to add the parameter :code:`dim` for consistency with the interface of :code:`Matrix` in dolfin.
        """
        self.model.init_parameter(x)
        
    def mult(self,x,y):
        """
        Apply the reduced Hessian (or the Gauss-Newton approximation) to the vector :code:`x`.
        Return the result in :code:`y`.
        """
        h = self.h
        
        m_plus = self.m0.copy()
        m_plus.axpy(h, x)
        self.model.solveFwd(self.state_plus, [self.state_plus, m_plus, self.adj_plus], self.tol)
        self.model.solveAdj(self.adj_plus, [self.state_plus, m_plus, self.adj_plus], self.tol)
        self.model.evalGradientParameter([self.state_plus, m_plus, self.adj_plus], self.g_plus, misfit_only = True)
        
        m_minus = self.m0.copy()
        m_minus.axpy(-h, x)
        self.model.solveFwd(self.state_minus, [self.state_minus, m_minus, self.adj_minus], self.tol)
        self.model.solveAdj(self.adj_minus, [self.state_minus, m_minus, self.adj_minus], self.tol)
        self.model.evalGradientParameter([self.state_minus, m_minus, self.adj_minus], self.g_minus, misfit_only = True)
        
        y.zero()
        y.axpy(1., self.g_plus)
        y.axpy(-1., self.g_minus)
        y*=(.5/h)
        
        if not self.misfit_only:
            self.model.applyR(x,self.yhelp)
            y.axpy(1., self.yhelp)

        
        self.ncalls += 1
    
    def inner(self,x,y):
        """
        Perform the inner product between :code:`x` and :code:`y` in the norm induced by the reduced Hessian :math:`H,\\, (x, y)_H = x' H y`.
        """
        Ay = self.model.generate_vector(PARAMETER)
        Ay.zero()
        self.mult(y,Ay)
        return x.inner(Ay)
