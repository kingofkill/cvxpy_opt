#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   opt_2.py
@Time    :   2021/03/29 19:32:38
@Author  :   Yanan Li
@Version :   1.0
@Contact :   YaNanLi@bupt.edu.cn
@Desc    :   None
'''

# here put the import lib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

X_c = np.load('./cvxpy/X_c.npy')
R_c = np.load('./cvxpy/R_c.npy')
P = np.load('./cvxpy/P.npy')
M = np.load('./cvxpy/M.npy')
S = np.load('./cvxpy/S.npy')
Q = np.load('./cvxpy/Q.npy')
B = np.load('./cvxpy/B.npy')
U = np.load('./cvxpy/U.npy')
E = np.load('./cvxpy/E.npy')
T = np.load('./cvxpy/T.npy')
A = np.load('./cvxpy/A.npy')

m, n = X_c.shape

I_n = np.ones(n)[:, np.newaxis]
I_m = np.ones(m)[:, np.newaxis]

C = np.array([25,50,25,25])[:,np.newaxis]

Y_c = np.dot(X_c, R_c)
# Matrix variable with shape X_c.shape.
X = cp.Variable(X_c.shape, boolean=True)

F1 = cp.multiply(X.T @ P, cp.inv_pos(U))
F2 = cp.multiply(X.T @ M, cp.inv_pos(E))
F3 = cp.multiply(X.T @ S, cp.inv_pos(T))
F = 1 / n * (cp.sum(F1) + cp.sum(F2) + cp.sum(F3))

V = cp.diag(Y_c @ A @ R_c.T @ X.T) @ Q

# objective = cp.Minimize((C.T @ cp.max(R_c.T @ X.T @ B, axis=1)) - 3000 * F)  can successfully run
# add expression V into objective（as below） will cause "Segmentation fault (core dumped)"
objective = cp.Minimize((C.T @ cp.max(R_c.T @ X.T @ B, axis=1)) + 20 * V - 3000 * F)
constraints = [X.T @ P <= U, X.T @ M <= E, X.T @ S <= T, X @ I_n == I_m]

start = datetime.now()
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI)
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal X var", X.value)
print("optimal Y var", np.dot(X.value, R_c))

print(datetime.now() - start)
np.save('./cvxpy/X_gurobi.npy', X.value)
