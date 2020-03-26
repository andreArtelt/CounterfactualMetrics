#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn_lvq import GmlvqModel, LgmlvqModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from plotting import plot_distmat, plot_distmats_boxplot


def encode_labels(y_test, y_pred):
    enc = OneHotEncoder(categories="auto")
    enc.fit(y_test)

    return enc.transform(y_test).toarray(), enc.transform(y_pred).toarray()


def compute_change_in_distmat_gmlvq(model, x_orig, y_target):
    # Compute change of metric
    Omega = np.dot(model.omega_.T, model.omega_)
    o_new = None
    o_new_dist = float("inf")
    n_dim = x_orig.shape[0]
    
    X = cp.Variable((n_dim, n_dim), PSD=True)

    # Search for suitable prototypes
    target_prototypes = []
    other_prototypes = []
    for p, l in zip(model.w_, model.c_w_):
        if l == y_target:
            target_prototypes.append(p)
        else:
            other_prototypes.append(p)
    
    # For each target prototype: Construct a mathematical program
    for p_i in target_prototypes:
        # Build constraints
        constraints = []
        for p_j in other_prototypes:
            constraints.append(-2. * x_orig.T @ X @ (p_i - p_j) + p_i.T @ X @ p_i - p_j.T @ X @ p_j <= 0)

        # Build costfunction
        f = cp.Minimize(cp.norm1(X - Omega))
        #f = cp.Minimize(cp.norm2(X - Omega))
        prob = cp.Problem(f, constraints)

        # Solve problem
        prob.solve(solver=cp.SCS, verbose=False)
        
        Omega_new = X.value
        
        # Validate distance matrix
        y_pred = None
        min_dist = float("inf")
        for p, l in zip(model.w_, model.c_w_):
            d = np.dot((p - x_orig), np.dot(Omega_new, (p - x_orig)))
            if d < min_dist:
                min_dist = d
                y_pred = l
        if y_pred == y_target:
            d = np.linalg.norm(Omega - Omega_new, 1)
            if d < o_new_dist:
                o_new_dist = d
                o_new = Omega_new
    
    if o_new is not None:
        return [o_new]
    else:
        #print("Did not find a counterfactual metric")
        return []


def compute_change_in_distmat_lgmlvq(model, x_orig, y_target):
    # Compute change of metric
    epsilon = 1e-5
    Omegas = []
    n_dim = x_orig.shape[0]
    
    X = cp.Variable((n_dim, n_dim), PSD=True)

    # Search for suitable prototypes
    model_omegas = [np.dot(o.T, o) for o in model.omegas_]
    target_prototypes = []
    target_omegas = []
    other_prototypes = []
    other_omegas = []
    for p, l, o in zip(model.w_, model.c_w_, model_omegas):
        if l == y_target:
            target_prototypes.append(p)
            target_omegas.append(o)
        else:
            other_prototypes.append(p)
            other_omegas.append(o)
    
    # For each target prototype: Construct a mathematical program
    for p_i, o_i in zip(target_prototypes, target_omegas):
        # Build constraints
        constraints = []
        for p_j, o_j in zip(other_prototypes, other_omegas):
            constraints.append(x_orig @ X @ x_orig - 2. * x_orig.T @ X @ p_i + p_i.T @ X @ p_i - x_orig.T @ o_j @ x_orig + 2. * x_orig.T @ o_j @ p_j - p_j.T @ o_j @ p_j + epsilon <= 0)

        # Build costfunction
        f = cp.Minimize(cp.norm1(X - o_i))
        #f = cp.Minimize(cp.norm2(X - o_i))
        prob = cp.Problem(f, constraints)

        # Solve problem
        prob.solve(solver=cp.MOSEK, verbose=False)
        
        Omega_new = X.value
        
        # Validate distance matrix
        y_pred = None
        min_dist = float("inf")
        for p, l, o in zip(model.w_, model.c_w_, model_omegas):
            d = np.dot((p - x_orig), np.dot(o, (p - x_orig)))
            if np.array_equal(p, p_i):
                d = np.dot((p - x_orig), np.dot(Omega_new, (p - x_orig)))

            if d < min_dist:
                min_dist = d
                y_pred = l
        print("Prediction under new distance matrix: {0}".format(y_pred))
        if y_pred == y_target:
            Omegas.append((Omega_new, o_i))

    # Plot matrices
    for o_new, o in Omegas:
        print(o)
        print(o_new)
        print("L1-distance to original matrix: {0}".format(np.linalg.norm(o - o_new, 1)))
        plot_distmat(o - o_new)


if __name__ == "__main__":
    ###############################################################################################################################################
    # TOY-DATASET: An unimportant feature becomes important!
    # Create data set
    n_samples = 50
    X = np.hstack((np.random.uniform(0, 5, n_samples).reshape(-1, 1), np.array([0 for _ in range(n_samples)]).reshape(-1, 1)))
    y = [0 for _ in range(n_samples)]

    X = np.vstack((X, np.hstack((np.random.uniform(7, 12, n_samples).reshape(-1, 1), np.array([5 for _ in range(n_samples)]).reshape(-1, 1)))))
    y += [1 for _ in range(n_samples)]
    y = np.array(y)

    from plotting import plot_classification_dataset, export_as_png
    plot_classification_dataset(X, y, show=False)
    export_as_png("toydata.png")

    # Fit model
    model = GmlvqModel(prototypes_per_class=1, random_state=4242)
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    y_, y_pred_ = encode_labels(y.reshape(-1, 1), y_pred.reshape(-1, 1))
    print("ROC-AUC: {0}".format(roc_auc_score(y_, y_pred_, average="weighted")))

    print("Omega\n{0}".format(np.dot(model.omega_.T, model.omega_)))
    print()

    # Compute counterfactual metric
    x_orig = np.array([10.0, 0])
    y_target = 1
    Omega_cf = compute_change_in_distmat_gmlvq(model, x_orig, y_target)[0]
    print("Omega_cf\n{0}".format(Omega_cf))

    plot_distmat(np.abs(np.dot(model.omega_.T, model.omega_)), show=False)
    export_as_png("omega.png")
    plot_distmat(np.abs(Omega_cf), show=False)
    export_as_png("omegacf.png")
    #plot_distmats_boxplot(np.abs(np.dot(model.omega_.T, model.omega_)))
    #plot_distmats_boxplot(np.abs(Omega_cf))
    ##################################################################################################################################################
