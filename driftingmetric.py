#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
random.seed(42)
import numpy as np
np.random.seed(42)
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn_lvq import GmlvqModel, LgmlvqModel, LmrslvqModel, MrslvqModel
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from plotting import plot_classification_dataset, plot_2d_decisionboundary, export_as_png, export_as_svg, plot_distmats_boxplot, plot_distmat
from metricchange import plot_distmats_boxplot, plot_distmat


def plot_two_mats(a, b, show=True):
    _, (ax0, ax1) = plt.subplots(1, 2)
    ax0.matshow(a)
    ax0.set_title("Covariance matrix drift")
    ax1.matshow(b)
    ax1.set_title("Model distance matrix adaption")
    if show is True:
        plt.show()


def sample_from_classdist(class_means, cov, n_samples_per_class=200):
    X = []
    y = []

    for i in range(class_means.shape[0]):
        y += [i for _ in range(n_samples_per_class)]
        for _ in range(n_samples_per_class):
            X.append(multivariate_normal(mean=class_means[i,:], cov=cov))

    X = np.array(X)
    y = np.array(y)

    return X, y

def sample_from_localclassdists(class_means, covs, n_samples_per_class=200):
    X = []
    y = []

    for i in range(class_means.shape[0]):
        y += [i for _ in range(n_samples_per_class)]
        for _ in range(n_samples_per_class):
            X.append(multivariate_normal(mean=class_means[i,:], cov=covs[i]))

    X = np.array(X)
    y = np.array(y)

    return X, y


def drifting_covs(n_covs=3, n_time=3):
    covs = rotating_covariancematrix()

    results = []
    for _ in range(n_time):
        results.append([random.choice(covs) for _ in range(n_covs)])

    return results


def rotating_covariancematrix(xvar=np.arange(0.1, 15.0, 5.0)):
    x1var = xvar
    x2var = list(reversed(xvar))

    results = []
    for x1v, x2v in zip(x1var, x2var):
        d = np.random.normal()
        results.append(np.array([[x1v, d], [d, x2v]]))

    return results


class MyModel():
    def __init__(self, class_means, distmat):
        self.class_means = class_means
        self.distmat = distmat
    
    def __dist(self, x, p):
        d = x - p
        return np.dot(d, np.dot(self.distmat, d))

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            x = X[i,:]
            y_pred.append(np.argmin([self.__dist(x, self.class_means[j,:]) for j in range(class_means.shape[0])]))

        return np.array(y_pred)
    
class MyModel2():   # Local distance based model (e.g. QDA)
    def __init__(self, class_means, distmats):
        self.class_means = class_means
        self.distmats = distmats
    
    def __dist(self, x, p, o):
        d = x - p
        return np.dot(d, np.dot(o, d))

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            x = X[i,:]
            y_pred.append(np.argmin([self.__dist(x, self.class_means[j,:], self.distmats[j]) for j in range(class_means.shape[0])]))

        return np.array(y_pred)


class MyModelQda():
    def __init__(self, class_means, distmats):
        self.class_means = class_means
        self.distmats = distmats
    
    def __dist(self, x, p, o):
        d = x - p
        return -.5*np.log(np.linalg.det(o)) + .5*np.dot(d, np.dot(o, d))

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            x = X[i,:]
            y_pred.append(np.argmin([self.__dist(x, self.class_means[j,:], self.distmats[j]) for j in range(class_means.shape[0])]))

        return np.array(y_pred)


def adjust_distmat(class_means, distmat_old, X, y, C=1.e-1):
    n_dim = class_means.shape[1]
    epsilon = 1e-1

    X_var = cp.Variable((n_dim, n_dim), PSD=True)
    s_var = cp.Variable(y.shape[0]) # Slack variable for allowing mistakes
    
    # Build constraints
    constraints = []
    for i in range(X.shape[0]):
        x = X[i,:]
        target_class = y[i]

        p_i = class_means[target_class, :]
        for j in range(class_means.shape[0]):
            if j == target_class:
                continue
            p_j = class_means[j, :]
            constraints.append(-2. * x.T @ X_var @ (p_i - p_j) + p_i.T @ X_var @ p_i - p_j.T @ X_var @ p_j + epsilon - s_var[i] <= 0)
    constraints.append(s_var >= 0)

    # Build costfunction
    cost = C*cp.norm1(X_var - distmat_old) + (1.-C)*cp.sum(cp.square(s_var))

    f = cp.Minimize(cost)
    prob = cp.Problem(f, constraints)
    print("DCP: {0}".format(prob.is_dcp()))

    # Solve problem
    prob.solve(solver=cp.SCS, verbose=False)
    
    return X_var.value


def adjust_localdistmats(class_means, distmats_old, X, y, C=1e-1):
    n_dim = class_means.shape[1]
    epsilon = 1e-1

    X_vars = [cp.Variable((n_dim, n_dim), PSD=True) for _ in distmats_old]
    s_var = cp.Variable(y.shape[0]) # Slack variable for allowing mistakes
    
    # Build constraints
    constraints = []
    for i in range(X.shape[0]):
        x = X[i,:]
        target_class = y[i]

        p_i = class_means[target_class, :]
        o_i = X_vars[target_class]
        for j in range(class_means.shape[0]):
            if j == target_class:
                continue
            p_j = class_means[j, :]
            o_j = X_vars[j]
            constraints.append(x.T @ (o_i - o_j) @ x - 2. * x.T @ (o_i @ p_i - o_j @ p_j) + p_i.T @ o_i @ p_i - p_j.T @ o_j @ p_j + epsilon - s_var[i] <= 0)
    constraints.append(s_var >= 0)

    # Build costfunction
    cost = C*cp.sum([cp.norm1(X_vars[i] - distmats_old[i]) for i in range(len(distmats_old))]) + (1.-C)*cp.sum(s_var)

    f = cp.Minimize(cost)
    prob = cp.Problem(f, constraints)
    print("DCP: {0}".format(prob.is_dcp()))

    # Solve problem
    prob.solve(solver=cp.SCS, verbose=False)
    
    return [X_var.value for X_var in X_vars]



if __name__ == "__main__":
    useLocalModel = False
    useDifferentCovs = False
    show_plot=False

    # Evaluation scores
    accuracy_noadjust = []
    accuracy_adjust = []

    # Create initial data set
    class_means = np.array([[0.0, 0.0], [5.0, 8.0]])
    #class_means = np.array([[0.0, 0.0], [5.0, 8.0, ], [8.0, 0.0]])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    #cov = np.array([[0.1, 0.0], [0.0, 5.0]])

    X, y = sample_from_classdist(class_means, cov)
    #plot_classification_dataset(X, y)
    
    model = GmlvqModel(prototypes_per_class=1, random_state=4242)   # Fit model to initial data set
    #model = MrslvqModel(prototypes_per_class=1, random_state=4242)
    model.fit(X, y)
    print(np.dot(model.omega_.T, model.omega_))

    mymodel = MyModel(model.w_, np.dot(model.omega_.T, model.omega_))
    y_pred = mymodel.predict(X)
    print("MyModel: {0}".format(accuracy_score(y, y_pred)))

    #model2 = LgmlvqModel(prototypes_per_class=1, random_state=4242)  
    model2 = LmrslvqModel(prototypes_per_class=1, random_state=4242)
    model2.fit(X, y)
    print([np.dot(o.T, o) for o in model2.omegas_])

    mymodel2 = MyModel2(model2.w_, [np.dot(o.T, o) for o in model2.omegas_])
    y_pred2 = mymodel2.predict(X)
    print("MyModel2: {0}".format(accuracy_score(y, y_pred2)))

    model3 = QuadraticDiscriminantAnalysis(store_covariance=True)
    model3.fit(X, y)

    if useLocalModel is False:
        plot_2d_decisionboundary(mymodel, X, y, title="Accuracy: {0}".format(accuracy_score(y, y_pred)), show=show_plot)
        if show_plot is False:
            export_as_png("distmatoriginal.png")
    else:
        plot_2d_decisionboundary(mymodel2, X, y, title="Accuracy: {0}".format(accuracy_score(y, y_pred2)), show=show_plot)
        if show_plot is False:
            export_as_png("localdistoriginal{0}.png")

    X_old = X
    y_old = y

    # Create rotating data set
    rotating_cov = rotating_covariancematrix() + rotating_covariancematrix()
    print(len(rotating_cov))
    print(list(reversed(rotating_covariancematrix())))
    rotating_data = [sample_from_classdist(class_means, cov) for cov in rotating_cov]
    if useDifferentCovs is True:
        rotating_cov = drifting_covs(n_covs=class_means.shape[0], n_time=3)
        rotating_data = [sample_from_localclassdists(class_means, cov) for cov in rotating_cov]

    # Apply model to rotating data set
    i = 0
    prev_cov = cov
    for data_rot, cur_cov in zip(rotating_data, rotating_cov):
        #plot_classification_dataset(X_rot, y_rot)
        X_rot, y_rot = data_rot

        if useLocalModel is False:
            accuracy_noadjust.append(accuracy_score(y_rot, mymodel.predict(X_rot)))
            plot_2d_decisionboundary(mymodel, X_rot, y_rot, X_old=X_old, y_old=y_old, title="Accuracy: {0}".format(accuracy_score(y_rot, mymodel.predict(X_rot))), show=show_plot)
            if show_plot is False:
                export_as_png("distmatnotadapt{0}.png".format(i))
        else:
            accuracy_noadjust.append(accuracy_score(y_rot, mymodel2.predict(X_rot)))
            plot_2d_decisionboundary(mymodel2, X_rot, y_rot, X_old=X_old, y_old=y_old, title="Accuracy: {0}".format(accuracy_score(y_rot, mymodel2.predict(X_rot))), show=show_plot)
            if show_plot is False:
                export_as_png("localdistmatnotadapt{0}.png".format(i))

        X_old = np.concatenate((X_old, X_rot), axis=0)
        y_old = np.concatenate((y_old, y_rot), axis=0)

        # Adjust model to drifted covariance matrix
        omegas = mymodel2.distmats
        distmats_new = adjust_localdistmats(model2.w_, omegas, X_rot, y_rot)
        if useLocalModel is True:
            for i in range(len(omegas)):
                #plot_distmat(np.abs(omegas[i] - distmats_new[i]))
                print(distmats_new[i])
        mymodel2.distmats = distmats_new

        distmat_new = adjust_distmat(class_means, mymodel.distmat, X_rot, y_rot)
        distmat_diff = np.abs(mymodel.distmat - distmat_new)
        if useLocalModel is False:
            #plot_distmat(np.abs(mymodel.distmat - distmat_new))
            print(distmat_new)
        mymodel.distmat = distmat_new

        # Compare difference of adapted matrix with ground truth adaption - are the same elementes changed (order of magnitute only!) -> e.g. hypothesis test
        cov_diff = np.abs(prev_cov - cur_cov)
        plot_two_mats(cov_diff, distmat_diff, show=show_plot)
        if show_plot is False:
            export_as_png("cov_distmat_diff{0}.png".format(i))
        
        if useLocalModel is False:
            accuracy_adjust.append(accuracy_score(y_rot, mymodel.predict(X_rot)))
            plot_2d_decisionboundary(mymodel, X_rot, y_rot, X_old=X_old, y_old=y_old, title="Accuracy: {0}".format(accuracy_score(y_rot, mymodel.predict(X_rot))), show=show_plot)
            if show_plot is False:
                export_as_png("distmatadapt{0}.png".format(i))
        else:
            accuracy_adjust.append(accuracy_score(y_rot, mymodel2.predict(X_rot)))
            plot_2d_decisionboundary(mymodel2, X_rot, y_rot, X_old=X_old, y_old=y_old, title="Accuracy: {0}".format(accuracy_score(y_rot, mymodel2.predict(X_rot))), show=show_plot)
            if show_plot is False:
                export_as_png("localdistmatadapt{0}.png".format(i))
        #break
        i += 1

    # Plot accuracy over time
    t = range(len(accuracy_adjust))

    plt.figure()
    plt.plot(t, accuracy_noadjust, 'r.-')
    plt.plot(t, accuracy_adjust, 'b.-')
    plt.ylabel("Accuracy")
    plt.xlabel("Time")
    plt.xticks(t)
    plt.legend(["Accuracy of unadjusted model", "Accuracy of adjusted model"])
    if show_plot is True:
        plt.show()
    else:
        export_as_svg("accuracy_driftadaption.svg")
