#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:09:16 2019

[1] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: 
uses and interpretations. Neuroimage, 52(3), 1059-1069.

https://github.com/fiuneuro/brainconn

@author: Carlos Coronel
"""

import numpy as np

def efficiency_bin(G, local=False):
    """
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.
    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        binary undirected connection matrix
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.
    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 :obj:`numpy.ndarray`
        local efficiency, only if local=True
    """
    def distance_inv(g):
        D = np.eye(len(g))
        n = 1
        nPATH = g.copy()
        L = (nPATH != 0)

        while np.any(L):
            D += n * L
            n += 1
            nPATH = np.dot(nPATH, g)
            L = (nPATH != 0) * (D == 0)
        D[np.logical_not(D)] = np.inf
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    G = binarize(G)
    n = len(G)  # number of nodes
    if local:
        E = np.zeros((n,))  # local efficiency

        for u in range(n):
            # find pairs of neighbors
            V, = np.where(np.logical_or(G[u, :], G[u, :].T))
            # inverse distance matrix
            e = distance_inv(G[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = e + e.T

            # symmetrized adjacency vector
            sa = G[u, V] + G[V, u].T
            numer = np.sum(np.outer(sa.T, sa) * se) / 2
            if numer != 0:
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                E[u] = numer / denom  # local efficiency

    else:
        e = distance_inv(G)
        E = np.sum(e) / (n * n - n)  # global efficiency
    return E


def transitivity_bu(A):
    """
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        binary undirected connection matrix
    Returns
    -------
    T : float
        transitivity scalar
    """
    tri3 = np.trace(np.dot(A, np.dot(A, A)))
    tri2 = np.sum(np.dot(A, A)) - np.trace(np.dot(A, A))
    return tri3 / tri2


def clustering_coef_bu(G):
    """
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        binary undirected connection matrix
    Returns
    -------
    C : Nx1 :obj:`numpy.ndarray`
        clustering coefficient vector
    """
    n = len(G)
    C = np.zeros((n,))

    for u in range(n):
        V, = np.where(G[u, :])
        k = len(V)
        if k >= 2:  # degree must be at least 2
            S = G[np.ix_(V, V)]
            C[u] = np.sum(S) / (k * k - k)

    return C


def distance_bin(G):
    """
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to node v. The average shortest path length is the
    characteristic path length of the network.
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        binary directed/undirected connection matrix
    Returns
    -------
    D : NxN
        distance matrix
    Notes
    -----
    Lengths between disconnected nodes are set to Inf.
    Lengths on the main diagonal are set to 0.
    Algorithm: Algebraic shortest paths.
    """
    G = binarize(G, copy=True)
    D = np.eye(len(G))
    n = 1
    nPATH = G.copy()  # n path matrix
    L = (nPATH != 0)  # shortest n-path matrix

    while np.any(L):
        D += n * L
        n += 1
        nPATH = np.dot(nPATH, G)
        L = (nPATH != 0) * (D == 0)

    D[D == 0] = np.inf  # disconnected nodes are assigned d=inf
    np.fill_diagonal(D, 0)
    return D


def binarize(W, copy = True):
    """
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.
    Returns
    -------
    W : NxN :obj:`numpy.ndarray`
        binary connectivity matrix
    """
    if copy:
        W = W.copy()
    W[W != 0] = 1
    # W = W.astype(int)  # causes tests to fail, but should be included
    return(W)


def invert(W, copy = True):
    """
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.
    If copy is not set, this function will *modify W in place.*
    Parameters
    ----------
    W : :obj:`numpy.ndarray`
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.
    Returns
    -------
    W : :obj:`numpy.ndarray`
        inverted connectivity matrix
    """
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return(W)


def get_uptri(x):
    nnodes = x.shape[0]
    npairs = (nnodes**2 - nnodes) // 2
    vector = np.zeros(npairs)
    
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            vector[idx] = x[row, col]
            idx = idx + 1
    
    return(vector)


def matrix_recon(x):
    npairs = len(x)
    nnodes = int((1 + np.sqrt(1 + 8 * npairs)) // 2)
    
    matrix = np.zeros((nnodes, nnodes))
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            matrix[row, col] = x[idx]
            idx = idx + 1
    matrix = matrix + matrix.T
   
    return(matrix)   

def thresholding(x, threshold = 0.20, zero_diag = True, direct = 'undirected'):
    nnodes = x.shape[0]
    if zero_diag == True:
        np.fill_diagonal(x, 0)
    
    if direct == 'directed':
        nlinks = nnodes**2 - nnodes
        x_vector = x.reshape((1, nnodes**2))
        to_get_links = int(nlinks * threshold)
        sorting = np.argsort(x_vector)[0,::-1]
        selection = sorting[0:to_get_links]
        to_delete = np.delete(np.arange(0, nnodes**2, 1), selection)
        x_thresholded = np.copy(x_vector[0,:])
        x_thresholded[to_delete] = 0
        x_thresholded = x_thresholded.reshape((nnodes, nnodes))
    elif direct == 'undirected':
        nlinks = (nnodes**2 - nnodes) // 2
        x_vector = get_uptri(x)
        to_get_links = int(nlinks * threshold)
        sorting = np.argsort(x_vector)[::-1]
        selection = sorting[0:to_get_links]
        to_delete = np.delete(np.arange(0, nlinks, 1), selection)
        x_vector[to_delete] = 0
        x_thresholded = matrix_recon(x_vector)
    else:
        print('Invalid type of matrix -> direct options: undirected or directed')
    
    return(x_thresholded)