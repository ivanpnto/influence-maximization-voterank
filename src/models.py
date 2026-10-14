# -*- coding: utf-8 -*-
import numpy as np

# ---------------------------
# VoteRank Variants
# ---------------------------

def vote_rank(G, k):
    """Implementación clásica de VoteRank (no ponderado)."""
    G = G.copy()
    seeds = []
    ability = {v: 1 for v in G.nodes()}
    
    k_avg = sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 1

    for _ in range(k):
        scores = {}
        for v in G.nodes():
            if v not in seeds:
                scores[v] = sum(ability[u] for u in G.neighbors(v))

        if not scores:
            break

        selected = max(scores, key=scores.get)
        seeds.append(selected)
        ability[selected] = 0

        for neighbor in G.neighbors(selected):
            ability[neighbor] = max(0, ability[neighbor] - 1/k_avg)

    return seeds


def w_vote_rank(G, k, weight_attr='weight'):
    """Implementación de Weighted VoteRank (WVoteRank)."""
    G = G.copy()
    seeds = []
    ability = {v: 1 for v in G.nodes()}
    
    k_avg = sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 1

    for _ in range(k):
        scores = {}
        for v in G.nodes():
            if v not in seeds:
                scores[v] = sum(
                    ability[u] * G[v][u].get(weight_attr, 1)
                    for u in G.neighbors(v)
                )

        if not scores:
            break

        selected = max(scores, key=scores.get)
        seeds.append(selected)
        ability[selected] = 0

        for neighbor in G.neighbors(selected):
            ability[neighbor] = max(
                0,
                ability[neighbor] - 1/k_avg
            )

    return seeds


def iw_vote_rank(G, k, weight_attr='weight'):
    """Implementación de Improved Weighted VoteRank (IWVoteRank con 2-hop)."""
    G = G.copy()
    seeds = []
    ability = {v: 1 for v in G.nodes()}
    
    k_avg = sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 1

    for _ in range(k):
        scores = {}
        for v in G.nodes():
            if v in seeds:
                continue

            score = 0
            for u in G.neighbors(v):
                w1 = G[v][u].get(weight_attr, 1)
                score += ability[u] * w1

                for w in G.neighbors(u):
                    if w != v:
                        w2 = G[u][w].get(weight_attr, 1)
                        score += ability[w] * w1 * w2

            scores[v] = score

        if not scores:
            break

        selected = max(scores, key=scores.get)
        seeds.append(selected)
        ability[selected] = 0

        # Descuento a 1-hop
        for u in G.neighbors(selected):
            ability[u] = max(0, ability[u] - 1/k_avg)

            # Descuento a 2-hop
            for w in G.neighbors(u):
                if w != selected:
                    ability[w] = max(0, ability[w] - 1/(2*k_avg))

    return seeds


# ---------------------------
# Random Walk with Restart (RWR) & WVotedRank Integration
# ---------------------------

def rwr(G, seed, alpha=0.15, max_iter=100, tol=1e-6, weight_attr='weight'):
    """Calcula Random Walk with Restart (RWR) desde un nodo semilla."""
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    P = np.zeros((n, n))
    
    for u in nodes:
        i = index[u]
        neighbors = list(G.neighbors(u))
        
        if len(neighbors) == 0:
            continue
        
        Z = sum(G[u][v].get(weight_attr, 1) for v in neighbors)
        if Z == 0:
            continue
        
        for v in neighbors:
            j = index[v]
            P[i][j] = G[u][v].get(weight_attr, 1) / Z
    
    r = np.zeros(n)
    r[index[seed]] = 1
    
    x = r.copy()
    
    for _ in range(max_iter):
        x_new = (1 - alpha) * np.dot(x, P) + alpha * r
        
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        
        x = x_new
    
    return {nodes[i]: x[i] for i in range(n)}


def compute_wvoterank_score(G, weight_attr='weight'):
    """Calcula la puntuación estructural basada en WVoteRank."""
    score = {}
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        degree = len(neighbors)
        
        if degree == 0:
            score[v] = 0
        else:
            weighted_sum = sum(G[v][u].get(weight_attr, 1) for u in neighbors)
            score[v] = np.sqrt(degree * weighted_sum)
            
    return score


def rwr_with_wvr_optionA(G, seed, alpha=0.15):
    """Integra RWR con la activación estructural WVotedRank (Opción A)."""
    rwr_scores = rwr(G, seed, alpha=alpha)
    wvr_scores = compute_wvoterank_score(G)
    
    final_scores = {}
    for v in G.nodes():
        final_scores[v] = rwr_scores[v] * wvr_scores[v]
        
    return final_scores