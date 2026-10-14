# -*- coding: utf-8 -*-
import random

def independent_cascade(G, seeds, p_attr='prob'):
    """Simula el modelo Independent Cascade (IC) de difusión de influencia."""
    active = set(seeds)
    newly_active = set(seeds)

    while newly_active:
        next_active = set()
        for u in newly_active:
            for v in G.neighbors(u):
                if v not in active:
                    if random.random() < G[u][v].get(p_attr, 0.1):
                        next_active.add(v)
        newly_active = next_active
        active |= newly_active

    return active

def average_spread(G, seeds, mc=200, p_attr='prob'):
    """Calcula el promedio de influencia alcanzada tras múltiples simulaciones Monte Carlo."""
    total = 0
    for _ in range(mc):
        total += len(independent_cascade(G, seeds, p_attr))
    return total / mc