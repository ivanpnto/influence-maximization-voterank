# -*- coding: utf-8 -*-

def ensure_weighted(G):
    """Asegura que cada arista tenga un atributo 'weight' (por defecto 1.0)."""
    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            data['weight'] = 1.0