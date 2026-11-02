# -*- coding: utf-8 -*-

def ensure_weighted(G):
    """Ensure every edge has a 'weight' attribute (default 1.0)."""
    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            data['weight'] = 1.0