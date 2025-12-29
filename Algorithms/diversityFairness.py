import networkx as nx


def compute_multi_pair_diversity(G, communities, G_attribute, weight="weight", resolution=1.0):
    """
    Unlabeled pairwise diversity modularity (multi-group).

    For each unordered pair (g,h), g<h, in community C:
      D_{g,h}(C) = L_{g,h}(C)/(2m) - resolution * K_{g->h}(C) * K_{h->g}(C) / ( (2m)*(2m) )

    Sum over pairs and communities.

    Returns:
      total_diversity, per_community_list
    """

    if G.is_directed():
        raise ValueError("Implemented for undirected graphs only.")

    deg = dict(G.degree(weight=weight))
    deg_sum = float(sum(deg.values()))
    if deg_sum == 0.0:
        return 0.0, [0.0 for _ in communities]

    m = deg_sum / 2.0
    two_m = 2.0 * m
    denom = two_m * two_m  # (2m)^2

    groups = sorted(set(G_attribute.values()))
    pair_keys = [(g, h) for i, g in enumerate(groups) for h in groups[i + 1:]]

    # cross_deg[(g,h)][u] = cross-degree of node u wrt edges between g and h (global)
    cross_deg = {pair: {u: 0.0 for u in G.nodes()} for pair in pair_keys}

    for u, v, data in G.edges(data=True):
        wt = float(data.get(weight, 1.0))
        gu = G_attribute[u]
        gv = G_attribute[v]
        if gu == gv:
            continue
        pair = (gu, gv) if gu < gv else (gv, gu)
        cross_deg[pair][u] += wt
        cross_deg[pair][v] += wt

    def community_contribution(community):
        if not community:
            return 0.0

        comm = set(community)

        # L_{g,h}(C): internal cross edges inside C for each pair
        L_C_pair = {pair: 0.0 for pair in pair_keys}
        for u, v, data in G.edges(comm, data=True):
            if v not in comm:
                continue
            wt = float(data.get(weight, 1.0))
            gu = G_attribute[u]
            gv = G_attribute[v]
            if gu == gv:
                continue
            pair = (gu, gv) if gu < gv else (gv, gu)
            L_C_pair[pair] += wt

        Q_C = 0.0
        for (g, h) in pair_keys:
            # K_{g->h}(C) and K_{h->g}(C) from global cross degrees
            K_g_to_h = sum(cross_deg[(g, h)][u] for u in comm if G_attribute[u] == g)
            K_h_to_g = sum(cross_deg[(g, h)][u] for u in comm if G_attribute[u] == h)

            term = (L_C_pair[(g, h)] / two_m) - (resolution * K_g_to_h * K_h_to_g / denom)
            Q_C += term

        return Q_C

    per_comm = [community_contribution(c) for c in communities]
    return float(sum(per_comm)), per_comm


def pair_DiversityFairnessMetric(G, communities, G_attribute, weight="weight", resolution=1.0):
    """
    Same interface: returns (diversity_total, diversity_list).
    This is the unlabeled (no m_{g,h}) pairwise version.
    """
    return compute_multi_pair_diversity(G, communities, G_attribute, weight, resolution)
