import numpy as np
import networkx as nx


def groupLDiversityMetric(G, communities, G_attribute, weight="weight", resolution=1.0):
    """
    Group-vs-rest labeled diversity modularity.

    Returns
    -------
    per_group_D : dict
        {g: D_L^g(S)} aggregated over communities.
    per_group_D_list : dict
        {g: [D_L^g(C_i) for each community C_i]} per-community contributions.
    """

    if G.is_directed():
        raise ValueError("groupDiversityMetric is implemented for undirected graphs only.")

    # total m (weighted)
    deg = dict(G.degree(weight=weight))
    deg_sum = float(sum(deg.values()))
    if deg_sum == 0.0:
        groups = sorted(set(G_attribute.values()))
        return {g: 0.0 for g in groups}, {g: [0.0 for _ in communities] for g in groups}

    m = deg_sum / 2.0  # total undirected edge weight
    groups = sorted(set(G_attribute.values()))

    # map nodes -> contiguous indices for arrays
    nodes = list(G.nodes())
    node_to_pos = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # deg_to_group[g][u] = (cross) degree of node u to group g (counts ONLY cross edges)
    deg_to_group = {g: np.zeros(n, dtype=np.float64) for g in groups}

    # cross_to_other[u] = total cross-degree of u to groups != a(u)
    cross_to_other = np.zeros(n, dtype=np.float64)

    # m_cross[g] = total cross edges incident to group g (i.e., m_{g,¬g})
    m_cross = {g: 0.0 for g in groups}

    # precompute cross-degrees + m_cross
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight, 1.0))
        gu = G_attribute[u]
        gv = G_attribute[v]
        if gu == gv:
            continue

        iu = node_to_pos[u]
        iv = node_to_pos[v]

        # u has 1 cross-edge to group gv, v has 1 cross-edge to group gu
        deg_to_group[gv][iu] += w
        deg_to_group[gu][iv] += w

        cross_to_other[iu] += w
        cross_to_other[iv] += w

        # each endpoint group counts this edge in its (g,¬g) total
        m_cross[gu] += w
        m_cross[gv] += w

    # outputs
    per_group_D_list = {g: [] for g in groups}

    # per-community computation
    for community in communities:
        if not community:
            for g in groups:
                per_group_D_list[g].append(0.0)
            continue

        S = set(community)
        idx = np.array([node_to_pos[u] for u in community], dtype=int)

        # K^{¬g -> g}(C) = sum_{v in C} k_v^g  (nodes in g contribute 0 anyway)
        K_rest_to_g = {g: float(deg_to_group[g][idx].sum()) for g in groups}

        # K^{g -> ¬g}(C) = sum_{u in C, a(u)=g} k_u^{¬g}
        K_g_to_rest = {g: 0.0 for g in groups}
        for u in community:
            gu = G_attribute[u]
            K_g_to_rest[gu] += float(cross_to_other[node_to_pos[u]])

        # In^{g,¬g}(C): internal cross-edges incident to each group g
        In_g = {g: 0.0 for g in groups}
        for u, v, w in G.edges(S, data=weight, default=1.0):
            if v not in S:
                continue
            gu = G_attribute[u]
            gv = G_attribute[v]
            if gu == gv:
                continue
            w = float(w)
            # this cross-edge counts for both endpoint groups under group-vs-rest
            In_g[gu] += w
            In_g[gv] += w

        # contributions
        for g in groups:
            if m_cross[g] == 0.0:
                per_group_D_list[g].append(0.0)
                continue

            contrib = (In_g[g] / (2.0 * m)) - (
                resolution * (K_g_to_rest[g] * K_rest_to_g[g]) / (2.0 * m * m_cross[g])
            )
            per_group_D_list[g].append(float(contrib))

    per_group_D = {g: float(sum(per_group_D_list[g])) for g in groups}
    return per_group_D, per_group_D_list
