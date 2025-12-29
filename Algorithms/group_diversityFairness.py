import numpy as np
import networkx as nx


def groupDiversityMetric(G, communities, G_attribute, weight="weight", resolution=1.0):
    """
    Unlabeled group-vs-rest diversity modularity.

    For each group g and community C:
      D_g(C) = In_g(C)/(2m) - resolution * K_g->rest(C) * K_rest->g(C) / ( (2m)*(2m) )

    where:
      - In_g(C): internal cross edges inside C incident to group g (g vs not-g)
      - K_g->rest(C): sum_{u in C, a(u)=g} k_u^{not-g}   (global cross-degree to rest)
      - K_rest->g(C): sum_{v in C, a(v)!=g} k_v^{g}      (global cross-degree to g)

    Returns:
      per_group_D      : dict[g -> sum_C D_g(C)]
      per_group_D_list : dict[g -> list of D_g(C) per community]
    """

    if G.is_directed():
        raise ValueError("Implemented for undirected graphs only.")

    deg = dict(G.degree(weight=weight))
    deg_sum = float(sum(deg.values()))
    if deg_sum == 0.0:
        groups = sorted(set(G_attribute.values()), key=lambda x: str(x))
        return {g: 0.0 for g in groups}, {g: [0.0 for _ in communities] for g in groups}

    m = deg_sum / 2.0
    two_m = 2.0 * m
    denom = two_m * two_m  # (2m)^2

    groups = sorted(set(G_attribute.values()), key=lambda x: str(x))

    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # deg_to_group[g][u] = cross-degree of node u to group g (ONLY cross edges)
    deg_to_group = {g: np.zeros(n, dtype=np.float64) for g in groups}

    # cross_to_rest[u] = cross-degree of u to nodes not in its own group
    cross_to_rest = np.zeros(n, dtype=np.float64)

    # precompute cross-degree structures (global)
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight, 1.0))
        gu = G_attribute[u]
        gv = G_attribute[v]
        if gu == gv:
            continue

        iu, iv = idx[u], idx[v]

        # u has cross-edge to group gv; v has cross-edge to group gu
        deg_to_group[gv][iu] += w
        deg_to_group[gu][iv] += w

        cross_to_rest[iu] += w
        cross_to_rest[iv] += w

    per_group_D_list = {g: [] for g in groups}

    for community in communities:
        if not community:
            for g in groups:
                per_group_D_list[g].append(0.0)
            continue

        S = set(community)
        comm_nodes = list(S)
        comm_idx = np.array([idx[u] for u in comm_nodes], dtype=int)

        # K_rest->g(C): for each g, sum over nodes in C of (cross-degree to g)
        K_rest_to_g = {g: float(deg_to_group[g][comm_idx].sum()) for g in groups}

        # K_g->rest(C): sum over nodes in C that are in group g of (cross-degree to rest)
        K_g_to_rest = {g: 0.0 for g in groups}
        for u in comm_nodes:
            gu = G_attribute[u]
            K_g_to_rest[gu] += float(cross_to_rest[idx[u]])

        # In_g(C): internal cross edges inside community incident to g
        In_g = {g: 0.0 for g in groups}
        for u, v, w in G.edges(S, data=weight, default=1.0):
            if v not in S:
                continue
            gu = G_attribute[u]
            gv = G_attribute[v]
            if gu == gv:
                continue
            w = float(w)
            In_g[gu] += w
            In_g[gv] += w

        # D_g(C)
        for g in groups:
            contrib = (In_g[g] / two_m) - (resolution * (K_g_to_rest[g] * K_rest_to_g[g]) / denom)
            per_group_D_list[g].append(float(contrib))

    per_group_D = {g: float(sum(per_group_D_list[g])) for g in groups}
    return per_group_D, per_group_D_list


def group_diversityMetric(G, communities, G_attribute, weight="weight", resolution=1.0):
    """
    Scalar unlabeled group-diversity (same interface as your old diversityMetric):
      - returns (D, D_list) where D is the mean over groups (avoids double counting),
        and D_list is the per-community mean over groups.
      - for 2 groups: mean == each group's value == the binary formula.
    """
    per_group_D, per_group_D_list = groupDiversityMetric(G, communities, G_attribute, weight, resolution)
    groups = list(per_group_D.keys())
    if not groups:
        return 0.0, [0.0 for _ in communities]

    total_scalar = float(np.mean([per_group_D[g] for g in groups]))
    per_comm_scalar = []
    for i in range(len(communities)):
        per_comm_scalar.append(float(np.mean([per_group_D_list[g][i] for g in groups])))

    return total_scalar, per_comm_scalar
