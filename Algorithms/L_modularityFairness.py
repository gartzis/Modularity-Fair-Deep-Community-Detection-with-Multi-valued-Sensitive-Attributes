import networkx as nx


def compute_multi_group_L_modularity_fairness(
    G, communities, G_attribute, weight="weight", resolution=1
):
    """
    Multi-group extension of your binary L-modularity fairness.

    Setup (undirected only):

    - Let groups = distinct values in G_attribute.

    For each group g and community C:

      Observed term:
        E_gg(C)  = # edges (u,v) inside C with attr[u] = attr[v] = g
        E_gh(C)  = # edges (u,v) inside C with {attr[u], attr[v]} = {g, h}, g != h

        O_g(C) = ( E_gg(C) + 0.5 * sum_{h != g} E_gh(C) ) / m

        (In the binary case, this matches:
           modularityR = fair_L_cR / m + inter_L / (2m)
         with fair_L_cR = E_RR(C), inter_L = E_RB(C).)

      Expected term:

        We keep *two* null-model components for each group g:

        1) Within-group (g,g) null model
           - within_deg[g][u] = number of edges from u to group g (both endpoints g)
           - within_edge_count[g] = total # of edges with both endpoints in g

           degree_within_g(C) = sum_{u in C} within_deg[g][u]

           norm_within[g] = 1 / (within_edge_count[g] * 4m)  if within_edge_count[g] > 0
                            1                               otherwise

           Expected within term:
             E_within_g(C) = resolution * (degree_within_g(C)^2) * norm_within[g]

           This reproduces your  degree_R^2 * normR  and degree_B^2 * normB
           when there are just two groups.

        2) Cross-group (g,h) null model
           - cross_deg[g][h][u] = # neighbors of u in group h (u has attribute g)
           - cross_edge_count[(g,h)] = total # edges between groups g and h (unordered)

           deg_{g->h}(C) = sum_{u in C} cross_deg[g][h][u]
           deg_{h->g}(C) = sum_{u in C} cross_deg[h][g][u]

           norm_inter[(g,h)] = 1 / (cross_edge_count[(g,h)] * 2m) if cross_edge_count > 0
                               1                                  otherwise

           Expected cross term for group g:
             E_cross_g(C) = resolution * sum_{h != g} deg_{g->h}(C) * deg_{h->g}(C) * norm_inter[(g,h)]

           In the binary case, this collapses exactly to:
             red_node_blue_degree * blue_node_red_degree * normInter
             and the symmetric expression for blue.

      Group L-modularity in C:
        Q_g(C) = O_g(C) - (E_within_g(C) + E_cross_g(C))

    We also compute the usual structural modularity per community:
      Q_C = L_C / m - resolution * (deg_C^2) / (deg_sum^2)

    Returns
    -------
    unfairness_gap : float
        Global gap max_g Q_g - min_g Q_g (aggregated over all communities).
    per_group_Q : dict
        {g: Q_g} where Q_g = sum_C Q_g(C).
    unfairness_normalized : float
        Gap normalized by |Q_total|, where Q_total = sum_C Q_C (standard modularity).
    per_group_Q_list : dict
        {g: [Q_g(C) for C in communities]} – per-group, per-community L-modularity.
    community_modularity_list : list
        [Q_C for each community C] – standard structural modularity per community.
    """

    if G.is_directed():
        raise ValueError("Multi-group L-modularity fairness is implemented for undirected graphs only.")

    # Base degrees and m
    deg = dict(G.degree(weight=weight))
    deg_sum = sum(deg.values())
    if deg_sum == 0:
        # No edges: everything is trivially zero
        groups = sorted(set(G_attribute.values()))
        per_group_Q_list = {g: [0.0 for _ in communities] for g in groups}
        per_group_Q = {g: 0.0 for g in groups}
        return 0.0, per_group_Q, 0.0, per_group_Q_list, [0.0 for _ in communities]

    m = deg_sum / 2.0
    groups = sorted(set(G_attribute.values()))

    # --- Precompute within-group node degrees and edge counts, and cross-group degrees ---

    # within_deg[g][u] = number of edges from u to same-group neighbors (group g)
    within_deg = {g: {u: 0.0 for u in G.nodes()} for g in groups}
    within_edge_count = {g: 0.0 for g in groups}

    # cross_deg[g][h][u] = # neighbors of u (attr g) in group h
    cross_deg = {
        g: {h: {u: 0.0 for u in G.nodes()} for h in groups if h != g}
        for g in groups
    }
    # cross_edge_count[(g,h)] = # edges between g and h (unordered, g < h)
    cross_edge_count = {(g, h): 0.0 for g in groups for h in groups if g < h}

    for u, v, wt in G.edges(data=weight, default=1.0):
        gu = G_attribute[u]
        gv = G_attribute[v]
        if gu == gv:
            # within-group edge
            within_edge_count[gu] += wt
            within_deg[gu][u] += wt
            within_deg[gu][v] += wt
        else:
            # cross-group edge
            pair = (gu, gv) if gu < gv else (gv, gu)
            cross_edge_count[pair] += wt

            # directed-as-roles degrees (like red_node_blue_weight, blue_node_red_weight)
            cross_deg[gu][gv][u] += wt
            cross_deg[gv][gu][v] += wt

    # --- Null-model norms: match your binary code when |groups| = 2 ---

    norm_within = {}
    for g in groups:
        if within_edge_count[g] != 0:
            # matches normR = 1 / (deg_sumR * 4m) where deg_sumR is #within edges
            norm_within[g] = 1.0 / (within_edge_count[g] * 4.0 * m)
        else:
            norm_within[g] = 1.0

    norm_inter = {}
    for pair, cnt in cross_edge_count.items():
        if cnt != 0:
            # matches normInter = 1 / (mInter * 2m) in your code
            norm_inter[pair] = 1.0 / (cnt * 2.0 * m)
        else:
            norm_inter[pair] = 1.0

    # Global modularity norm (like norm = 1 / deg_sum**2)
    norm_global = 1.0 / (deg_sum ** 2)

    per_group_Q_list = {g: [] for g in groups}
    community_modularity_list = []

    # --- Per-community contributions ---

    for community in communities:
        if not community:
            community_modularity_list.append(0.0)
            for g in groups:
                per_group_Q_list[g].append(0.0)
            continue

        S = set(community)

        # Structural modularity pieces
        out_degree_sum = sum(deg[u] for u in S)
        L_c = 0.0

        # Internal edges per group and per pair (g,h)
        E_within_in_C = {g: 0.0 for g in groups}
        E_cross_in_C = {(g, h): 0.0 for g in groups for h in groups if g < h}

        for u, v, wt in G.edges(S, data=weight, default=1.0):
            if v not in S:
                continue
            L_c += wt
            gu = G_attribute[u]
            gv = G_attribute[v]
            if gu == gv:
                E_within_in_C[gu] += wt
            else:
                pair = (gu, gv) if gu < gv else (gv, gu)
                E_cross_in_C[pair] += wt

        in_degree_sum = out_degree_sum  # undirected
        Q_C = (L_c / m) - (resolution * out_degree_sum * in_degree_sum * norm_global)
        community_modularity_list.append(Q_C)

        # Group-specific L-modularity contributions
        for g in groups:
            # Observed part: within edges + half of all cross edges touching group g
            cross_sum_for_g = 0.0
            for h in groups:
                if h == g:
                    continue
                pair = (g, h) if g < h else (h, g)
                cross_sum_for_g += E_cross_in_C[pair]
            O_gC = (E_within_in_C[g] + 0.5 * cross_sum_for_g) / m

            # Expected within-group part
            deg_within_g_C = sum(within_deg[g][u] for u in S)
            exp_within_g = resolution * (deg_within_g_C ** 2) * norm_within[g]

            # Expected cross-group part
            exp_cross_g = 0.0
            for h in groups:
                if h == g:
                    continue
                pair = (g, h) if g < h else (h, g)
                deg_g_to_h_C = sum(cross_deg[g][h][u] for u in S)
                deg_h_to_g_C = sum(cross_deg[h][g][u] for u in S)
                exp_cross_g += resolution * deg_g_to_h_C * deg_h_to_g_C * norm_inter[pair]

            Q_gC = O_gC - (exp_within_g + exp_cross_g)
            per_group_Q_list[g].append(Q_gC)

    per_group_Q = {g: sum(per_group_Q_list[g]) for g in groups}

    if len(groups) >= 2:
        vals = list(per_group_Q.values())
        Q_min = min(vals)
        Q_max = max(vals)
        unfairness_gap = Q_max - Q_min
        Q_total = sum(community_modularity_list)
        unfairness_normalized = unfairness_gap / abs(Q_total) if Q_total != 0 else 0.0
    else:
        unfairness_gap = 0.0
        unfairness_normalized = 0.0

    return (
        unfairness_gap,
        per_group_Q,
        unfairness_normalized,
        per_group_Q_list,
        community_modularity_list,
    )


def LModularityFairnessMetric(
    G, communities, G_attribute, weight="weight", resolution=1
):
    """
    Multi-group L-modularity fairness metric.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph.
    communities : list of lists
        Partition of nodes (each community is a list of nodes).
    G_attribute : dict
        {node: group_label} for all nodes in G.
    weight : str
        Edge weight attribute (default 'weight').
    resolution : float
        Resolution parameter.

    Returns
    -------
    unfairness_gap : float
        Global group L-modularity gap  max_g Q_g - min_g Q_g.
    unfairness_per_community : list of float
        For each community C: max_g Q_g(C) - min_g Q_g(C).
    unfairness_normalized : float
        Global normalized gap (gap / |Q_total|), where Q_total is structural modularity.
    per_group_Q_list : dict
        {g: [Q_g(C) for each C]} – group L-modularity per community.
    per_group_Q : dict
        {g: Q_g} – aggregated group L-modularity over all communities.
    """
    (
        unfairness_gap,
        per_group_Q,
        unfairness_normalized,
        per_group_Q_list,
        community_modularity_list,
    ) = compute_multi_group_L_modularity_fairness(
        G, communities, G_attribute, weight=weight, resolution=resolution
    )

    groups = sorted(set(G_attribute.values()))
    unfairness_per_community = []
    for i in range(len(community_modularity_list)):
        vals_i = [per_group_Q_list[g][i] for g in groups]
        gap_i = max(vals_i) - min(vals_i) if len(vals_i) >= 2 else 0.0
        unfairness_per_community.append(gap_i)

    return (
        unfairness_gap,
        unfairness_per_community,
        unfairness_normalized,
        per_group_Q_list,
        per_group_Q,
    )
