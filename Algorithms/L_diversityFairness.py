import networkx as nx
import pandas as pd


def compute_multi_L_diversity(G, communities, G_attribute,
                                  weight="weight", resolution=1):
    """
    Multi-group generalization of your original diversity modularity.

    Binary case (original code):
      For two groups (say 0 and 1), you computed for each community C:

        Q_C = inter_L(C) / (2m)
              - resolution * degree_R(C) * degree_B(C) * (1 / (2 m mInter)),

      where
        - inter_L(C) = number of cross-group edges inside C,
        - degree_R(C), degree_B(C) are cross-degree masses for the
          two groups inside C (from red_weight / blue_weight),
        - m = total number of edges,
        - mInter = total number of cross-group edges.

    Here we extend this to multiple groups {g} by summing the same
    type of term over all unordered group pairs (g, h), g < h.

    For each pair (g, h) and community C:
      - L_C^{(g,h)} = number (or weight) of edges inside C with
                      endpoints in groups g and h,
      - deg_C^{(g,h),g} = sum of cross-degrees (towards h) of group-g
                          nodes in C,
      - deg_C^{(g,h),h} = sum of cross-degrees (towards g) of group-h
                          nodes in C,
      - m_inter^{(g,h)} = total number (or weight) of (g,h) edges.

      Then

        Q_C^{(g,h)} =
            L_C^{(g,h)} / (2m)
            - resolution * deg_C^{(g,h),g} * deg_C^{(g,h),h}
                          / (2 m m_inter^{(g,h)}).

    The total diversity for community C is

        Q_C_div = sum_{g < h} Q_C^{(g,h)},

    and the overall diversity is Q_div = sum_C Q_C_div.

    When there are exactly two groups, this collapses to your original
    diversity definition.
    """

    if G.is_directed():
        raise ValueError("Diversity metric is implemented for undirected graphs only.")

    # Total edges m (same as in your original code)
    deg = dict(G.degree(weight=weight))
    deg_sum = sum(deg.values())
    m = deg_sum / 2.0
    if m == 0.0:
        # Edgeless graph
        return 0.0, [0.0 for _ in communities]

    # Distinct attribute values (groups)
    groups = sorted(set(G_attribute.values()))

    # All unordered group pairs (g, h), g < h
    pair_keys = []
    for i, g in enumerate(groups):
        for h in groups[i + 1:]:
            pair_keys.append((g, h))

    # For each pair (g,h):
    #   - cross_deg[(g,h)][u] = cross-degree of node u wrt edges between g and h
    #   - m_inter[(g,h)] = total weight of (g,h) edges in the graph
    cross_deg = {pair: {u: 0.0 for u in G.nodes()} for pair in pair_keys}
    m_inter = {pair: 0.0 for pair in pair_keys}

    for u, v, data in G.edges(data=True):
        wt = data.get(weight, 1.0)
        gu = G_attribute[u]
        gv = G_attribute[v]

        if gu == gv:
            continue  # intra-group edge → irrelevant for diversity

        # Unordered pair key (g_low, g_high)
        if gu < gv:
            pair = (gu, gv)
        else:
            pair = (gv, gu)

        # Global count of (g,h) cross edges
        m_inter[pair] += wt

        # Cross-degrees: each endpoint gets +wt for that pair
        cross_deg[pair][u] += wt
        cross_deg[pair][v] += wt

    def community_contribution(community):
        if not community:
            return 0.0

        comm = set(community)
        Q_C = 0.0

        # For this community, we need L_C^{(g,h)} for each pair (g,h)
        L_C_pair = {pair: 0.0 for pair in pair_keys}

        for u, v, data in G.edges(comm, data=True):
            if v not in comm:
                continue  # ensure internal
            wt = data.get(weight, 1.0)
            gu = G_attribute[u]
            gv = G_attribute[v]

            if gu == gv:
                continue  # only cross-group edges matter

            if gu < gv:
                pair = (gu, gv)
            else:
                pair = (gv, gu)

            L_C_pair[pair] += wt

        # Now compute the pairwise contributions and sum them
        for pair in pair_keys:
            m_gh = m_inter[pair]
            if m_gh == 0.0:
                continue  # no (g,h) edges in the whole graph

            g, h = pair

            # Cross-degree masses for groups g and h inside this community
            deg_C_g = sum(
                cross_deg[pair][u]
                for u in comm
                if G_attribute[u] == g
            )
            deg_C_h = sum(
                cross_deg[pair][u]
                for u in comm
                if G_attribute[u] == h
            )

            if deg_C_g == 0.0 or deg_C_h == 0.0:
                continue  # no cross-degree mass for this pair in C

            L_C_gh = L_C_pair[pair]

            # EXACT same structure as your binary code, but per (g,h):
            #   L_C^{(g,h)} / (2m) - resolution * deg_C_g * deg_C_h / (2 m m_gh)
            term = (L_C_gh / (2.0 * m)) \
                   - resolution * deg_C_g * deg_C_h / (2.0 * m * m_gh)

            Q_C += term

        return Q_C

    community_diversity_list = [community_contribution(c) for c in communities]
    diversity_modularity = sum(community_diversity_list)

    return diversity_modularity, community_diversity_list


def LdiversityMetric(G, communities, G_attribute, weight="weight", resolution=1):
    """
    Wrapper with the same interface as your original diversityMetric:

        diversityModularity, diversityModularityList = diversityMetric(...)

    but generalized to multiple groups.
    For |groups| = 2 it reduces exactly to the original diversity metric.
    """
    return compute_multi_L_diversity(
        G, communities, G_attribute, weight=weight, resolution=resolution
    )
