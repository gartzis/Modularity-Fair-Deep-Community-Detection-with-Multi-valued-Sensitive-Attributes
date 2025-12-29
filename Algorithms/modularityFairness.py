import networkx as nx
import pandas as pd


def compute_multi_group_modularity_fairness(G, communities, G_attribute,
                                            weight="weight", resolution=1):
    """
    Exact multi-group generalization of the original red/blue modularity fairness.

    For each community C and group g:
      - L_{C,g} = internal group volume:
          2 * (# edges (u,v) in C with attr(u)=g and attr(v)=g)
        + 1 * (# edges (u,v) in C with exactly one endpoint in group g).
      - deg_C  = sum of degrees of nodes in C (using 'weight').
      - deg_{C,g} = sum of degrees of nodes in C whose attribute == g.

    Then:
      Q_{C,g} = L_{C,g} / (2m) - resolution * deg_C * deg_{C,g} * norm,

    where:
      m = total number of edges (undirected: m = sum(deg)/2),
      norm = 1 / (sum(deg)^2) for undirected graphs.

    This matches the original red/blue definition when there are exactly 2 groups.
    """

    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1.0 / (m ** 2) if m > 0 else 0.0
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2.0
        norm = 1.0 / (deg_sum ** 2) if deg_sum > 0 else 0.0

    groups = sorted(set(G_attribute.values()))

    community_modularity_list = []
    per_group_Q_list = {g: [] for g in groups}

    def community_contribution(community):
        if not community:
            return 0.0, {g: 0.0 for g in groups}

        comm = set(community)

        # Community degree masses
        deg_C = sum(out_degree[u] for u in comm)
        deg_C_in = sum(in_degree[u] for u in comm) if directed else deg_C

        # Internal edges and group internal volumes
        L_C = 0.0
        group_internal = {g: 0.0 for g in groups}

        for u, v, wt in G.edges(comm, data=weight, default=1):
            if v not in comm:
                continue  # ensure internal

            L_C += wt
            gu = G_attribute[u]
            gv = G_attribute[v]

            if gu == gv:
                # same-group edge → contributes 2 * wt to that group's internal volume
                group_internal[gu] += 2.0 * wt
            else:
                # cross-group edge → 1 * wt to each group's internal volume
                group_internal[gu] += wt
                group_internal[gv] += wt

        # Standard modularity contribution of C
        if m > 0:
            Q_C = (L_C / m) - resolution * deg_C * deg_C_in * norm
        else:
            Q_C = 0.0

        # Group degree masses (global degrees, as in original binary code)
        group_deg = {g: 0.0 for g in groups}
        for u in comm:
            g = G_attribute[u]
            group_deg[g] += out_degree[u]

        # Per-group modularity contributions Q_{C,g}
        group_mod = {}
        for g in groups:
            deg_Cg = group_deg[g]
            if m > 0 and deg_Cg > 0:
                group_mod[g] = (group_internal[g] / (2.0 * m)) - \
                               resolution * deg_C * deg_Cg * norm
            else:
                group_mod[g] = 0.0

        return Q_C, group_mod

    # Compute per-community & per-group
    for community in communities:
        Q_C, group_mod = community_contribution(community)
        community_modularity_list.append(Q_C)
        for g in groups:
            per_group_Q_list[g].append(group_mod[g])

    # Aggregate group modularities
    per_group_Q = {g: sum(per_group_Q_list[g]) for g in groups}
    Q_total = sum(community_modularity_list)

    # Global gap + normalized gap
    if len(groups) >= 2:
        vals = list(per_group_Q.values())
        Q_min = min(vals)
        Q_max = max(vals)
        unfairness_gap = Q_max - Q_min
        unfairness_normalized = unfairness_gap / abs(Q_total) if Q_total != 0 else 0.0
    else:
        unfairness_gap = 0.0
        unfairness_normalized = 0.0

    return (unfairness_gap,
            per_group_Q,
            unfairness_normalized,
            per_group_Q_list,
            community_modularity_list)



def multiModularityFairnessMetric(G, communities, G_attribute,
                                  weight="weight", resolution=1):
    """
    Parameters
    ----------
    G : networkx.Graph
        Graph with any number of attribute values in G_attribute.
    communities : list of lists
        Partition of nodes.
    G_attribute : dict
        {node: attribute_value} for all nodes in G.
    weight, resolution : as before.

    Returns
    -------
    unfairness_gap : float
        Global group modularity gap max_g Q_g - min_g Q_g.
    unfairness_per_community : list of float
        Per-community group modularity gap (max_g Q_{C,g} - min_g Q_{C,g}).
    unfairness_normalized : float
        Global normalized gap (see compute_multi_group_modularity_fairness).
    per_group_Q_list : dict
        {g: [Q_{C,g} for C in communities]} – per-group modularity per community.
    per_group_Q : dict
        {g: Q_g} – aggregated group modularity over the whole partition.
    """
    (unfairness_gap,
     per_group_Q,
     unfairness_normalized,
     per_group_Q_list,
     community_modularity_list) = compute_multi_group_modularity_fairness(
        G, communities, G_attribute, weight=weight, resolution=resolution
    )

    # Optionally: per-community unfairness as max_g Q_{C,g} - min_g Q_{C,g}
    groups = sorted(set(G_attribute.values()))
    unfairness_per_community = []
    for i in range(len(community_modularity_list)):
        vals_i = [per_group_Q_list[g][i] for g in groups]
        gap_i = max(vals_i) - min(vals_i) if len(vals_i) >= 2 else 0.0
        unfairness_per_community.append(gap_i)

    return (unfairness_gap,
            unfairness_per_community,
            unfairness_normalized,
            per_group_Q_list,
            per_group_Q)
