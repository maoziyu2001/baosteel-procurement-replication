"""Three local-search neighborhoods used in the paper."""


def n1_single_inversion(best_solution, context, check_neighbor):
    """Remove one active base-supplier-period link."""
    neighbors = []
    for i in context.I:
        for j in context.J:
            for tau in context.tau_list:
                if best_solution.yijt[i, j, context.t, tau] == 1:
                    neighbor = deepcopy_solution(best_solution)
                    neighbor.yijt[i, j, context.t, tau] = 0
                    neighbor.decode[i, j, context.t, tau] = 0
                    if check_neighbor(neighbor):
                        neighbors.append(neighbor)
    return neighbors


def n2_single_base_consolidation(best_solution, context, check_neighbor):
    """Merge a smaller supplier order into another supplier at the same base."""
    neighbors = []
    for i in context.I:
        for tau in context.tau_list:
            for j1 in context.J:
                for j2 in context.J:
                    if j1 >= j2:
                        continue
                    if best_solution.yijt[i, j1, context.t, tau] and best_solution.yijt[i, j2, context.t, tau]:
                        iron1 = best_solution.decode[i, j1, context.t, tau] * context.z[j1]
                        iron2 = best_solution.decode[i, j2, context.t, tau] * context.z[j2]
                        remove_j, keep_j = (j1, j2) if iron1 <= iron2 else (j2, j1)
                        neighbor = deepcopy_solution(best_solution)
                        moved_iron = neighbor.decode[i, remove_j, context.t, tau] * context.z[remove_j]
                        neighbor.yijt[i, remove_j, context.t, tau] = 0
                        neighbor.decode[i, remove_j, context.t, tau] = 0
                        neighbor.decode[i, keep_j, context.t, tau] += moved_iron / context.z[keep_j]
                        if check_neighbor(neighbor):
                            neighbors.append(neighbor)
    return neighbors


def n3_cross_period_consolidation(best_solution, context, check_neighbor):
    """Move a later order to an earlier period for the same base-supplier pair."""
    neighbors = []
    for i in context.I:
        for j in context.J:
            for tau1 in context.tau_list[:-1]:
                for tau2 in context.tau_list[context.tau_list.index(tau1) + 1:]:
                    if best_solution.yijt[i, j, context.t, tau1] and best_solution.yijt[i, j, context.t, tau2]:
                        neighbor = deepcopy_solution(best_solution)
                        neighbor.yijt[i, j, context.t, tau2] = 0
                        neighbor.decode[i, j, context.t, tau1] += neighbor.decode[i, j, context.t, tau2]
                        neighbor.decode[i, j, context.t, tau2] = 0
                        if check_neighbor(neighbor):
                            neighbors.append(neighbor)
    return neighbors


def deepcopy_solution(solution):
    import copy

    return copy.deepcopy(solution)

