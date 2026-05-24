"""Fixed-point supplier-block exchange crossover.

Each supplier index is treated as a fixed exchange point. For a supplier j, the
complete base-period purchase relation block y[:, j, :, :] is inherited from
one of the two parents. This preserves supplier-level procurement patterns.
"""


def fixed_point_exchange_crossover(parent1, parent2, child1, child2, suppliers, pc):
    pc = min(max(pc, 0.0), 1.0)
    for j in suppliers:
        if random_uniform() < pc:
            child1.yijt[:, j, :, :] = parent1.yijt[:, j, :, :].copy()
            child2.yijt[:, j, :, :] = parent2.yijt[:, j, :, :].copy()
        else:
            child1.yijt[:, j, :, :] = parent2.yijt[:, j, :, :].copy()
            child2.yijt[:, j, :, :] = parent1.yijt[:, j, :, :].copy()
    return child1, child2


def random_uniform():
    import numpy as np

    return np.random.rand()

