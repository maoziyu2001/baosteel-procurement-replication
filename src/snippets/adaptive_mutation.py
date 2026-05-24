"""Adaptive mutation probability and mixed mutation operators."""


def adaptive_mutation_probability(population, child, k2=0.5, k4=0.5):
    f_ave = sum(ind.fitness for ind in population) / len(population)
    f_max = max(ind.fitness for ind in population)
    if child.fitness >= f_ave:
        pm = k2 * (f_max - child.fitness) / (f_max - f_ave + 1e-6)
    else:
        pm = k4
    return min(max(pm, 0.0), 1.0)


def mixed_mutation(individual, context, pm, RA):
    """Choose adaptive transfer or random flip according to RA."""
    import numpy as np

    if np.random.rand() >= pm:
        return "none"
    if np.random.rand() < RA:
        adaptive_supplier_transfer(individual, context)
        return "adaptive_transfer"
    random_bit_flip(individual, context)
    return "random_flip"


def random_bit_flip(individual, context):
    import numpy as np

    for i in context.I:
        for j in context.J:
            for tau in context.tau_list:
                if np.random.rand() < 0.15:
                    individual.yijt[i, j, context.t, tau] = 1 - individual.yijt[i, j, context.t, tau]
                if context.alpha[i][j] < context.beta[i]:
                    individual.yijt[i, j, context.t, tau] = 0


def adaptive_supplier_transfer(individual, context):
    import random

    i = random.choice(context.I)
    tau = random.choice(context.tau_list)
    current = [j for j in context.J if individual.yijt[i, j, context.t, tau] == 1]
    eligible = [j for j in context.J if context.alpha[i][j] >= context.beta[i]]
    if not eligible:
        return
    total_decisions = individual.yijt.sum()
    average = max(1, int(total_decisions / (len(context.tau_list) * len(context.I))))
    count = random.randint(1, min(max(average, 1), len(eligible)))
    selected = random.sample(eligible, count)
    for j in current:
        individual.yijt[i, j, context.t, tau] = 0
    for j in selected:
        individual.yijt[i, j, context.t, tau] = 1

