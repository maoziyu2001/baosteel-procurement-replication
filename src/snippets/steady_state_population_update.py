"""Elite-preserving steady-state population update."""


def steady_state_update(population, child1, child2, is_duplicate):
    if is_duplicate(child1, population) or is_duplicate(child2, population):
        return population, False
    population.sort(key=lambda ind: ind.cost)
    worst_two = population[-2:]
    if child1.cost < worst_two[0].cost and child2.cost < worst_two[0].cost:
        population = population[:-2] + [child1, child2]
        return population, True
    return population, False

