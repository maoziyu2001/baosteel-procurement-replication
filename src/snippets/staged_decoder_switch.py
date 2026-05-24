"""Staged decoder switching used in GA-hybrid."""


def evaluate_with_staged_decoder(individual, gen, K_explore, context):
    if gen < K_explore:
        decoder = "greedy"
        feasible, cost, x, lower_bound = solve_procument_plan_greedy(context, individual.yijt)
    else:
        decoder = "solver"
        feasible, cost, x, lower_bound = solve_procument_plan_mip(context, individual.yijt)

    individual.feasible = bool(feasible)
    individual.cost = cost
    individual.decode = x
    individual.lowerbound = lower_bound
    return decoder, individual


def solve_procument_plan_greedy(context, yijt):
    raise NotImplementedError("See src/ga_hybrid/decoder_greedy.py")


def solve_procument_plan_mip(context, yijt):
    raise NotImplementedError("See src/ga_hybrid/decoder_mip.py")

