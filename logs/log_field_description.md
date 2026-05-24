# Log field description

- `event`: event type, such as `RUN_START`, `DECODE_SWITCH`, `LOCAL_SEARCH_START`.
- `gen`: GA generation.
- `phase`: `exploration` or `exploitation`.
- `decoder`: `greedy` or `solver`.
- `candidate_neighbors`: number of generated neighbors for a local-search neighborhood.
- `feasible_neighbors`: number of feasible neighbors after constraint checking.
- `improvement`: accepted reduction in objective value.
- `improvement_pct`: relative objective improvement.
- `best_cost`: incumbent best objective value.
- `lower_bound`: reference lower bound when available.
- `gap_pct`: gap computed from `best_cost` and `lower_bound`.

