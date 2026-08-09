"""Task 1.6: the reproduction benchmark.

Two properties, per the plan. Reachability says the instance is *valid* -- a
maze with a state that cannot reach the goal is not a harder problem, it is a
different one, with an infinite optimal cost that would quietly poison every
error number downstream. Seed-determinism is what "fixed seeds" means; without
it the 20 paired seeds at 3.5 are not paired.

What is deliberately not tested: row sums and dtypes (the builder's job, and it
rejects violations at construction), and the unique-path property, which is a
design choice confirmed structurally at build time -- a perfect maze is a
spanning tree, and `edges == cells - 1` plus connectivity is exactly that.

Reachability is checked on the MDP's *support graph*, not on the carve, because
the support graph is what value iteration actually traverses. A CSR bug that
lost successors would leave the carve intact and still strand states here.
"""

import numpy as np
import pytest

from mdpagg.maze import GOAL, make_standard_maze

P = 0.9
SEED = 0
FIELDS = ("sa_begin", "succ_begin", "succ_state", "succ_prob", "cost")


def _states_that_reach(m, target: int) -> np.ndarray:
    """Reverse BFS over every transition with positive probability."""
    state_of_pair = np.repeat(np.arange(m.num_states), np.diff(m.sa_begin))
    src = np.repeat(state_of_pair, np.diff(m.succ_begin))
    dst = np.asarray(m.succ_state)

    keep = np.asarray(m.succ_prob) > 0.0
    src, dst = src[keep], dst[keep]

    order = np.argsort(dst, kind="stable")
    pred = src[order]
    begin = np.searchsorted(dst[order], np.arange(m.num_states + 1))

    seen = np.zeros(m.num_states, dtype=bool)
    seen[target] = True
    frontier = np.array([target])

    while frontier.size:
        counts = begin[frontier + 1] - begin[frontier]
        offset = np.arange(int(counts.sum())) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        candidates = pred[np.repeat(begin[frontier], counts) + offset]
        frontier = np.unique(candidates[~seen[candidates]])
        seen[frontier] = True

    return seen


@pytest.mark.parametrize("dims", [(20, 20), (100, 100)], ids=str)
def test_goal_is_reachable_from_every_state(dims):
    m = make_standard_maze(dims, P, SEED)

    assert _states_that_reach(m, GOAL).all()


@pytest.mark.slow
def test_goal_is_reachable_at_benchmark_scale():
    """500x500, the size 3.5 reports against. Slow: the carve is ~6 s."""
    m = make_standard_maze((500, 500), P, SEED)

    assert _states_that_reach(m, GOAL).all()


def test_same_seed_gives_an_identical_maze():
    a = make_standard_maze((30, 30), P, 11)
    b = make_standard_maze((30, 30), P, 11)

    for field in FIELDS:
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_different_seeds_give_different_mazes():
    """Guards the failure the test above cannot see: a generator that ignores
    the seed entirely reproduces perfectly."""
    a = make_standard_maze((30, 30), P, 11)
    c = make_standard_maze((30, 30), P, 12)

    assert not np.array_equal(a.succ_state, c.succ_state)


def test_goal_absorbs_and_entering_it_is_rewarded():
    """Beyond the plan's two tests, and here because the goal's cost is a
    reading of the paper rather than something the plan specified: the terminal
    state pays a *reward* on entry, not zero cost. If that silently reverted,
    V* would change everywhere and nothing else in the suite would notice.
    """
    m = make_standard_maze((4, 4), P, SEED)

    goal_pair = m.pair_index(GOAL, 0)
    assert m.num_actions(GOAL) == 1
    assert m.successors(goal_pair).tolist() == [GOAL]
    assert m.cost[goal_pair] == 0.0

    # Some action somewhere must reach the goal, and it must be cheaper than a
    # plain move.
    entering = [
        p
        for p in range(m.num_pairs)
        if p != goal_pair and GOAL in m.successors(p).tolist()
    ]
    assert entering
    assert min(m.cost[p] for p in entering) < 0.0
