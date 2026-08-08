"""
    V*(s) = min_a [ c(s, a) + gamma * sum_s' p(s' | s, a) V*(s') ]
    minimizing total cost
"""

from typing import NamedTuple

import numpy as np

from mdpagg.mdp import TabularMdp, build
from mdpagg.types import INDEX, VALUE, ValueArray


class Fixture(NamedTuple):
## known optimial value function

    name: str
    mdp: TabularMdp
    gamma: float
    v_star: ValueArray


def two_state_chain() -> Fixture:
    """Two states, the discount/gamma = 0.9. State 0 has two possible actions, state 1 has one.

    (s, a) -> cost, successors

        (0, 0) -> 1.0, {0: 0.5, 1: 0.5}    
        (0, 1) -> 3.0, {1: 1.0}
        (1, 0) -> 2.0, {0: 1.0}

        V0 = 1 + 0.9 (0.5 V0 + 0.5 V1)
        V1 = 2 + 0.9 V0

        V0 = 1 + 0.45 V0 + 0.45 (2 + 0.9 V0)
           = 1 + 0.45 V0 + 0.9 + 0.405 V0
           = 1.9 + 0.855 V0
        (1 - 0.855) V0 = 1.9
        V0 = 1.9 / 0.145 = 1900 / 145 = 380 / 29 = 13.1034482758...
        V1 = 2 + 0.9 (380 / 29) = 2 + 342 / 29 = 400 / 29 = 13.7931034482...

        3 + 0.9 (400 / 29) = 3 + 360 / 29 = 447 / 29 = 15.4137931034...
    
    V1 is more optimal node to begin
    """
    # pair 0 = (0, 0), pair 1 = (0, 1), pair 2 = (1, 0)
    mdp = build(
        sa_begin=np.array([0, 2, 3], dtype=INDEX),
        succ_begin=np.array([0, 2, 3, 4], dtype=INDEX),
        succ_state=np.array([0, 1, 1, 0], dtype=INDEX),
        succ_prob=np.array([0.5, 0.5, 1.0, 1.0], dtype=VALUE),
        cost=np.array([1.0, 3.0, 2.0], dtype=VALUE),
    )
    v_star = np.array([380.0 / 29.0, 400.0 / 29.0], dtype=VALUE)
    return Fixture("two_state_chain", mdp, 0.9, v_star)


ROWS, COLS = 3, 3
GOAL = 8
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1)) ## up down left and right


def tiny_gridworld() -> Fixture:
    """A 3x3 deterministic grid, gamma = 0.9, unit cost per move.

    States are row-major:

        0 1 2
        3 4 5
        6 7 8


        V*(s) = 1 + g + g^2 + ... + g^(d-1) = (1 - g^d) / (1 - g),  g = 0.9

        d = 0:  0
        d = 1:  1
        d = 2:  1 + 0.9              = 1.9
        d = 3:  1 + 0.9 + 0.81       = 2.71
        d = 4:  1 + 0.9 + 0.81 + 0.729 = 3.439

        d           V*
        4 3 2     3.439  2.71  1.9
        3 2 1     2.71   1.9   1.0
        2 1 0     1.9    1.0   0.0
    """
    sa_begin = [0]
    succ_begin = [0]
    succ_state: list[int] = []
    succ_prob: list[float] = []
    cost: list[float] = []

    for s in range(ROWS * COLS):
        if s == GOAL:
            succ_state.append(GOAL)
            succ_prob.append(1.0)
            cost.append(0.0)
            succ_begin.append(len(succ_state))
        else:
            r, c = divmod(s, COLS)
            for dr, dc in MOVES:
                r2 = min(max(r + dr, 0), ROWS - 1)
                c2 = min(max(c + dc, 0), COLS - 1)
                succ_state.append(r2 * COLS + c2)
                succ_prob.append(1.0)
                cost.append(1.0)
                succ_begin.append(len(succ_state))
        sa_begin.append(len(cost))

    mdp = build(
        sa_begin=np.array(sa_begin, dtype=INDEX),
        succ_begin=np.array(succ_begin, dtype=INDEX),
        succ_state=np.array(succ_state, dtype=INDEX),
        succ_prob=np.array(succ_prob, dtype=VALUE),
        cost=np.array(cost, dtype=VALUE),
    )
    v_star = np.array(
        [
            3.439, 2.71, 1.9,
            2.71,  1.9,  1.0,
            1.9,   1.0,  0.0,
        ],
        dtype=VALUE,
    )
    return Fixture("tiny_gridworld", mdp, 0.9, v_star)


ALL_FIXTURES = (two_state_chain, tiny_gridworld)
