"""commit-vs-revise: the synthetic falsifier (docs/research_plan.md §7.1).

A streaming state-tracking task. Each slot holds a RUNNING SUM (mod n_vals) of the
values written to it by SET events. The model must:
  - REVISE a slot's value when later SET events arrive (state changes), and
  - PRESERVE committed state across NOISE (distractor) events that must NOT update it,
  - and emit the current value on QUERY.

The target is a running aggregate (not any single token), so attention cannot solve it
by copying one position — state must be maintained in the residual stream. This is the
task that should separate `ledger` from both `hc2` (symmetric, no protection) and
`delta_only` (erasable, no protection); and on which `suppress_only` should be <= vanilla.

Token layout (single flat vocab):
    0            PAD
    1 SET  2 NOISE  3 QUERY            (op tokens)
    4 .. 4+n_slots-1                   (slot tokens)
    4+n_slots .. 4+n_slots+n_vals-1    (value tokens)
Each event is a 3-token triple [op, slot, value]. For QUERY the value token is the
correct answer; we supervise predicting it from the [.., QUERY, slot] prefix.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch


class CommitReviseTask:
    def __init__(
        self,
        n_slots: int = 4,
        n_vals: int = 8,
        min_events: int = 6,
        max_events: int = 20,
        p_set: float = 0.45,
        p_noise: float = 0.30,
        p_query: float = 0.25,
    ):
        self.n_slots = n_slots
        self.n_vals = n_vals
        self.min_events = min_events
        self.max_events = max_events
        self.p_set = p_set
        self.p_noise = p_noise
        self.p_query = p_query
        # token ids
        self.PAD, self.SET, self.NOISE, self.QUERY = 0, 1, 2, 3
        self.slot0 = 4
        self.val0 = 4 + n_slots
        self.vocab_size = self.val0 + n_vals

    def sample(self, rng: random.Random) -> Tuple[List[int], List[bool]]:
        slots = [0] * self.n_slots
        ids: List[int] = []
        is_answer: List[bool] = []
        n_events = rng.randint(self.min_events, self.max_events)
        had_query = False
        for _ in range(n_events):
            r = rng.random()
            if r < self.p_set:
                s, v = rng.randrange(self.n_slots), rng.randrange(self.n_vals)
                slots[s] = (slots[s] + v) % self.n_vals
                ids += [self.SET, self.slot0 + s, self.val0 + v]
                is_answer += [False, False, False]
            elif r < self.p_set + self.p_noise:
                # distractor: looks like a write, references a slot, but must be ignored
                s, v = rng.randrange(self.n_slots), rng.randrange(self.n_vals)
                ids += [self.NOISE, self.slot0 + s, self.val0 + v]
                is_answer += [False, False, False]
            else:
                s = rng.randrange(self.n_slots)
                ids += [self.QUERY, self.slot0 + s, self.val0 + slots[s]]
                is_answer += [False, False, True]
                had_query = True
        if not had_query:  # guarantee at least one supervised position
            s = rng.randrange(self.n_slots)
            ids += [self.QUERY, self.slot0 + s, self.val0 + slots[s]]
            is_answer += [False, False, True]
        return ids, is_answer


def make_batch(task: CommitReviseTask, batch_size: int, block_size: int, rng: random.Random):
    """Returns (ids, targets) LongTensors of shape (batch_size, block_size).

    targets[t] = ids[t+1] only where t+1 is an answer position, else -100 (ignored).
    """
    ids_b, tgt_b = [], []
    for _ in range(batch_size):
        ids, is_answer = task.sample(rng)
        ids, is_answer = ids[:block_size], is_answer[:block_size]
        T = len(ids)
        tgt = [-100] * T
        for t in range(T - 1):
            if is_answer[t + 1]:
                tgt[t] = ids[t + 1]
        pad = block_size - T
        ids_b.append(ids + [task.PAD] * pad)
        tgt_b.append(tgt + [-100] * pad)
    return torch.tensor(ids_b, dtype=torch.long), torch.tensor(tgt_b, dtype=torch.long)
