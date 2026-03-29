# sat_io.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable
import random
import pathlib

Lit = int           # positive i means x_i, negative i means ¬x_i
Clause = List[Lit]  # disjunction of literals
CNF = List[Clause]  # conjunction of clauses
Assignment = Dict[int, bool]  # variable -> True/False

@dataclass
class SatInstance:
    name: str
    nvars: int
    clauses: CNF
    known_model: Optional[Assignment] = None

# ---------------------------
# DIMACS CNF I/O
# ---------------------------

def parse_dimacs_cnf(path: str | pathlib.Path) -> Tuple[int, CNF]:
    """
    Parse a DIMACS CNF file. Returns (nvars, clauses).
    Ignores comments and supports multi-line 'v' model lines if present (ignored here).
    """
    nvars = 0
    clauses: CNF = []
    cur_clause: List[int] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('c') or s.startswith('v') or s.startswith('s') or s.startswith('%'):
                continue
            if s.startswith('p'):
                # e.g., "p cnf 20 91"
                parts = s.split()
                if len(parts) >= 4 and parts[1].lower() == 'cnf':
                    nvars = int(parts[2])
                continue
            # data line(s)
            for tok in s.split():
                lit = int(tok)
                if lit == 0:
                    if cur_clause:
                        clauses.append(cur_clause)
                        cur_clause = []
                else:
                    cur_clause.append(lit)
    if cur_clause:
        # some files omit trailing 0 on the last line; close last clause if present
        clauses.append(cur_clause)
    # ensure nvars covers all literals
    max_var = max((abs(l) for cl in clauses for l in cl), default=0)
    nvars = max(nvars, max_var)
    return nvars, clauses


def write_dimacs_cnf(path: str | pathlib.Path, nvars: int, clauses: CNF) -> None:
    """Write CNF in DIMACS format."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {nvars} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(str(l) for l in cl) + " 0\n")


def parse_dimacs_model(text_or_path: str) -> Assignment:
    """
    Parse a SAT solver model from DIMACS-style output:
      - lines starting with 'v' containing literals, possibly split across lines
      - or a single space-separated line of ints ending with 0
    Returns dict var->bool.
    """
    if "\n" in text_or_path or " " in text_or_path:
        text = text_or_path
    else:
        text = pathlib.Path(text_or_path).read_text(encoding="utf-8", errors="ignore")

    lits: List[int] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('c') or s.startswith('s'):
            continue
        if s.startswith('v'):
            s = s[1:].strip()
        for tok in s.split():
            val = int(tok)
            if val == 0:
                continue
            lits.append(val)

    model: Assignment = {}
    for lit in lits:
        v = abs(lit)
        model[v] = (lit > 0)
    return model

# ---------------------------
# Utilities
# ---------------------------

def evaluate_clause(clause: Clause, model: Assignment) -> bool:
    """True if any literal in the clause is satisfied under the model."""
    for lit in clause:
        v = abs(lit)
        val = model.get(v, None)
        if val is None:
            # unassigned variable -> clause may still become True; treat as False here
            continue
        if (lit > 0 and val) or (lit < 0 and not val):
            return True
    return False

def evaluate_cnf(nvars: int, clauses: CNF, model: Assignment) -> bool:
    """True iff all clauses are satisfied under the (possibly partial) model."""
    return all(evaluate_clause(cl, model) for cl in clauses)

def model_to_dimacs_vector(nvars: int, model: Assignment) -> List[int]:
    """Return a compact DIMACS vector [±1, ±2, ..., ±n] representing the model (unassigned -> positive by default)."""
    out = []
    for v in range(1, nvars + 1):
        val = model.get(v, True)
        out.append(v if val else -v)
    return out

# ---------------------------
# Tiny built-in benchmarks with known solutions
# ---------------------------

# Each instance lists its satisfying assignment for quick testing.
TINY_BENCHMARKS: Dict[str, SatInstance] = {}

# 1) Simple 3-CNF, 3 vars
TINY_BENCHMARKS["toy3"] = SatInstance(
    name="toy3",
    nvars=3,
    clauses=[
        [1, -2, 3],
        [-1, 2],
        [3],
    ],
    known_model={1: True, 2: True, 3: True},
)

# 2) Pigeonhole-like tiny UNSAT (2 holes, 3 pigeons is UNSAT), compact encoding
# Variables x_{p,h} -> flatten as: x11=1,x12=2,x21=3,x22=4,x31=5,x32=6
def _phl_3_into_2() -> SatInstance:
    nvars = 6
    # each pigeon in some hole
    clauses = [
        [1, 2], [3, 4], [5, 6],
        # no hole has two pigeons
        [-1, -3], [-1, -5], [-3, -5],  # hole1 pairs
        [-2, -4], [-2, -6], [-4, -6],  # hole2 pairs
    ]
    return SatInstance("pigeonhole_3_2_unsat", nvars, clauses, known_model=None)

TINY_BENCHMARKS["pigeonhole_3_2_unsat"] = _phl_3_into_2()

# 3) Small SAT with a unique model (over 4 vars)
TINY_BENCHMARKS["unique4"] = SatInstance(
    name="unique4",
    nvars=4,
    clauses=[
        [1],               # x1 = True
        [2, 3],            # x2 or x3
        [-2, -3],          # not both (so exactly one of x2,x3)
        [4, -2],           # x4 or ¬x2
        [4, -3],           # x4 or ¬x3 -> forces x4 = True
    ],
    known_model={1: True, 2: True, 3: False, 4: True},
)

# ---------------------------
# Planted k-SAT generator (known solution)
# ---------------------------

def planted_k_sat(nvars: int, mclauses: int, k: int = 3, seed: Optional[int] = None
                  ) -> Tuple[CNF, Assignment]:
    """
    Generate a random k-SAT instance with a *planted* satisfying assignment.
    Steps:
      1) Pick a random model A.
      2) Repeatedly sample a k-clause uniformly from variables and negations
         conditioned on being satisfied by A.
    Returns (clauses, planted_assignment).
    """
    rnd = random.Random(seed)
    assignment: Assignment = {v: rnd.choice([True, False]) for v in range(1, nvars + 1)}
    clauses: CNF = []
    for _ in range(mclauses):
        # choose k distinct variables
        vars_k = rnd.sample(range(1, nvars + 1), k if k <= nvars else nvars)
        lits = []
        # Ensure clause satisfied by A: pick negation pattern that leaves at least one lit true
        # Strategy: make exactly one literal align with the assignment, others random
        true_pos = rnd.randrange(len(vars_k))
        for i, v in enumerate(vars_k):
            val = assignment[v]
            if i == true_pos:
                lit = v if val else -v
            else:
                # random sign (could still be satisfied, but at least one is guaranteed)
                lit = v if rnd.choice([True, False]) else -v
            lits.append(lit)
        # Deduplicate and avoid tautologies; if tautological, resample small tweak
        if any(x == -y for x in lits for y in lits):
            # try again with deterministic signs to avoid tautology
            lits = []
            for i, v in enumerate(vars_k):
                val = assignment[v]
                lit = v if (val if i == 0 else not val) else -(v)
                lits.append(lit)
        clauses.append(lits)
    return clauses, assignment

# ---------------------------
# Small demo / sanity check
# ---------------------------

def _demo():
    for name, inst in TINY_BENCHMARKS.items():
        ok = (inst.known_model is None) or evaluate_cnf(inst.nvars, inst.clauses, inst.known_model)
        print(f"{name:>24}: clauses={len(inst.clauses):2d}, "
              f"nvars={inst.nvars:2d}, known_model_ok={ok}")

if __name__ == "__main__":
    _demo()
