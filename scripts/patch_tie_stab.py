#!/usr/bin/env python3
"""Patch an existing cohesive-fairing INP with the two convergence fixes:

  (1) Add `position tolerance=<tol>` to the 4 sandwich *Tie cards (COMPUTED was
      silently dropping cohesive-layer secondary nodes -> zero pivots).
  (2) Insert `*Damage Stabilization` (viscous regularization) after every
      `*Damage Evolution` so the cohesive softening branch does not stall Static.

Frame ties are left untouched (they were tying cleanly with COMPUTED).
This mirrors the source fix in src/generate_cohesive_fairing.py so the patched
INP and a future CAE regeneration are physically identical.
"""
import sys

SANDWICH_TIES = (
    'Tie-AdhInner-Core',
    'Tie-AdhOuter-OuterSkin',
    'Tie-Core-AdhOuter',
    'Tie-InnerSkin-AdhInner',
)
TIE_TOL = 0.1     # mm  (= adh_t*0.5; small enough not to distort the 0.2mm cohesive layer)
VISC = '1e-05'    # cohesive damage stabilization viscosity coefficient


def main(src, dst):
    n_tie = 0
    n_stab = 0
    out = []
    with open(src) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        # (1) sandwich tie cards
        if s.startswith('*Tie,') and 'position tolerance' not in s.lower():
            if any(('name=%s' % t) in s for t in SANDWICH_TIES):
                line = line.rstrip('\n') + ', position tolerance=%s\n' % TIE_TOL
                n_tie += 1
        out.append(line)
        # (2) damage stabilization after each *Damage Evolution + its data line
        if s.startswith('*Damage Evolution'):
            # copy the single data line that follows, then inject stabilization
            if i + 1 < len(lines):
                out.append(lines[i + 1])
                i += 1
            out.append('*Damage Stabilization\n')
            out.append(' %s\n' % VISC)
            n_stab += 1
        i += 1
    with open(dst, 'w') as f:
        f.writelines(out)
    print('patched ties: %d (expected 4)' % n_tie)
    print('patched damage-stabilization blocks: %d (expected 2)' % n_stab)
    print('wrote: %s' % dst)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
