# -*- coding: utf-8 -*-
"""manufacturing_variability.py

Manufacturing variability module for GT FEM models.
Applies realistic scatter to material properties, ply angles, and CZM parameters
using a reproducible random seed.

Usage:
    from manufacturing_variability import ManufacturingVariability

    mv = ManufacturingVariability(seed=42)
    perturbed_props = mv.perturb_cfrp(E1=160000, E2=10000, G12=5000, ...)
    perturbed_czm = mv.perturb_czm(Kn=1e5, Ks=5e4, tn=50, ts=40, ...)
    perturbed_angles = mv.perturb_ply_angles([45, 0, -45, 90, ...])

Each call draws independent samples. Use the same seed across runs for
reproducibility.
"""

import random
import math


class ManufacturingVariability(object):
    """Apply manufacturing scatter to FEM model properties.

    Scatter sources:
      - CFRP elastic: E1(CoV 5%), E2(8%), G12/G13/G23(10%)
      - CFRP CTE: alpha_11(10%), alpha_22(10%)
      - Core elastic: all E/G(8%), core CTE(10%)
      - CZM: Kn/Ks(15%), tn/ts(15%), GIc/GIIc(20%)
      - Ply angle: +/- 2 deg uniform
      - Thickness: +/- 5% uniform on face and core
    """

    def __init__(self, seed=None):
        """Initialize RNG with given seed for reproducibility."""
        self.rng = random.Random(seed)
        self.seed = seed

    def _gaussian(self, mean, cov):
        """Draw from Gaussian, clamp to [50%, 150%] of mean to avoid extremes."""
        sigma = abs(mean) * cov
        val = self.rng.gauss(mean, sigma)
        lo = mean * 0.5 if mean > 0 else mean * 1.5
        hi = mean * 1.5 if mean > 0 else mean * 0.5
        return max(lo, min(hi, val))

    def _uniform(self, mean, half_range):
        """Draw from uniform [mean - half_range, mean + half_range]."""
        return self.rng.uniform(mean - half_range, mean + half_range)

    # -----------------------------------------------------------------
    # CFRP face sheet properties
    # -----------------------------------------------------------------
    def perturb_cfrp(self, E1, E2, NU12, G12, G13, G23,
                     CTE_11, CTE_22, density):
        """Return dict of perturbed CFRP lamina properties.

        CoV values from literature:
          E1: 5%  (fiber-dominated, low scatter)
          E2: 8%  (matrix-dominated)
          G12/G13/G23: 10% (matrix/shear)
          NU12: 5%
          CTE: 10%
          density: 2%
        """
        return {
            'E1': self._gaussian(E1, 0.05),
            'E2': self._gaussian(E2, 0.08),
            'NU12': max(0.05, min(0.45, self._gaussian(NU12, 0.05))),
            'G12': self._gaussian(G12, 0.10),
            'G13': self._gaussian(G13, 0.10),
            'G23': self._gaussian(G23, 0.10),
            'CTE_11': self._gaussian(CTE_11, 0.10),
            'CTE_22': self._gaussian(CTE_22, 0.10),
            'density': self._gaussian(density, 0.02),
        }

    # -----------------------------------------------------------------
    # Honeycomb core properties
    # -----------------------------------------------------------------
    def perturb_core(self, E1, E2, E3, NU12, NU13, NU23,
                     G12, G13, G23, CTE, density):
        """Return dict of perturbed core engineering constants.

        CoV: E/G 8%, NU 5%, CTE 10%, density 3%
        """
        return {
            'E1': self._gaussian(E1, 0.08),
            'E2': self._gaussian(E2, 0.08),
            'E3': self._gaussian(E3, 0.08),
            'NU12': max(0.001, min(0.1, self._gaussian(NU12, 0.05))),
            'NU13': max(0.001, min(0.1, self._gaussian(NU13, 0.05))),
            'NU23': max(0.001, min(0.1, self._gaussian(NU23, 0.05))),
            'G12': self._gaussian(G12, 0.08),
            'G13': self._gaussian(G13, 0.08),
            'G23': self._gaussian(G23, 0.08),
            'CTE': self._gaussian(CTE, 0.10),
            'density': self._gaussian(density, 0.03),
        }

    # -----------------------------------------------------------------
    # CZM (cohesive zone) parameters
    # -----------------------------------------------------------------
    def perturb_czm(self, Kn, Ks, tn, ts, GIc, GIIc, BK_eta):
        """Return dict of perturbed CZM parameters.

        CoV: stiffness 15%, strength 15%, fracture energy 20%, BK exponent 10%
        """
        return {
            'Kn': self._gaussian(Kn, 0.15),
            'Ks': self._gaussian(Ks, 0.15),
            'tn': self._gaussian(tn, 0.15),
            'ts': self._gaussian(ts, 0.15),
            'GIc': self._gaussian(GIc, 0.20),
            'GIIc': self._gaussian(GIIc, 0.20),
            'BK_eta': self._gaussian(BK_eta, 0.10),
        }

    # -----------------------------------------------------------------
    # Ply angle perturbation
    # -----------------------------------------------------------------
    def perturb_ply_angles(self, angles, max_dev_deg=2.0):
        """Perturb ply orientations by +/- max_dev_deg (uniform).

        Args:
            angles: list of nominal angles (degrees)
            max_dev_deg: maximum deviation (default 2 degrees)

        Returns:
            list of perturbed angles (float)
        """
        return [self._uniform(a, max_dev_deg) for a in angles]

    # -----------------------------------------------------------------
    # Thickness perturbation
    # -----------------------------------------------------------------
    def perturb_thickness(self, nominal, cov=0.05):
        """Perturb a thickness value.

        Returns a positive value with given CoV.
        """
        val = self._gaussian(nominal, cov)
        return max(nominal * 0.5, val)

    # -----------------------------------------------------------------
    # Summary (for logging)
    # -----------------------------------------------------------------
    def summary(self, cfrp, core, czm, angles_8, face_t, core_t):
        """Return a multi-line string summarizing perturbations."""
        lines = []
        lines.append("Manufacturing Variability (seed=%s):" % self.seed)
        lines.append("  CFRP: E1=%.0f(%.1f%%) E2=%.0f G12=%.0f" % (
            cfrp['E1'], 100 * (cfrp['E1'] / 160000.0 - 1),
            cfrp['E2'], cfrp['G12']))
        lines.append("  Core: E1=%.1f G12=%.1f G13=%.1f" % (
            core['E1'], core['G12'], core['G13']))
        lines.append("  CZM: Kn=%.0e tn=%.1f GIc=%.3f" % (
            czm['Kn'], czm['tn'], czm['GIc']))
        lines.append("  Ply angles: %s" % ', '.join(
            ['%.1f' % a for a in angles_8]))
        lines.append("  Thickness: face=%.3f core=%.2f" % (face_t, core_t))
        return '\n'.join(lines)
