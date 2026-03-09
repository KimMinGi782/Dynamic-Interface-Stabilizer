#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Interface Stabilization Simulator v1.0 (SIM-ONLY)
========================================================
Purpose:
- Demonstrates the concept of "dynamic interface stabilization structures
  whose interface properties change based on operating state" via simulation.
- Compares Baseline (static interface) vs Dynamic Interface (state-reactive interface):
  1) 2C voltage sag (drop)
  2) DCIR change
  3) ΔT temperature rise
  4) Capacity retention (N cycles)
  5) Safety event count (rule-based)

Note:
- This is a surrogate/concept-proof model for patent/proposal demonstration purposes.
- No composition ratios, thicknesses, or process details are included.

Usage:
  python dynamic_interface_sim_v1.py --out out_demo --cycles 50 --crate 2.0 --seed 42

Output:
  out_demo/
    summary.csv
    trace_baseline.csv
    trace_dynamic.csv
    01_voltage_drop.png
    02_dcir.png
    03_dT.png
    04_capacity.png
    05_safety_events.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_csv(path, header, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# -----------------------------
# Surrogate Battery + Interface Models
# -----------------------------
class BatterySurrogate:
    """
    Simple 0D battery surrogate model:
    - capacity (state of health): decreases from 1.0
    - dcir (internal resistance proxy): increases with cycling
    - temp (surface temperature rise proxy): I^2*R based + cooling term
    - voltage sag: I*R based + noise
    """
    def __init__(self, seed=0, ambient=25.0):
        self.rng = np.random.default_rng(seed)
        self.ambient = ambient

        # Initial values (relative/arbitrary units)
        self.cap = 1.00    # 1.0 = 100%
        self.dcir = 1.00   # baseline R0 = 1.0
        self.temp = ambient

    def step_cycle(self, crate, interface_policy):
        """
        Advance one cycle.
        - crate: e.g., 0.5, 1, 2, 3C
        - interface_policy: dict controlling baseline vs dynamic behavior
        Returns:
        - voltage_drop, dcir, dT, capacity, safety_event (0/1)
        """
        I = crate  # simplified: current ∝ C-rate

        # Dynamic interface response parameters
        k_resp = interface_policy.get("k_resp", 0.0)   # 0=static, >0=dynamic
        k_heat = interface_policy.get("k_heat", 0.0)   # heat dissipation gain
        k_rs   = interface_policy.get("k_rs", 0.0)     # resistance growth suppression

        # "Response intensity" = f(I, temp) — simulates state-reactive interface
        response = k_resp * (0.6 * I + 0.4 * (max(0.0, (self.temp - self.ambient)) / 20.0))
        response = clip(response, 0.0, 0.8)

        # Resistance growth
        stress = (0.010 * I) + (0.006 * max(0.0, self.temp - self.ambient) / 10.0)
        noise  = self.rng.normal(0.0, 0.002)
        growth = (0.008 + stress + noise) * (1.0 - k_rs * response)
        growth = max(0.0, growth)
        self.dcir *= (1.0 + growth)

        # Temperature rise
        p_heat = (I * I) * self.dcir
        heat_factor = clip(0.45 * (1.0 - k_heat * response), 0.10, 0.45)
        self.temp = self.temp + heat_factor * p_heat - 0.18 * (self.temp - self.ambient)
        dT = self.temp - self.ambient

        # Capacity fade
        fade = (0.0018 + 0.0006 * I + 0.0008 * max(0.0, dT) / 25.0) * (1.0 - 0.6 * response)
        self.cap = clip(self.cap * (1.0 - max(0.0, fade)), 0.0, 1.0)

        # Voltage drop proxy
        v_drop = (I * self.dcir) * 0.12 + self.rng.normal(0.0, 0.01)

        # Safety event (rule-based)
        event = 1 if (dT > 35.0 or self.dcir > 1.85) else 0

        return float(v_drop), float(self.dcir), float(dT), float(self.cap), int(event)


def run_sim(cycles, crate, seed, out_dir):
    ensure_dir(out_dir)

    # Baseline: static interface (no dynamic response)
    baseline_policy = dict(k_resp=0.0, k_heat=0.0, k_rs=0.0)

    # Dynamic: state-reactive interface
    dynamic_policy = dict(k_resp=0.75, k_heat=0.55, k_rs=0.70)

    base = BatterySurrogate(seed=seed,     ambient=25.0)
    dyn  = BatterySurrogate(seed=seed + 1, ambient=25.0)

    trace_base, trace_dyn = [], []

    for c in range(1, cycles + 1):
        vb, rb, tb, cb, eb = base.step_cycle(crate=crate, interface_policy=baseline_policy)
        vd, rd, td, cd, ed = dyn.step_cycle(crate=crate,  interface_policy=dynamic_policy)
        trace_base.append([c, crate, vb, rb, tb, cb, eb])
        trace_dyn.append( [c, crate, vd, rd, td, cd, ed])

    header = ["cycle", "crate", "voltage_drop", "dcir", "deltaT", "capacity", "safety_event"]
    save_csv(os.path.join(out_dir, "trace_baseline.csv"), header, trace_base)
    save_csv(os.path.join(out_dir, "trace_dynamic.csv"),  header, trace_dyn)

    base_events = sum(r[-1] for r in trace_base)
    dyn_events  = sum(r[-1] for r in trace_dyn)
    summary_rows = [
        ["Baseline", cycles, crate, trace_base[-1][3], trace_base[-1][4], trace_base[-1][5], base_events],
        ["Dynamic",  cycles, crate, trace_dyn[-1][3],  trace_dyn[-1][4],  trace_dyn[-1][5],  dyn_events],
    ]
    save_csv(
        os.path.join(out_dir, "summary.csv"),
        ["variant", "cycles", "crate", "final_dcir", "final_deltaT", "final_capacity", "safety_events"],
        summary_rows
    )

    # Arrays for plotting
    x  = np.arange(1, cycles + 1)
    vb = np.array([r[2] for r in trace_base]); vd = np.array([r[2] for r in trace_dyn])
    rb = np.array([r[3] for r in trace_base]); rd = np.array([r[3] for r in trace_dyn])
    tb = np.array([r[4] for r in trace_base]); td = np.array([r[4] for r in trace_dyn])
    cb = np.array([r[5] for r in trace_base]) * 100; cd = np.array([r[5] for r in trace_dyn]) * 100
    eb = np.cumsum([r[6] for r in trace_base]); ed = np.cumsum([r[6] for r in trace_dyn])

    plots = [
        ("01_voltage_drop.png", vb, vd, "Voltage drop (proxy)", f"Voltage Sag @ {crate:.1f}C"),
        ("02_dcir.png",         rb, rd, "DCIR (normalized)",     "DCIR Growth"),
        ("03_dT.png",           tb, td, "ΔT (°C, proxy)",        "Temperature Rise Proxy"),
        ("04_capacity.png",     cb, cd, "Capacity retention (%)", "Capacity Retention"),
        ("05_safety_events.png",eb, ed, "Cumulative safety events", "Safety Events (rule-based)"),
    ]

    for fname, yb, yd, ylabel, title in plots:
        plt.figure()
        plt.plot(x, yb, label="Baseline", color="#e05c5c")
        plt.plot(x, yd, label="Dynamic",  color="#4a90d9")
        plt.xlabel("Cycle")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    print(f"\n✅ Done! Results saved to: {out_dir}/")
    print("\nSummary:")
    print(f"{'Variant':<12} {'Final DCIR':>12} {'Final ΔT':>10} {'Capacity%':>10} {'SafeEvents':>12}")
    for row in summary_rows:
        print(f"{row[0]:<12} {row[3]:>12.4f} {row[4]:>10.2f} {row[5]*100:>9.1f}% {row[6]:>12}")


def main():
    ap = argparse.ArgumentParser(description="Dynamic Interface Stabilization Simulator")
    ap.add_argument("--out",    type=str,   default="out_demo", help="Output folder")
    ap.add_argument("--cycles", type=int,   default=50,         help="Number of cycles (default: 50)")
    ap.add_argument("--crate",  type=float, default=2.0,        help="C-rate, e.g. 2.0 for 2C (default: 2.0)")
    ap.add_argument("--seed",   type=int,   default=42,         help="Random seed (default: 42)")
    args = ap.parse_args()

    run_sim(cycles=args.cycles, crate=args.crate, seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
