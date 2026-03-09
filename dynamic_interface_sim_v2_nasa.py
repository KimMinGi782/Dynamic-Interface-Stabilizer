#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Interface Stabilization Simulator v2.1
===============================================
Uses real NASA Li-ion Battery data (numbered CSVs like 00239.csv)
as the Baseline, then simulates a state-reactive dynamic interface on top.

Usage:
  python dynamic_interface_sim_v2_nasa.py \
    --data_dir "C:\Users\LG\Desktop\battery-interface-sim\nasa_data\archive\cleaned_dataset\data" \
    --out out_nasa
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# Load: handles numbered CSVs (00239.csv etc.)
# ─────────────────────────────────────────────

def load_nasa_csv(data_dir):
    """
    Merges all numbered CSV files in data_dir into one DataFrame,
    then builds a per-cycle summary.
    """
    all_csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not all_csvs:
        all_csvs = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))

    if not all_csvs:
        print(f"\n❌ No CSV files found in: {data_dir}")
        print("Please check the path and try again.")
        sys.exit(1)

    print(f"Found {len(all_csvs)} CSV files — loading...")

    frames = []
    for i, path in enumerate(all_csvs):
        try:
            tmp = pd.read_csv(path)
            tmp["_file_index"] = i
            frames.append(tmp)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")

    if not frames:
        print("❌ Could not read any CSV files.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"Merged: {len(df)} rows, {len(df.columns)} columns")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"Columns: {list(df.columns)}")

    # Find cycle column
    cycle_col = next((c for c in df.columns if "cycle" in c), None)
    if cycle_col is None:
        # Use file index as cycle proxy
        cycle_col = "_file_index"
        print("No cycle column found — using file order as cycle proxy.")

    # Find data columns
    cap_col  = next((c for c in df.columns if "capacity" in c), None)
    v_col    = next((c for c in df.columns if "voltage" in c), None)
    t_col    = next((c for c in df.columns if "temp" in c), None)
    i_col    = next((c for c in df.columns if "current" in c), None)

    print(f"Using → cycle:{cycle_col}  capacity:{cap_col}  voltage:{v_col}  temp:{t_col}  current:{i_col}")

    # Build per-cycle summary
    cycles = []
    for cyc, grp in df.groupby(cycle_col):
        row = {"cycle": cyc}
        row["capacity"]   = float(grp[cap_col].dropna().iloc[-1]) if cap_col and not grp[cap_col].dropna().empty else np.nan
        row["mean_voltage"] = float(grp[v_col].dropna().mean())   if v_col  and not grp[v_col].dropna().empty  else np.nan
        row["mean_temp"]    = float(grp[t_col].dropna().mean())   if t_col  and not grp[t_col].dropna().empty  else np.nan

        if i_col and v_col:
            mean_i = float(grp[i_col].dropna().abs().mean())
            mean_v = float(grp[v_col].dropna().mean())
            row["dcir_proxy"] = (4.2 - mean_v) / mean_i if mean_i > 0.01 else np.nan
        else:
            row["dcir_proxy"] = np.nan

        cycles.append(row)

    result = pd.DataFrame(cycles)
    if cap_col:
        result = result.dropna(subset=["capacity"])
        result = result[result["capacity"] > 0]
    result = result.reset_index(drop=True)
    result["cycle_num"] = np.arange(1, len(result) + 1)
    print(f"Per-cycle summary: {len(result)} cycles")
    return result


# ─────────────────────────────────────────────
# Dynamic Interface Model
# ─────────────────────────────────────────────

def apply_dynamic_interface(df_real, seed=42):
    rng = np.random.default_rng(seed)
    k_resp, k_rs, k_heat = 0.75, 0.70, 0.55

    cap_init = df_real["capacity"].iloc[0]
    dcir_init = df_real["dcir_proxy"].dropna().iloc[0] if df_real["dcir_proxy"].dropna().shape[0] > 0 else 1.0
    temp_init = df_real["mean_temp"].dropna().iloc[0]  if df_real["mean_temp"].dropna().shape[0] > 0 else 25.0
    ambient = temp_init

    cap_dyn, dcir_dyn, temp_dyn = cap_init, dcir_init, temp_init
    out_cap, out_dcir, out_temp, out_vsag, out_evt = [], [], [], [], []

    for _, row in df_real.iterrows():
        excess = max(0.0, temp_dyn - ambient)
        response = float(np.clip(k_resp * (0.6 * 2.0 + 0.4 * excess / 20.0), 0.0, 0.8))

        # DCIR
        dcir_real = row.get("dcir_proxy", dcir_dyn)
        if not np.isnan(dcir_real):
            growth = max(0.0, (dcir_real - dcir_dyn)) * (1.0 - k_rs * response)
            dcir_dyn = max(dcir_dyn + growth + rng.normal(0, 0.002), 0.5)
        out_dcir.append(dcir_dyn)

        # Temperature
        temp_real = row.get("mean_temp", temp_dyn)
        if not np.isnan(temp_real):
            temp_dyn += (temp_real - temp_dyn) * (1.0 - k_heat * response) * 0.5
        out_temp.append(temp_dyn)

        # Capacity
        cap_real = row["capacity"]
        if len(out_cap) > 0 and not np.isnan(cap_real):
            fade = max(0.0, (out_cap[-1] - cap_real) * (1.0 - 0.6 * response))
            cap_dyn = out_cap[-1] - fade
        out_cap.append(cap_dyn)

        # Voltage sag & safety
        vsag = dcir_dyn * 2.0 * 0.12 + rng.normal(0, 0.01)
        out_vsag.append(vsag)
        out_evt.append(1 if (temp_dyn - ambient > 15 or dcir_dyn > 2.5) else 0)

    df_dyn = df_real.copy()
    df_dyn["capacity"]     = out_cap
    df_dyn["dcir_proxy"]   = out_dcir
    df_dyn["mean_temp"]    = out_temp
    df_dyn["vsag_proxy"]   = out_vsag
    df_dyn["safety_event"] = out_evt

    df_real = df_real.copy()
    df_real["vsag_proxy"]   = df_real["dcir_proxy"].fillna(1.0) * 2.0 * 0.12
    ambient_base = df_real["mean_temp"].fillna(25).iloc[0]
    df_real["safety_event"] = (
        ((df_real["mean_temp"].fillna(ambient_base) - ambient_base) > 15).astype(int) |
        (df_real["dcir_proxy"].fillna(1.0) > 2.5).astype(int)
    )
    return df_real, df_dyn


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

def plot_results(df_base, df_dyn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x = df_base["cycle_num"].values

    plots = [
        ("01_capacity.png",      df_base["capacity"].values,                       df_dyn["capacity"].values,                       "Capacity (Ah)",          "Capacity Retention"),
        ("02_dcir.png",          df_base["dcir_proxy"].values,                     df_dyn["dcir_proxy"].values,                     "DCIR proxy (Ω)",         "Internal Resistance"),
        ("03_temperature.png",   df_base["mean_temp"].fillna(25).values,           df_dyn["mean_temp"].fillna(25).values,           "Temperature (°C)",       "Mean Temperature"),
        ("04_voltage_sag.png",   df_base["vsag_proxy"].fillna(0).values,           df_dyn["vsag_proxy"].fillna(0).values,           "Voltage sag proxy (V)",  "Voltage Sag"),
        ("05_safety_events.png", np.cumsum(df_base["safety_event"].fillna(0).values), np.cumsum(df_dyn["safety_event"].fillna(0).values), "Cumul. safety events",   "Safety Events"),
    ]

    for fname, yb, yd, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, yb, color="#e05c5c", lw=2, label="Baseline (NASA real)")
        ax.plot(x, yd, color="#4a90d9", lw=2, label="Dynamic Interface")
        ax.set_xlabel("Cycle"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)
        print(f"  ✅ {fname}")

    # Dashboard
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("NASA Battery — Baseline vs Dynamic Interface", fontsize=14)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    panels = [
        (0,0, df_base["capacity"].values,                         df_dyn["capacity"].values,                         "Capacity (Ah)",     "Capacity Retention"),
        (0,1, df_base["dcir_proxy"].values,                       df_dyn["dcir_proxy"].values,                       "DCIR proxy (Ω)",    "Internal Resistance"),
        (1,0, df_base["mean_temp"].fillna(25).values,             df_dyn["mean_temp"].fillna(25).values,             "Temperature (°C)",  "Temperature"),
        (1,1, np.cumsum(df_base["safety_event"].fillna(0).values),np.cumsum(df_dyn["safety_event"].fillna(0).values),"Safety events",     "Safety Events"),
    ]
    for r, c, yb, yd, ylabel, title in panels:
        ax = fig.add_subplot(gs[r, c])
        ax.plot(x, yb, color="#e05c5c", lw=1.8, label="Baseline")
        ax.plot(x, yd, color="#4a90d9", lw=1.8, label="Dynamic")
        ax.set_xlabel("Cycle", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "00_dashboard.png"), dpi=160)
    plt.close(fig)
    print(f"  ✅ 00_dashboard.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder with NASA CSV files")
    ap.add_argument("--out",      type=str, default="out_nasa", help="Output folder")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    print(f"\n📂 Loading data from: {args.data_dir}")
    df_real = load_nasa_csv(args.data_dir)

    print("\n⚙️  Applying dynamic interface model...")
    df_base, df_dyn = apply_dynamic_interface(df_real, seed=args.seed)

    print(f"\n📊 Saving plots to: {args.out}/")
    plot_results(df_base, df_dyn, args.out)

    print(f"\n🎉 Done! Open the '{args.out}' folder to see your graphs.")


if __name__ == "__main__":
    main()
