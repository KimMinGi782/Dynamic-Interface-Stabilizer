# Dynamic Interface Stabilization Simulator

> Concept-proof simulation demonstrating that a **state-reactive (dynamic) battery interface**
> outperforms a static interface across all key performance metrics.
> Baseline data sourced from the **NASA Li-ion Battery Aging Dataset**.

---

## Key Results (NASA Data)

| Metric | Improvement |
|---|---|
| Capacity Retention | +35% more capacity retained |
| Internal Resistance | -30% less resistance growth |
| Operating Temperature | Lower thermal stress |
| Safety Events | Significantly fewer |

---

## Files

| File | Description |
|---|---|
| `dynamic_interface_sim_v1.py` | Fully synthetic surrogate model (no data needed) |
| `dynamic_interface_sim_v2_nasa.py` | NASA data as baseline + dynamic model applied |
| `dynamic_interface_invest.py` | Investment-ready charts with NASA data |

---

## NASA Dataset

Real Li-ion 18650 battery (2Ah) aging data from NASA Ames Prognostics Center of Excellence.
Each cycle records voltage, current, temperature, and capacity until 30% capacity fade.

Download: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset

Place downloaded files in a folder called `nasa_data/` inside this project.

---

## Installation

```
python --version
```
Need Python 3.8+. Download at https://www.python.org if needed.

```
pip install -r requirements.txt
```

---

## How to Run

### Option 1 — Investment charts (recommended)
```
python dynamic_interface_invest.py --data_dir "nasa_data/archive/cleaned_dataset/data" --out out_invest
```

### Option 2 — Full simulation with NASA data
```
python dynamic_interface_sim_v2_nasa.py --data_dir "nasa_data/archive/cleaned_dataset/data" --out out_nasa
```

### Option 3 — Synthetic only (no data needed)
```
python dynamic_interface_sim_v1.py --cycles 50 --crate 2.0 --out out_demo
```

---

## Output

```
out_invest/
├── 00_dashboard_INVEST.png   <- All 4 metrics in one figure (for presentations)
├── 01_capacity.png
├── 02_dcir.png
├── 03_temperature.png
└── 04_safety.png
```

---

## Disclaimer

This is a surrogate/concept-proof model for demonstration purposes.
No proprietary composition, thickness, or process data is included.
The dynamic interface response is simulated on top of real NASA baseline data.

---

## Citation

B. Saha and K. Goebel (2007). "Battery Data Set",
NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

---

## License

MIT License
