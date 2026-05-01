# Annulus Velocity Component Plot Export

This folder is self-contained for regenerating the `u_theta(r)` and `u_r(r)` plots for:

- `theta = 0.00`
- `Rin = 1.00`
- `Rout = 2.00`
- `sidelen = 0.1`

## Contents

- `generate_velocity_component_profiles.py`: plotting/export script
- `analytical/0.00_1.00_2.00.csv`: analytical source data
- `discrete/0.00_1.00_2.00.csv`: numerical/discrete source data
- `plots/`: generated clean CSVs, PNGs, and PDFs
- `requirements-viz.txt`: Python package requirements for plotting

## Regenerate

From this folder:

```bash
python3 generate_velocity_component_profiles.py --component theta
python3 generate_velocity_component_profiles.py --component r
```

Both commands export all 500 source rows to clean CSVs. The plots use only rows where `1.1 <= r <= 1.9`, marked by the `in_plot_window` column.
