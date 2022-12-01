# Quality Indicators for Preference-based Evolutionary Multi-objective Optimization Using a Reference Point

This repository provides the Python implementation of preference-based quality indicators used in the following paper.

> Ryoji Tanabe, Ke Li, **Quality Indicators for Preference-based Evolutionary Multi-objective Optimization Using a Reference Point: A Review and Analysis**, submitted.

## Requirements

This code require Python (=>3.8), [pygmo](https://esa.github.io/pygmo2/index.html) and [pymoo](https://pymoo.org/).

## Usage

The following command calculates the MASF values of the 10 point sets on the Pareto front of the DTLZ2 problem (see Fig. 2 in the paper):

```
python qindicator.py
```

The results are saved in a directory ``pp_results``. Here, the CSV files of the 10 point sets are in ``pset_dataset``.  The IGD-reference point sets are in ``ref_point_dataset`` The weight vector sets for the R2 calculation are in ``weight_point_dataset``. More details can be found in ``qindicator.py``.
