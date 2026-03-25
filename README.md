# Hull‑White Modelization

This repository contains a **Python implementation of the Hull‑White one‑factor short-rate model** from scratch, focused on pricing interest rate derivatives and simulating short rate paths using real market data.

The **Hull‑White model** is a popular interest rate model in quantitative finance, extending the classical Vasicek model by allowing time-dependent drift to fit the initial term structure of interest rates. ([Wikipedia](https://en.wikipedia.org/wiki/Hull%E2%80%93White_model))

## Features

- Implementation of the Hull‑White model from scratch  
- Calibration to market OIS/forward curves  
- Simulation of short-rate paths  
- Pricing and visualization of interest rate derivatives (CMS, cap swaptions, bond options)  
- Clear Python modules and Jupyter notebook  

## Repository Structure
Hull‑and‑White‑modelization/

│

├── hull_and_white.ipynb ← Main notebook with code & visualizations

├── HW1F.py ← Hull‑White model classes

├── stripping.py ← Yield curve bootstrap & helper functions

├── data/ ← Market data (OIS, forward curves, etc.)

├── .gitignore

└── README.md
