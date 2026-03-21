# Option Pricing Models

This project implements several option pricing models that I built while studying quantitative finance at CentraleSupélec. It's a hands-on way to explore the concepts I come across in class and during my internship.

## About this project

I'm a student in the Quantitative Finance track at CentraleSupélec. This repository is my attempt to put into practice what I learn about stochastic modeling, Monte Carlo methods, and financial derivatives. The code here reflects my understanding of how pricing models work under the hood.

## Models implemented

- **Black-Scholes**: closed-form pricing and Greeks for European options
- **Heston**: stochastic volatility model
- **Merton Jump-Diffusion**: price jumps following a Poisson process
- **Bates**: combines Heston and Merton (stochastic volatility + jumps)
- **Longstaff-Schwartz**: regression-based pricing for American options
- **Implied volatility**: extracting volatility smiles from simulated prices

## Project structure
src/ # core modules (models, Monte Carlo engine, utils)
notebooks/ # Jupyter notebooks with examples and analysis
requirements.txt # Python dependencies


## Getting started
git clone https://github.com/Alexandre-Reyob/option-pricing-models.git
cd option-pricing-models
pip install -r requirements.txt
jupyter notebook notebooks/

## Notebooks
The notebooks are numbered in a logical order:

Black-Scholes basics and Greeks
Comparing European option prices across models
American options and the Longstaff-Schwartz algorithm
Volatility smiles from Heston simulations

## About me
I'm Alexandre Reyob, a student at CentraleSupélec, a French engineering school. I built this project to deepen my understanding of the models used in trading desks and asset management. It's not meant for production use, it was just a learning exercise that I'm happy to share.

