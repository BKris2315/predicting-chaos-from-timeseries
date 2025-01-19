# Predicting Chaos from Timeseries

## Introduvtion
This project was made as the final assignment for the [Classical and Quantum Chaos](https://physics.bme.hu/BMETE15AF45_kov?language=en) course at the Budapest University of Technology and Economics (BME). 

The project aims to verify and implement in python the proposed 0-1 test for chaos in classical systems by Wernecke et al [1]

As a basis the standard lorenz system will be used in this project due to it being well-known and analyized thoroughly in the literature.

## Usage

The project can be used as a package/tool for different metrics such as Lyapunov exponent (logharitmic and Benettin [2, 3] algorithm, and Lyapunov spectrum via Benettin approach [2, 3] as well).

If the user just wants to reproduce results then it is suggested to run the `notebooks/chaos_proj.ipynb`


[1] Wernecke, H., Sándor, B., & Gros, C. (2017). How to test for partially predictable chaos. *Scientific Reports, 7*(1), 1087. https://doi.org/10.1038/s41598-017-01083-x

[2] Benettin, G., Galgani, L., Giorgilli, A. & Strelcyn, J.-M. Lyapunov characteristic exponents for smooth dynamical systems
and for hamiltonian systems; a method for computing all of them. part 1: Theory. Meccanica 15, 9–20 (1980).

[3] Skokos, C. The Lyapunov characteristic exponents and their computation. In Dynamics of Small Solar System Bodies and
Exoplanets, 63–135 (Springer, 2010).
