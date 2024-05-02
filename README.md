# Optimisation for periodic orbits

This repository contains code to (1) search for candidate periodic orbits via gradient-based optimisation and (2) converge solutions using a high-dimensional Newton-Raphson solver. The routines are wrappers for the JAX-CFD DNS code (https://github.com/google/jax-cfd) which is used for efficient computation and backpropagation of derivatives through time forward maps. 

The directory is structured as a python package and for installation users should clone this repository and add it to the python path: 
 

