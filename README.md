# Optimisation for periodic orbits

Authors: Jacob Page, Peter Norgaard, Michael Brenner & Rich Kerswell

This repository contains code to (1) search for candidate periodic orbits via gradient-based optimisation and (2) converge solutions using a high-dimensional Newton-Raphson solver. The routines are wrappers for the JAX-CFD DNS code (please refer to installation guide here https://github.com/google/jax-cfd) which is used for efficient computation and backpropagation of derivatives through time forward maps. 

The directory is structured as a python package and for installation users should clone this repository and add it to the python path: 

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/dir/"
``` 

For details on the method please refer to our paper "Recurrent flow patterns as a basis for two-dimensional turbulence: predicting statistics from structures" (accepted in Proc. Nat. Acad. Sci., see https://arxiv.org/abs/2212.11886).
Implementation of the method relies on the bounded while loops from the excellent equinox library (https://github.com/patrick-kidger/equinox) to enable jit compilation with a varying target time (period) $T$. 
