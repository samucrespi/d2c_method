# d2c: a New Methodology for Simulating Terrestrial Planet Formation with Post-collisional Debris
This repository introduces a novel methodology for including post-collisional debris in simulations of terrestrial planet formation.

## Problem Statement
Collisions between protoplanets can produce a vast amount of particles ranging in size from dust to large asteroids. Including all these new particles in N-body simulations is unfeasible, and approximations are required.

## Solution
Instead of including all the post-collisional debris, this methodology includes in the simulation a limited number of effective particles, each of which embodies a swarm of debris with similar orbits.

## Method
After a collision occurs, the relevant collision parameters are retrieved. By using open-source SPH catalogues (Winter et al. 2022), it is possible to obtain the physical and dynamical properties of the post-collisional debris. However, the actual collision may not be present in the available catalogue, therefore interpolation is required. The interpolation result is a considerable number of debris, which mass is weighted by the distance (or a function of it) between the actual collision and the SPH collisions from the catalogue.

The debris are then clustered in 3D velocity-space using a k-mean clustering algorithm. The resulting clustered debris are included in the simulation as semi-interacting particles (they only interact with each other). Collisions between clustered debris and protoplanets always result in a merging event.

## Benefits
This methodology is more efficient than including all the post-collisional debris, reducing computational costs and allowing for more realistic simulations of terrestrial planet formation. It also provides a way to account for the effects of post-collisional debris without compromising the accuracy of the simulation.

## Usage
To use this methodology, please refer to the documentation provided in this repository. The documentation provides step-by-step instructions for implementation.

## Contributions
Contributions to this repository are welcome. Please refer to the contribution guidelines provided in this repository.

## References
Winter, A. J., Chambers, J. E., & Price, M. C. (2022). Smoothed Particle Hydrodynamics Catalogs of Planetesimal Impacts. The Astrophysical Journal Supplement Series, 166(1), 12. [https://doi.org/10.3847/1538-4365/ac248c](https://doi.org/10.3847/1538-4365/ac248c)
