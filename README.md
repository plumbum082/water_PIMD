# water_PIMD

## Introduction

This water forcefield describes the potential energy surface of water with the nonbonding interaction predicted by multipolar polarizable model (implemented in DMFF.ADMP) in the long range and machine learning model (EANN) in the short range and bonding interaction calculated by MB-pol. The path integral molecular dynamics (PIMD) simulations are preformed by i-PI.

In this repository, the folder `pimd_32beads` contains files to run a standard 32-bead PIMD simulation, while the folder `piglet_6beads` contains files to run a 6-bead PIGLET simulation, for which the parameters in `input.xml` can be obtained from the [GLE4MD website](http://gle4md.org/index.html?page=matrix). For further details please refer to the references below.

## Installation

- DMFF - <https://github.com/deepmodeling/DMFF>

- i-PI - `pip install i-pi`

## Example

Run PIMD simulations with this command line below.

```sh
sbatch sub.sh
```

## References

<a id="1">[1]</a> 
M. Ceriotti and D. E. Manolopoulos, “Efficient First-Principles Calculation of the Quantum Kinetic Energy and Momentum Distribution of Nuclei”, Phys. Rev. Lett. 109, 100604 (2012)

<a id="2">[2]</a> 
M. Ceriotti, J. More, D. Manolopoulos, “i-PI: A Python interface for ab initio path integral molecular dynamics simulations”, Comp. Phys. Comm. 185(3), 1019 (2014)

