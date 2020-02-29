BOND PERCOLATIONS
=================
This repository was created as an assignment in `Statistical Mechanics` in `Weizmann Institute of Science` by prof. Oren Raz.
This task is about the statistical phenomenon of percolation, and renormalization.

The assignment is about bond percolation on 2D square lattice.

# Functions
## Question 1 - finding P(trajectory) vs. p
- Calls function `trajectory_per_p` with predefined parameters
- Plots the phase transition graph.

## trajectory_per_p
- Loops for all p's needed, for all realizations requested.
- for each realization, calculates whether a trjectory is found on the lattice.
- Returns a p values array and the probability to find trajectory for each p.

# Class - lattice
## __init__
- A lattice is defined by
    - N - length scale
    - p - probability for a closed edge on the lattice
- properties:
    - numclusters - Num of different clusters found on the lattice
    - clusters - 2d array for each site, with the cluster ID it is a part of
    - percolators - array of cluster ID's that are percolating
    - 
     

# References
The code was written by *Keren Avnery*, based on original code by mickp
https://github.com/mickp/bondperc