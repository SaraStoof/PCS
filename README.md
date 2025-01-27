# PCS
Project Computational Science Group 7 \
Ishana Bohorey \
Sara Stoof \
Oskar Linke \
Windar Mazzori

# 3D Mold modelling
The aim of this project is to simulate mold growth in 3D using DLA (diffusion-limited aggregation). The main file ``dla_3d.py`` runs a set number of simulations based on the given runtime arguments.

# Dependencies
* Numpy, this is used to calculate things you couldn't do with base Python, mostly with regards to points in a circle. \
Install using ``pip install numpy``
* Numba, this is used to speed up most of the program. \
Install using ``pip install numba`` 
* Matplotlib, this is used to visualize the final grid. \
Install using ``pip install matplotlib``

# Runtime instructions
To start the program, run ``python3 dla_3d.py <NUM_SIMS> <BATCH_SIZE> <TEMP> <RH>`` where every value between brackets is a variable. If you simply run ``python3 dla_3d.py`` (or if you don't run it with the correct amount of arguments) the default values will be used instead.
* ``NUM_SIMS`` is the amount of simulations that will be run. Since there is a stochastic element in the program, taking the average over a few runs is necessary to obtain a reasonably recreatable result. \
 The default for this is 5.
* ``BATCH_SIZE`` is the amount of particles that is spawned in with each step. \
The default for this is 1000.
* ``TEMP`` is the temperature in celsius, which could influence the growth and decay of particles. \
The default for this is 30.
* ``RH`` is the relative humidity, which also influences the growth and decay of particles. \
The default for this is 97.

# Output
Running ``dla_3d.py`` gives some data and a few plots as output.
## Output format
All data is outputted to ``stdin``. The first line prints the time taken to run the simulations. The second and third line print the attachment probability and decay probability, which are calculated using the user inputted ``TEMP`` and ``RH``. The fourth, fifth and sixth line show the mold coverage, where the sixth line shows the mold coverage only on the surface. The seventh and eighth line show the M-value, which is the mold index, a way of showing how much mold has grown.

## Plots
The first plot is the number of particles per layer. The second plot is a horizontal slice of the top layer.  The third plot is a vertical slice through the center. The last plot is a 3D plot of the average grid of every simulation run.
