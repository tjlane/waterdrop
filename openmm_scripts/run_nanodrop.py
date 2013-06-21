#!/usr/bin/env python

"""
Run a water simulation of a small droplet of water. This simulation will
include no barostat, and will attempt to remove gaseous water that diffuses
away from the main droplet. Does this by checking every 1ps to see if there
is any water further than a certain radius cutoff from the center of the
drop, and if there is, rming it.

Note: this sim is run in NVT, and purposefully has no barostat/thermostat.
It requires the input to be an equilibrated nanodrop system.
"""

from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout


# --- parameters ------------------------------------------------------------ #

pdb = PDBFile('nanodrop_equilibrated.pdb')
forcefield = ForceField('iamoeba.xml')
start_temperature = 300*kelvin # This needs to match the equilibration cond.
timestep = 0.5 # femtoseconds
r_cutoff = 1.0 # nm? not in use atm
sim_len_in_steps = 200000000 # 100 ns
log_file = 'nanodrop_sim.log'

# --------------------------------------------------------------------------- #

system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
    nonbondedCutoff=0.7*nanometers, vdwCutoff=0.9*nanometer,
    constraints=None, rigidWater=False,
    polarization='direct', useDispersionCorrection=True)
integrator = VerletIntegrator(0.5*femtoseconds)

platform = Platform.getPlatformByName('CUDA')
platform.loadPluginsFromDirectory('/home/tjlane/opt/openmm/lib/plugins')

properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': '2'}
simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(start_temperature)

# write coords every ps
simulation.reporters.append(DCDReporter('output.dcd', int(1000/timestep)))

# log info to a file every ps
f = open(log_file, 'w')
print("Logging output to: %s" % log_file)
simulation.reporters.append(StateDataReporter(f, int(1000/timestep), step=True, 
    potentialEnergy=True, temperature=True))


# run the production calculation -- every ps looping to remove water
# that gets too far away
print('Running Production (%f ps)...' % (sim_len_in_steps / timestep,))

num_rounds = sim_len_in_steps / int(1000/timestep)

for i in range(num_rounds):
    simulation.step( int(1000/timestep) ) # run one ps
    # TJL : skipping deletion for now, making sure all else works
    print("Round %d/%d : removing water further than %f from COM" % (i+1, num_rounds, r_cutoff))

f.close()
print('Done!')

