#!/usr/bin/env python

"""
Run a simple water simulation
"""

from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout


# --- parameters ------------------------------------------------------------ #

pdb = PDBFile('water512.pdb')
forcefield = ForceField('iamoeba.xml')
temperature = 300*kelvin
pressure = 1*bar
timestep = 0.5 # femtoseconds
sim_len_in_steps = 20000000 # 10 ns
log_file = 'my_first_water.log'

# --------------------------------------------------------------------------- #

system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, 
    nonbondedCutoff=0.7*nanometers, vdwCutoff=0.9*nanometer, 
    constraints=None, rigidWater=False,
    polarization='direct', useDispersionCorrection=True)
integrator = LangevinIntegrator(temperature, 1.0/picoseconds, timestep*femtoseconds)
integrator.setConstraintTolerance(0.00001)

# add a monte carlo barostat
system.addForce(MonteCarloBarostat(pressure, temperature))

platform = Platform.getPlatformByName('CUDA')
platform.loadPluginsFromDirectory('/home/tjlane/opt/openmm/lib/plugins')

properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': '2'}
simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()

# equilibrate for 50 ps
simulation.context.setVelocitiesToTemperature(temperature)
print('Equilibrating for 25ps...')
simulation.step( int(25000/timestep) )

# write coords every ps
simulation.reporters.append(DCDReporter('output.dcd', int(1000/timestep)))

# log info to a file every ps
f = open(log_file, 'w')
print("Logging output to: %s" % log_file)
simulation.reporters.append(StateDataReporter(f, int(1000/timestep), step=True, 
    potentialEnergy=True, temperature=True))

print('Running Production (%f ps)...' % (sim_len_in_steps / timestep,))
simulation.step(sim_len_in_steps)

f.close()
print('Done!')

