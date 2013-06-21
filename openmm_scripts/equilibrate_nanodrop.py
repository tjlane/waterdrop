#!/usr/bin/env python

"""
Equilibrate a nanodrop sim
"""

from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout


# --- parameters ------------------------------------------------------------ #

pdb = PDBFile('nanodrop.pdb')
forcefield = ForceField('iamoeba.xml')
temperature = 300*kelvin
timestep = 0.5 # femtoseconds
log_file = 'equilibration.log'

# --------------------------------------------------------------------------- #

system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, 
    nonbondedCutoff=0.7*nanometers, vdwCutoff=0.9*nanometer, 
    constraints=None, rigidWater=False,
    polarization='direct', useDispersionCorrection=True)
integrator = LangevinIntegrator(temperature, 1.0/picoseconds, timestep*femtoseconds)
integrator.setConstraintTolerance(0.00001)


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

out_fn = 'nanodrop_equilibrated.pdb'
print('Saving: %s...' % out_fn)
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open(out_fn, 'w'))

print('Done!')

