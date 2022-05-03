# Elastodynamic Finite Integration Technique using Julia
The Elastodynamic Finite Integration Technique is a numerical solution technique used to simulate elastodynamic waves in solid media. Support arbitrary sources, isotropic material properties, frequency ranges, and physical size. 

![Simulation rendered volumetrically using GLMakie](https://raw.githubusercontent.com/S-Gol/JuliaEFIT/main/ReadmeImages/Volumetric_100.gif)

## Implementation

The EFIT Simulations are implemented in pure Julia using the ParallelStencils library. This library provides high-performance finite-differencing compatible with both multithreaded CPU and GPU/CUDA execution

## Usage

The main component of this system is the EFITGrid struct. It contains stress, velocity, and material information on the simulation. It can be initialized with

```Julia
EFITGrid(matGrid::Array,materials::Array, dt::Number, ds::Number)
```

`matGrid` is an array of integers of the desired (3D) dimensions for the simulation space. Each int references an element of `materials` which contains all of the materials used in the model. 

Example:
```Julia
#Create a sim space of size nx, ny, nz
matGrid = ones(Int, nx,ny,nz)
#Create the array of materials to be used 
materials = [Main.EFIT.IsoMats["steel"],Main.EFIT.IsoMats["Aluminum"]]
```

dT and dS come from the CFL equations - 

```Julia 
#Maximum sound speed in the model
c = 6000
#Frequency, hz
f0=1e6

dx = c/(8*f0)
dt = dx/(c*sqrt(3))
```

Once the EFITGrid is initialized with these values, we can use `EFIT.SimStep!()` to perform a simulation step.

Sources are added by directly modifying the velocity. Additionally, the `applySource!()` function can be used to apply a source value to a large array of points, using `ParallelStencils` acceleration.  