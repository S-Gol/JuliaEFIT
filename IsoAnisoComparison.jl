using Revise
using GLMakie
using Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using NPZ
include("EFITModule/EFIT.jl")



ParallelStencil.@reset_parallel_stencil()
USE_GPU = false
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float32, 3)
else
    @init_parallel_stencil(Threads, Float32, 3)
end



#Create a grid of integer material indices 
nx = ny = nz = 100
matGrid = ones(Int16, nx,ny,nz)
#Create the array of materials to be used 
#Use anisotropic stiffness matrices for isotropic materials 
materials = [Main.EFIT.IsoMats["steel"], Main.EFIT.IsoMats["Polystyrene"]]
anisoMaterials = [Main.EFIT.AnisoMat(m) for m in materials]

#Add a section of the second reflector in the middle
matGrid[40:60,40:60,50:60] .=2
#Maximum sound speed in the model
c = 6000
#Frequency, hz
f0=1e6
#Period
t0 = 1.00 / f0
#Maximum spatial increment
dx = c/(8*f0)
#Maximum time increment
dt = dx/(c*sqrt(3))

#Create the grid class that stores all simulation information
isoGrid = Main.EFIT.EFITGrid(matGrid,materials,dt,dx);
anisoGrid = Main.EFIT.EFITGrid(matGrid,anisoMaterials,dt,dx);
grids = [isoGrid, anisoGrid]

#Create a source function from which input can be taken
function source(t)
    v = exp(-((2*(t-2*t0)/(t0))^2))*sin(2*pi*f0*t)*0.1
    return Data.Number(v)
end

#Source positions
const sx=50
const sy=50
const sz=98

# animation settings
nframes = 250
framerate = 30

tIterator = 0:isoGrid.dt:isoGrid.dt*nframes
fig,ax,plt = volume(Array(isoGrid.vx),algorithm=:mip,colorrange = (0, 1e-6),colormap=:curl, transparency=true)

#Step the simulation
function stepSim(t)
    println(t)
    for g in grids
        Main.EFIT.SimStep!(g)
        @parallel (45:55,45:55,sz:sz) Main.EFIT.applySource!(g.vx,g.vy,g.vz, Data.Number(0.0), Data.Number(0.0), (source(t)))

    end
    error = sqrt.((isoGrid.vx .- anisoGrid.vx).^2 + (isoGrid.vy .- anisoGrid.vy).^2 + (isoGrid.vz .- anisoGrid.vz).^2)

    #Update the plot
    plt.volume = error
end



record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)

#Store the result as a brick of values file for LLNL VisIT
#writeResult = Main.EFIT.writeToBOV(1.0,100, grid,directory="A://")




