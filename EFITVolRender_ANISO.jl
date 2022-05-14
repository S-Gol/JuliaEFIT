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
materials = [Main.EFIT.AnisoMat(Main.EFIT.IsoMats["steel"]),
Main.EFIT.AnisoMat(Main.EFIT.IsoMat(10,5,1))]

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
grid = Main.EFIT.EFITGrid(matGrid,materials,dt,dx);


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

tIterator = 0:grid.dt:grid.dt*nframes
fig,ax,plt = volume(Array(grid.vx),algorithm=:mip,colorrange = (0, 0.01),colormap=:curl, transparency=true)

#Step the simulation
function stepSim(t)
    println(t)
    #Isotropic step function
    Main.EFIT.SimStep!(grid)
    #Apply the source to the Z-direction
    @parallel (45:55,45:55,sz:sz) Main.EFIT.applySource!(grid.vx,grid.vy,grid.vz, Data.Number(0.0), Data.Number(0.0), (source(t)))

    #Update the plot
    plt.volume = sqrt.(grid.vx.^2 .+ grid.vy.^2 .+ grid.vz.^2)
end



#record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)

#Store the result as a brick of values file for LLNL VisIT
#writeResult = Main.EFIT.writeToBOV(1.0,100, grid,directory="A://")




