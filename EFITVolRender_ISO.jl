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
matGrid = ones(Int, nx,ny,nz)
#Create the array of materials to be used 
materials = [Main.EFIT.IsoMat(100,0,10),Main.EFIT.IsoMats["Austenite"]]
#Add a section of the second reflector in the middle
matGrid.= 1
matGrid[2:end-3,2:end-3,2:end-3].=2

#Maximum sound speed in the model
c = 6000
#Frequency, hz
f0=1e6

#Maximum spatial increment
dx = 0.5*c/(8*f0)
#Maximum time increment
dt = dx/(c*sqrt(3))

#Create the grid class that stores all simulation information
grid = Main.EFIT.EFITGrid(matGrid,materials,dt,dx);

#Source positions
sx=50
sy=50
sz=97

# animation settings
nframes = 1600
framerate = 30

tIterator = 0:grid.dt:grid.dt*nframes
fig,ax,plt = volume(Array(grid.vx),algorithm=:mip,colorrange = (0, 0.04),colormap=:curl, transparency=true)

#Step the simulation
function stepSim(t)
    println(t)
    #Isotropic step function
    Main.EFIT.SimStep!(grid)
    #Apply the source to the Z-direction
    Main.EFIT.applyShapedSource!(grid, (sx,sy,sz),(1,2),t,sourceParams=(f0),shape=(10),θ=0,ϕ=0)
    #Update the plot
    plt.volume = sqrt.(grid.vx.^2 .+ grid.vy.^2 .+ grid.vz.^2)
end

record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)

#Store the result as a brick of values file for LLNL VisIT
writeResult = Main.EFIT.writeToBOV(1.0,100, grid,directory="A://")




