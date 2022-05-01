using Revise
using GLMakie
using Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

ParallelStencil.@reset_parallel_stencil()
USE_GPU = false
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float32, 3)
else
    @init_parallel_stencil(Threads, Float32, 3)
end


include("EFITModule/EFIT.jl")
nThreads = Threads.nthreads()
println("n threads: $nThreads")


#Material declarations
materials = [Main.EFIT.IsoMats["steel"],Main.EFIT.IsoMats["lightweightGeneric"]]

c = 8000
#Frequency, hz
f0=1e6
#Period
t0 = 1.00 / f0

dx = c/(8*f0)
dt = dx/(c*sqrt(3))

matGrid = ones(Int32, 100, 100, 100);
matGrid[40:60,40:60,40:50].=2
grid = Main.EFIT.EFITGrid(matGrid,materials,dt,dx);



function source(t)
    v = exp(-((2*(t-2*t0)/(t0))^2))*sin(2*pi*f0*t)*0.1
    return Data.Number(v)
end
const sx=50
const sy=50
const sz=98

# animation settings
nframes = 500
framerate = 30

tIterator = 0:grid.dt:grid.dt*nframes
fig,ax,plt = volume(Array(grid.vx),algorithm=:mip,colorrange = (0, 0.005),colormap=:curl, transparency=true)

println(grid.dtds)
function stepSim(t)
    println(t)
    Main.EFIT.IsoStep!(grid)
    @parallel (sx:sx+1,sy:sy+1,sz:sz+1) Main.EFIT.applySource!(grid.vx,grid.vy,grid.vz, Data.Number(0.0), Data.Number(0.0), (source(t)))

    plt.volume = sqrt.(Array(grid.vx).^2 .+ Array(grid.vy).^2 .+ Array(grid.vz).^2)
end

record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)




