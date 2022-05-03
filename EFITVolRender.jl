using Revise
using GLMakie
using Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using NPZ


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
#matGrid = npzread("MeshFiles/Rail.npy")[:,5:end,1:340].+1

nx = ny = nz = 100
matGrid = ones(Int, nx,ny,nz)
materials = [Main.EFIT.IsoMats["steel"],Main.EFIT.IsoMat(1,0,1)]
matGrid[40:60,40:60,50:60] .=2
c = 6000
#Frequency, hz
f0=1e6
#Period
t0 = 1.00 / f0

dx = c/(8*f0)
dt = dx/(c*sqrt(3))


grid = Main.EFIT.EFITGrid(matGrid,materials,dt,dx);



function source(t)
    v = exp(-((2*(t-2*t0)/(t0))^2))*sin(2*pi*f0*t)*0.1
    return Data.Number(v)
end
const sx=50
const sy=50
const sz=98

# animation settings
nframes = 50
framerate = 30

tIterator = 0:grid.dt:grid.dt*nframes
fig,ax,plt = volume(Array(grid.vx),algorithm=:mip,colorrange = (0, 0.005),colormap=:curl, transparency=true)

println(grid.dtds)
function stepSim(t)

    println(t)
    Main.EFIT.IsoStep!(grid)
    Threads.@threads for x in 40:60
        for y in 40:60
            @parallel (x:x,y:y,sz:sz) Main.EFIT.applySource!(grid.vx,grid.vy,grid.vz, Data.Number(0.0), Data.Number(0.0), (source(t)))
        end
    end
    plt.volume = sqrt.(grid.vx.^2 .+ grid.vy.^2 .+ grid.vz.^2)
end



record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)
velMag = sqrt.(grid.vx.^2 .+ grid.vy.^2 .+ grid.vz.^2)

writeResult = Main.EFIT.writeToBOV(velMag, 1.0,100, grid,directory="A://")




