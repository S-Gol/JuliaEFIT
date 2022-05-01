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


include("EFITModule/EFITParallelStencils.jl")
nThreads = Threads.nthreads()
println("n threads: $nThreads")


#Material declarations
materials = [Main.EFIT.IsoMats["steel"],Main.EFIT.IsoMat(1,0,1)]

matGrid = ones(Int32, 100, 100, 100);
#matGrid[40:60,40:60,40:50].=2
grid = Main.EFIT.EFITGrid(matGrid,materials,0.0019,20);

nLayers = float(10)
for x = 1:floor(Int,nLayers)
    for y = 1:floor(Int,nLayers)
        weight = min(exp(-0.015*(x)),exp(-0.015*(y)))

        #+- X directions
        grid.BCWeights[x,:,:] .= weight
        grid.BCWeights[100-x,:,:] .= weight

        #+- z directions
        grid.BCWeights[:,y,:] .= weight
        grid.BCWeights[:,100-y,:] .= weight
    end

end
println(materials[1])
println(grid.dt)
println(grid.ds)

#Frequency, hz
f0=30
#Period
t0 = 1.00 / f0

function source(t)
    v = exp(-((2*(t-2*t0)/(t0))^2))*sin(2*pi*f0*t)*0.1
    return v
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
    grid.vz[50,50,50]+=source(t)
    Main.EFIT.IsoStep!(grid)
    plt.volume = sqrt.(Array(grid.vx).^2 .+ Array(grid.vy).^2 .+ Array(grid.vz).^2)
end

record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)




