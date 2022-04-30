using Revise
using GLMakie
using Colors


include("EFITModule/EFIT.jl")
nThreads = Threads.nthreads()
println("n threads: $nThreads")

#Plotting

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
    return [0,0,v]
end
const sx=50
const sy=50
const sz=98

# animation settings
nframes = 500
framerate = 30

tIterator = 0:grid.dt:grid.dt*nframes
fig,ax,plt = volume(grid.v[:,:,:,1],algorithm=:mip,colorrange = (0, 0.005),colormap=:curl, transparency=true)

function stepSim(t)
    println(t)

    grid.v[sx,sy,sz,:]+=source(t)
    Main.EFIT.IsoStep!(grid)
    plt.volume = sqrt.(grid.v[:,:,:,1].^2 .+ grid.v[:,:,:,2].^2 .+ grid.v[:,:,:,3].^2)
end

record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)




