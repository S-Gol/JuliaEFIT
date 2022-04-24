using Revise
using GLMakie
using Colors


include("EFITModule/EFIT.jl")
nThreads = Threads.nthreads()
println("n threads: $nThreads")

#Plotting

#Material declarations
materials = [Main.EFIT.IsoMat(3300.0, 1905.0, 2800.0)]

matGrid = ones(Int32, 200, 200, 200);
grid = Main.EFIT.EFITGrid(matGrid,materials,0.001,10);

println(materials[1])
println(grid.dt)
println(grid.ds)
#Frequency, hz
f0=30
#Period
t0 = 1.00 / f0

function source(t, nt)
    v = exp(-((2*(t-2*t0)/(t0))^2))*sin(2*pi*f0*t)*0.1
    return [0,0,v]
end
const sx=50
const sy=50
const sz=50

# animation settings
nframes = 500
framerate = 30

tIterator = 0:grid.dt:grid.dt*nframes
fig,ax,plt = volume(grid.v[:,:,:,1],algorithm=:mip,colorrange = (0, 0.005),colormap=:curl, transparency=true)

function stepSim(t)
    println(t)

    grid.v[sx,sy,sz,:]+=source(t,0)
    Main.EFIT.IsoStep!(grid)
    plt.volume = grid.v[:,:,:,1]
end

record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)




