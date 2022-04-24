using Revise
using Plots

include("EFITModule/EFIT.jl")
nThreads = Threads.nthreads()
println("n threads: $nThreads")
#Material declarationsj



materials = [Main.EFIT.IsoMat(3300.0, 1905.0, 2800.0)]

matGrid = ones(Int32, 100, 100, 100);
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

anim = @animate for n = 1:500
    println(n)
    Main.EFIT.IsoStep!(grid)

    grid.v[sx,sy,sz,:]+=source(grid.dt*float(n),0)
    Plots.heatmap(abs.(grid.v[:,:,50,3]), clim=(0,0.005))
end
    
gif(anim, "heatmap.gif", fps = 60)
    



