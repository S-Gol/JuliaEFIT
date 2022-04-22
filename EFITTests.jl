using Revise
using Plots

include("EFITModule/EFIT.jl")
nThreads = Threads.nthreads()
println("n threads: $nThreads")
#Material declarations



materials = [Main.EFIT.IsoMat(3300.0, 1905.0, 2800.0)]

matGrid = ones(Int32, 100, 100, 100);
grid = Main.EFIT.EFITGrid(matGrid,materials,0.00001,1);
grid.v[50,50,50,1]=0.001



anim = @animate for i = 1:500
    println(i)
    Main.EFIT.IsoStep!(grid)
    Plots.heatmap(abs.(grid.σ[:,:,50,1,1]), clim=(0,1000))
end
    
gif(anim, "heatmap.gif", fps = 10)
    



