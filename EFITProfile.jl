using Revise
using Plots
using ProfileView


include("EFITModule/EFIT.jl")
nThreads = Threads.nthreads()
println("n threads: $nThreads")

#Material declarationsj
materials = [Main.EFIT.IsoMat(3300.0, 1905.0, 2800.0)]

matGrid = ones(Int32, 100, 100, 100);
grid = Main.EFIT.EFITGrid(matGrid,materials,0.00001,1);
grid.v[50,50,50,1]=0.001
Main.EFIT.IsoStep!(grid)
function profile()

    for i in 1:10
        println(i)
        Main.EFIT.IsoStep!(grid);
    end
end
#@time profile()
@ProfileView.profview profile()   



