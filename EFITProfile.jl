using Revise
using ProfileView
using Traceur
using BenchmarkTools

include("EFITModule/EFIT.jl")

nThreads = Threads.nthreads()
println("n threads: $nThreads")

#Material declarationsj
materials = [Main.EFIT.IsoMat(3300.0, 1905.0, 2800.0)]

matGrid = ones(Int32, 100, 100, 100);
grid = Main.EFIT.EFITGrid(matGrid,materials,0.00001,1);
Main.EFIT.IsoStep!(grid)



function profile()
    for i in 1:50
        println(i)
        Main.EFIT.IsoStep!(grid)
    end
end
function threadedVel()
    Threads.@threads for N in CartesianIndices((2:grid.xSize-1,2:grid.ySize-1,2:grid.zSize-1))
        Main.EFIT.velUpdate!(grid,N)
    end
end
#@time Main.EFIT.velUpdate!(grid,CartesianIndex(5,5,5))
#@time Main.EFIT.IsoStep!(grid)
#@ProfileView.profview profile()   
#trace Main.EFIT.velUpdate!(grid,CartesianIndex(5,5,5))

#@benchmark Main.EFIT.IsoStep!(grid)

#println("Old velocity benchmark: ")
#@benchmark threadedVel()

#println("ParallelStencil benchmark: ")

@benchmark Main.EFIT.IsoStep!(grid)
#Main.EFIT.IsoStep!(grid)