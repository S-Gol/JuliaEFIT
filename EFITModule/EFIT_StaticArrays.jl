module EFIT
    using LoopVectorization
    using Unrolled
    export EFITGrid, EFITMaterial, IsoMat, IsoSim
    abstract type EFITMaterial end
    struct IsoMat <: EFITMaterial
        ρ::Float32
        λ::Float32
        μ::Float32
        function IsoMat(cl::Number, cs::Number, ρ::Number)
            λ=ρ*(cl^2+cs^2)
            μ=ρ*cs^2
            new(ρ,λ,μ)
        end
    end
    
    
    mutable struct EFITGrid{T<:EFITMaterial}
        v::Array{NTuple{3,Float32},3}
        v′::Array{NTuple{3,Float32},3}

        σ::Array{NTuple{9,Float32},3}
        σ′::Array{NTuple{9,Float32},3}

        matIdx::Array{Int32,3}
        materials::Array{T,1}

        dt::Float32
        ds::Float32

        xSize::Int32
        ySize::Int32
        zSize::Int32

        function EFITGrid(matGrid::Array,materials::AbstractArray, dt::Number, ds::Number)
            xSize = size(matGrid)[1]
            ySize = size(matGrid)[2]
            zSize = size(matGrid)[3]

            v = Array{NTuple{3,Float32},3}(undef, xSize,ySize,zSize)
            v′ = Array{NTuple{3,Float32},3}(undef, xSize,ySize,zSize)
            σ = Array{NTuple{9,Float32},3}(undef, xSize,ySize,zSize)
            σ′ = Array{NTuple{9,Float32},3}(undef, xSize,ySize,zSize)
            
            Threads.@threads for N in CartesianIndices((1:xSize,1:ySize,1:zSize))
                v[N]=(0.0,0.0,0.0)
                v′[N]=(0.0,0.0,0.0)
                σ[N]=(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
                σ′[N]=(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
            end
            println("Creating grid of size $xSize, $ySize, $zSize")
            println(typeof(σ′))
            new{eltype(materials)}(v,v′,σ,σ′,matGrid,materials,dt,ds,xSize,ySize,zSize)
        end
    end 

    """Averages ρ across the given direction"""
    function averagedρ(grid::EFITGrid, I::CartesianIndex,dir::Int64,offsets::Vector{CartesianIndex{3}})::Float32 
        ρavg::Float32 = (2.0 / ((grid.materials[grid.matIdx[I]]).ρ+(grid.materials[grid.matIdx[I+offsets[dir]]]).ρ))
        return ρavg
    end
    function copyArrs!(grid)
        Threads.@threads for N in CartesianIndices((2:grid.xSize-1,2:grid.ySize-1,2:grid.zSize-1))
            grid.σ[N] += grid.σ′[N] * grid.dt

            for i in 1:3
                grid.v[N,i] += grid.v′[N,i] * grid.dt
            end
        end
    end

    const offsets = [CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]
     
    """Performs a single timestep on an EFITGrid struct"""
    function IsoStep!(grid::EFITGrid)
        invds = 1.0 / grid.ds

        Threads.@threads for N in CartesianIndices((2:grid.xSize-1,2:grid.ySize-1,2:grid.zSize-1))
            #Update the velocity derivatives
            #Iterate each direction - x,y,z
            v0, v1, v2, v3 = grid.σ[N], grid.σ[N-offsets[1]], grid.σ[N-offsets[2]], grid.σ[N-offsets[3]]
            grid.v′[N] = (averagedρ(grid, N, 1, offsets) * invds *
                          (grid.σ[N+offsets[1]][1] - v0[1] + v0[2] - v2[2] + v0[3] - v3[3]),
                          averagedρ(grid, N, 2, offsets) * invds *
                          (v0[4] - v1[4] + grid.σ[N+offsets[2]][5] - v0[5] + v0[6] - v3[6]),
                          averagedρ(grid, N, 3, offsets) * invds *
                          (v0[7] - v1[7] + v0[8] - v2[8] + grid.σ[N+offsets[3]][9] - v0[9]))
                  
     
        end
        #Finish, update all integrals
 


    end

end
