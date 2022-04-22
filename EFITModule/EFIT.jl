module EFIT
    using StaticArrays
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
        v::Array{Float32,4}
        v′::Array{Float32,4}

        σ::Array{Float32,5}
        σ′::Array{Float32,5}

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

            v = zeros(Float32,xSize,ySize,zSize,3)
            v′ = zeros(Float32,xSize,ySize,zSize,3)
            σ = zeros(Float32,xSize,ySize,zSize,6,6)
            σ′ = zeros(Float32,xSize,ySize,zSize,6,6)

            println("Creating grid of size $xSize, $ySize, $zSize")
            new{eltype(materials)}(v,v′,σ,σ′,matGrid,materials,dt,ds,xSize,ySize,zSize)
        end
    end 



    """Performs a single timestep on an EFITGrid struct"""
    function IsoStep!(grid::EFITGrid)
        function getAveragedρ(I,dir)
            offset = CartesianIndex(1,0,0)
            return 2 / ((grid.materials[grid.matIdx[I]]).ρ+(grid.materials[grid.matIdx[I+offset]]).ρ)
        end
        offsets = [CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]

        Threads.@threads for N in CartesianIndices((2:grid.xSize-1,2:grid.ySize-1,2:grid.zSize-1))
            #Update the velocity derivatives
            #Iterate each direction - x,y,z
            for dir in 1:3
                sigmaComp = 0
                #Iterate the second index  
                for i in 1:3
                    #On diagonals, we use a different offset
                    if i == dir
                        sigmaComp += grid.σ[(N+offsets[i]),dir,i]-grid.σ[N,dir,i]
                    else
                        sigmaComp += grid.σ[N,dir,i]-grid.σ[(N-offsets[i]),dir,i]
                    end
                end
                grid.v′[N,dir] = getAveragedρ(N,dir)*(1/grid.ds)*sigmaComp
            end
            #Update the diagonal stresses
            λ = grid.materials[grid.matIdx[N]].λ
            λ2μ = (λ + grid.materials[grid.matIdx[N]].μ*2)
            
            for dir in 1:3
                vComp = 0
                for i in 1:3
                    if i == dir
                        vComp += λ2μ*(grid.v[N,i]-grid.v[N-offsets[i],i])
                    else
                        vComp += λ*(grid.v[N,i]-grid.v[N-offsets[i],i])
                    end
                end
                grid.σ′[N,dir,dir]=vComp/grid.ds
            end
        end
        #Finish, update all integrals
        @. grid.v+=grid.v′*grid.dt
        @. grid.σ+=grid.σ′*grid.dt

    end

end
