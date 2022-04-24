module EFIT
    using LoopVectorization
    using StaticArrays

    export EFITGrid, EFITMaterial, IsoMat, IsoSim, IsoMats
    abstract type EFITMaterial end
    struct IsoMat <: EFITMaterial
        ρ::Float32
        λ::Float32
        μ::Float32
        function IsoMat(cl::Number, cs::Number, ρ::Number)
            λ=ρ*(cl^2-2*cs^2)
            μ=ρ*cs^2
            new(ρ,λ,μ)
        end
    end
    
    
    struct EFITGrid{T<:EFITMaterial}
        v::Array{Float32,4}

        σ::Array{Float32,5}

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
            σ = zeros(Float32,xSize,ySize,zSize,3,3)

            println("Creating grid of size $xSize, $ySize, $zSize")
            new{eltype(materials)}(v,σ,matGrid,materials,dt,ds,xSize,ySize,zSize)
        end
    end 

    """Averages ρ across the given direction"""
    function averagedρ(grid::EFITGrid, I::CartesianIndex,dir::Int64,offsets::Vector{CartesianIndex{3}})::Float32 
        ρavg::Float32 = (2.0 / ((grid.materials[grid.matIdx[I]]).ρ+(grid.materials[grid.matIdx[I+offsets[dir]]]).ρ))
        return ρavg
    end

    const offsets = [CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]
    function velDeriv!(grid::EFITGrid,N::CartesianIndex)
        @inbounds for dir in 1:3
            sigmaComp::Float32 = 0
            @inbounds for i in 1:3
                #On diagonals, we use a different offset
                if i == dir
                    sigmaComp += grid.σ[(N+offsets[i]),dir,i]-grid.σ[N,dir,i]
                else
                    sigmaComp += grid.σ[N,dir,i]-grid.σ[(N-offsets[i]),dir,i]
                end
            end
            grid.v[N,dir] += averagedρ(grid, N,dir,offsets)*(1.0/grid.ds)*sigmaComp*grid.dt
        end      
    end
    """Performs a single timestep on an EFITGrid struct"""
    function IsoStep!(grid::EFITGrid)
        Threads.@threads for N in CartesianIndices((2:grid.xSize-1,2:grid.ySize-1,2:grid.zSize-1))
            #Update the velocity derivatives
            #Iterate each direction - x,y,z
            velDeriv!(grid,N)
        end
        Threads.@threads for N in CartesianIndices((2:grid.xSize-1,2:grid.ySize-1,2:grid.zSize-1))
            #Update the diagonal stresses
            λ = grid.materials[grid.matIdx[N]].λ
            λ2μ = (λ + grid.materials[grid.matIdx[N]].μ*2)
            @inbounds for dir in 1:3
                vComp::Float32 = 0
                @inbounds for i in 1:3
                    if i == dir
                        @views vComp += λ2μ*(grid.v[N,i]-grid.v[N-offsets[i],i])
                    else
                        @views vComp += λ*(grid.v[N,i]-grid.v[N-offsets[i],i])
                    end
                end
                grid.σ[N,dir,dir]+=grid.dt*vComp/grid.ds
            end   
            
            #Update the shear stresses
            @inbounds for i in 1:3
                @inbounds for j in (i+1):3
                    μΔs::Float32 = (1.0/grid.ds) * 4.0/(
                        (1.0/grid.materials[grid.matIdx[N]].μ) + (1.0/grid.materials[grid.matIdx[N+offsets[i]]].μ) +
                        (1.0/grid.materials[grid.matIdx[N+offsets[j]]].μ) + (1.0/grid.materials[grid.matIdx[N+offsets[i]+offsets[j]]].μ)
                    )
                    σT::Float32 = μΔs * ((grid.v[N+offsets[i],j]-grid.v[N,j])+(grid.v[N+offsets[j],i]-grid.v[N,i]))
                    grid.σ[N,i,j]+=grid.dt*σT    
                    grid.σ[N,j,i]=grid.σ[N,i,j]
                end
            end       
        end
    end


    #Sample materials
    IsoMats = Dict(
        "steel"=>IsoMat(5960, 3235, 8000),
        "lightweightGeneric"=>IsoMat(596, 323, 800),
        "Aluminum"=>IsoMat(6420, 3040, 2700),
        "Berylium"=>IsoMat(12890, 8880, 1870),
        "Brass"=>IsoMat(4700, 2110, 8600),
        "Copper"=>IsoMat(4760, 2325, 8930),
        "Gold"=>IsoMat(3240, 1200, 19700),
        "Iron"=>IsoMat(5960, 3240, 7850),
        "Lead"=>IsoMat(2160, 700, 11400),
        "Molybdenum"=>IsoMat(6250, 3350, 10100),
        "Nickel"=>IsoMat(5480, 2990, 8850),
        "Platinum"=>IsoMat(3260, 1730, 21400),
        "Silver"=>IsoMat(3650, 1610, 10400),
        "Mild steel"=>IsoMat(5960, 3235, 7850),
        "Stainless"=>IsoMat(5790, 3100, 7900),
        "Tin"=>IsoMat(3320, 1670, 7300),
        "Titanium"=>IsoMat(6070, 3125, 4500),
        "Tungsten"=>IsoMat(5220, 2890, 19300),
        "Tungsten Carbide"=>IsoMat(6655, 3980, 13800),
        "Zinc"=>IsoMat(4210, 2440, 7100),
        "Fused silica"=>IsoMat(5968, 3764, 2200),
        "Pyrex"=>IsoMat(5640, 3280, 2320),
        "Glass"=>IsoMat(3980, 2380, 3880),
        "Lucite"=>IsoMat(2680, 1100, 1180),
        "Nylon"=>IsoMat(2620, 1070, 1110),
        "Polyethylene"=>IsoMat(1950, 540, 900),
        "Polystyrene"=>IsoMat(2350, 1120, 1060),
)
end
