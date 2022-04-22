abstract type EFITMaterial end
mutable struct IsoMat <: EFITMaterial
    ρ::Float32
    λ::Float32
    μ::Float32
    function IsoMat(cl::Number, cs::Number, ρ::Number)
        λ=ρ*(cl^2+cs^2)
        μ=ρ*cs^2
        new(λ,μ,ρ)
    end
end


mutable struct EFITGrid
    u::Array{Float32,3}
    τ::Array{Float32,5}
    matIdx::Array{Int32,3}
    materials::Array{EFITMaterial,1}
    function EFITGrid(xSize::Number, ySize::Number, zSize::Number, materials)
        u=zeros(Float32,xSize,ySize,zSize)
        τ=zeros(Float32,xSize,ySize,zSize,6,6)
        matIndices = zeros(Int32, xSize, ySize, zSize)
        new(u,τ,matIndices,materials)
    end
end 



