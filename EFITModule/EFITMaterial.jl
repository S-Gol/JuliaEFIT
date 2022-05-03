abstract type EFITMaterial end
struct IsoMat <: EFITMaterial
    ρ::Float32
    λ::Float32
    μ::Float32
    λ2μ::Float32
    function IsoMat(cl::Number, cs::Number, ρ::Number)
        λ=ρ*(cl^2-2*cs^2)
        μ=ρ*cs^2
        λ2μ = λ+2*μ
        new(ρ,λ,μ,λ2μ)
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