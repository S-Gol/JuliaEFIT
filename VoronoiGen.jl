using LoopVectorization
const offsets = [CartesianIndex(-1,0,0),CartesianIndex(1,0,0),CartesianIndex(0,-1,0),CartesianIndex(0,1,0),CartesianIndex(0,0,-1),CartesianIndex(0,0,1)]

function initFloodFill!(nx::Number,ny::Number,nz::Number, nGrains::Number, matGrid::Array{Int,3}, positions::Array{Int,2})
    Qs = Array{Array{CartesianIndex}}(undef, nGrains)
    offsets = [CartesianIndex(-1,0,0),CartesianIndex(1,0,0),CartesianIndex(0,-1,0),CartesianIndex(0,1,0),CartesianIndex(0,0,-1),CartesianIndex(0,0,1)]
    #Initialize the queue
    for i in 1:size(positions,1)
        p = CartesianIndex(positions[i,1],positions[i,2],positions[i,3])
        Qs[i] = Array{CartesianIndex,1}()
        matGrid[p]=i
        for o in offsets
            new = p + o
            if new[1] < nx && new[1] > 0 && new[2] < ny && new[2] > 0 && new[3] < nz && new[3] > 0
                if matGrid[new] == 0
                    push!(Qs[i], new)
                end
            end
        end
    end
    return Qs
end
function stepFloodFill!(matGrid::Array{Int,3}, Qs::Array{Array{CartesianIndex}})
    length::Int = 0
    
    @inbounds for i in 1:size(Qs,1)
        if size(Qs[i],1) > 0
            q0 = popfirst!(Qs[i])
            if matGrid[q0] == 0
                matGrid[q0] = i
                
                for o in offsets
                    new = q0 + o
                    if new[1] <= nx && new[1] > 0 && new[2] <= ny && new[2] > 0 && new[3] <= nz && new[3] > 0
                        if matGrid[new] == 0
                            push!(Qs[i], new)
                            length += 1 
                        end
                    end
                end
            end
        end
    end
    return length
    
end

function fullRandomGen(nx,ny,nz, nSeeds)
    matGrid = zeros(Int,nx,ny,nz)
    positions = reduce(hcat,[rand(1:nx, nGrains),rand(1:ny, nGrains),rand(1:nz, nGrains)]);
    Qs = initFloodFill!(nx,ny,nz,nGrains,matGrid,positions);
    count = 1
    while count > 0
        count = stepFloodFill!(matGrid, Qs)
    end
    return matGrid
end    

