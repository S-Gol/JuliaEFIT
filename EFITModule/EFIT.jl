module EFIT
    using LoopVectorization
    
    include("EFITMaterial.jl")
    export EFITGrid, EFITMaterial, IsoMat, IsoSim, IsoMats, AnisoMats, AnisoMat, writeToBOV

    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    ParallelStencil.@reset_parallel_stencil()
    
    USE_GPU = false
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float32, 3)
    else
        @init_parallel_stencil(Threads, Float32, 3)
    end


    struct EFITGrid{T<:EFITMaterial}
        vx::Data.Array
        vy::Data.Array
        vz::Data.Array

        σxx::Data.Array
        σyy::Data.Array
        σzz::Data.Array
        σxy::Data.Array
        σxz::Data.Array
        σyz::Data.Array

        dt::Float32
        ds::Float32
        dtds::Float32

        xSize::Int32
        ySize::Int32
        zSize::Int32

        #BCWeights

        matGrid::Array{Int16, 3}
        materials::Array{T,1}
        matPermDict::Dict{Int64,Matrix{Float32}}

        function EFITGrid(matGrid::Array,materials::Array, dt::Number, ds::Number)
            xSize = size(matGrid)[1]
            ySize = size(matGrid)[2]
            zSize = size(matGrid)[3]

            vx = @zeros(xSize,ySize,zSize)
            vy = @zeros(xSize,ySize,zSize)
            vz = @zeros(xSize,ySize,zSize)

            σxx = @zeros(xSize,ySize,zSize)
            σyy = @zeros(xSize,ySize,zSize)
            σzz = @zeros(xSize,ySize,zSize)

            σxy = @zeros(xSize,ySize,zSize)
            σxz = @zeros(xSize,ySize,zSize)
            σyz = @zeros(xSize,ySize,zSize)

            #=BCWeights = @ones(xSize,ySize,zSize)=#
            println("")
            println("Creating grid of size $xSize, $ySize, $zSize")
            println("")

            if eltype(materials) == AnisoMat
                matDict = Dict{Int64, Matrix{Float32}}()
                setAveragePerms!(matDict, matGrid, materials)

                new{eltype(materials)}(vx,vy,vz,σxx,σyy,σzz,σxy,σxz,σyz,
                dt,ds,dt/ds,xSize,ySize,zSize,#=Data.Array(BCWeights),=#
                matGrid, materials,matDict)
            else
                new{eltype(materials)}(vx,vy,vz,σxx,σyy,σzz,σxy,σxz,σyz,
                dt,ds,dt/ds,xSize,ySize,zSize,#=Data.Array(BCWeights),=#
                matGrid, materials,Dict{Int64, Matrix{Float32}}())
            end
        end
    end 

    """Performs a single timestep on an EFITGrid struct"""
    function SimStep!(grid::EFITGrid{IsoMat})
        @parallel (2:grid.xSize-1, 2:grid.ySize-1, 2:grid.zSize-1) computeV!(
        grid.vx,grid.vy,grid.vz, 
        grid.σxx,grid.σyy,grid.σzz,grid.σxy,grid.σxz,grid.σyz,
        grid.dtds,grid.matGrid,grid.materials
        )

        @parallel (2:grid.xSize-1, 2:grid.ySize-1, 2:grid.zSize-1) computeσIso!(
        grid.vx,grid.vy,grid.vz, 
        grid.σxx,grid.σyy,grid.σzz,grid.σxy,grid.σxz,grid.σyz,
        grid.dtds,grid.matGrid,grid.materials
        )
    end
    function SimStep!(grid::EFITGrid{AnisoMat})
        @parallel (2:grid.xSize-1, 2:grid.ySize-1, 2:grid.zSize-1) computeV!(
        grid.vx,grid.vy,grid.vz, 
        grid.σxx,grid.σyy,grid.σzz,grid.σxy,grid.σxz,grid.σyz,
        grid.dtds,grid.matGrid,grid.materials
        )
        @parallel (2:grid.xSize-1, 2:grid.ySize-1, 2:grid.zSize-1) computeσAniso!(
        grid.vx,grid.vy,grid.vz, 
        grid.σxx,grid.σyy,grid.σzz,grid.σxy,grid.σxz,grid.σyz,
        grid.dtds,grid.matGrid,grid.materials,grid.matPermDict
        )

    end

    #Velocity calculations - common between isotropic and anisotropic
    @parallel_indices (x,y,z) function computeV!(vx::Data.Array,vy::Data.Array,vz::Data.Array, σxx::Data.Array,
        σyy::Data.Array,σzz::Data.Array,σxy::Data.Array,σxz::Data.Array,σyz::Data.Array,
        dtds::Data.Number, matGrid::Array{Int16,3}, mats::AbstractArray)
        #Velocities from stresses
        ρN = mats[(matGrid[x,y,z])].ρ
        vx[x,y,z] = vx[x,y,z] + dtds*(σxx[x+1,y,z]-σxx[x,y,z] + σxy[x,y,z]-σxy[x,y-1,z] + σxz[x,y,z]-σxz[x,y,z-1]) * 2/(ρN + mats[(matGrid[x+1,y,z])].ρ)
        vy[x,y,z] = vy[x,y,z] + dtds*(σxy[x,y,z]-σxy[x-1,y,z] + σyy[x,y+1,z]-σyy[x,y,z] + σyz[x,y,z]-σyz[x,y,z-1]) * 2/(ρN + mats[(matGrid[x,y+1,z])].ρ)
        vz[x,y,z] = vz[x,y,z] + dtds*(σxz[x,y,z]-σxz[x-1,y,z] + σyz[x,y,z]-σyz[x,y-1,z] + σzz[x,y,z+1]-σzz[x,y,z]) * 2/(ρN + mats[(matGrid[x,y,z+1])].ρ)

        return 
    end

    function av(A::Data.Array, x::Integer,y::Integer,z::Integer,dir1::Integer,dir2::Integer)
        i1 = dir1==1
        i2 = dir2==1

        j1 = dir1==2
        j2 = dir2==2

        k1 = dir1==3
        k2 = dir2==3

        return A[x,y,z] + A[x+i1,y+j1,z+k1] + A[x+i2,y+j2,z+k2] + A[x+i1+i2,y+j1+j2,z+k1+k2]  
    end

    
    const R = 1::Integer
    const F = 2::Integer
    const U = 3::Integer

    function dav(A::Data.Array,x::Integer, y::Integer, z::Integer, dir1::Integer, dir2::Integer, face::Integer)
        return av(A,x,y,z, dir1, dir2) - av(A,x-(face==1),y-(face==2),z-(face==3),dir1,dir2)
    end

    #Stress calculations for anisotropic case
    @parallel_indices (x,y,z) function computeσAniso!(vx::Data.Array,vy::Data.Array,vz::Data.Array, σxx::Data.Array,
        σyy::Data.Array,σzz::Data.Array,σxy::Data.Array,σxz::Data.Array,σyz::Data.Array,
        dtds::Data.Number, matGrid::Array{Int16,3}, mats::AbstractArray, cDict::Dict{Int64,Matrix{Float32}})
        
        #=From Halkjaer eq 4.7
        R+X
        L-X   

        F+Y
        B-Y

        U-Z
        D+Z
        =#
    


        mID = matGrid[x,y,z]
        #Velocity differencing terms
        dvxx = vx[x,y,z]-vx[x-1,y,z]
        dvyy = vy[x,y,z]-vy[x,y-1,z]
        dvzz = vz[x,y,z]-vz[x,y,z-1]

        #Averaged Cs from Halkjaer 4.13, 4.14
        c13 = cDict[hashIDX(matGrid[x,y,z],matGrid[x+1,y,z],matGrid[x,y,z+1],matGrid[x+1,y,z+1])]
        c23 = cDict[hashIDX(matGrid[x,y,z],matGrid[x,y+1,z],matGrid[x,y,z+1],matGrid[x,y+1,z+1])]
        c12 = cDict[hashIDX(matGrid[x,y,z],matGrid[x+1,y,z],matGrid[x,y+1,z],matGrid[x+1,y+1,z])]

        #Velocity averaging terms for 4.7
        #Terms cancelled per Leckey CNDE expansions
        vi4 = 0.25*((vy[x,y,z+1] + vy[x,y-1,z+1]) - (vy[x,y,z-1] + vy[x,y-1,z-1]) + (vz[x,y+1,z] + vz[x,y+1,z-1]) - (vz[x,y-1,z] + vz[x,y-1,z-1]))
        vi5 = 0.25*((vx[x,y,z+1] + vx[x-1,y,z+1]) - (vx[x,y,z-1] + vx[x-1,y,z-1]) + (vz[x+1,y,z] + vz[x+1,y,z-1]) - (vz[x-1,y,z] + vz[x-1,y,z-1]))
        vi6 = 0.25*((vx[x+1,y,z] + vx[x-1,y+1,z]) - (vx[x-1,y,z] + vx[x-1,y-1,z]) + (vy[x+1,y,z] + vy[x+1,y-1,z]) - (vy[x-1,y,z] + vy[x-1,y-1,z]))


        #Normal stresses
        σxx[x,y,z] = σxx[x,y,z] + dtds*(mats[mID].c[1,1]*dvxx + mats[mID].c[1,2]*dvyy + mats[mID].c[1,3]*dvzz + 
            mats[mID].c[1,4] * vi4 + mats[mID].c[1,5] * vi5 + mats[mID].c[1,6] * vi6)

        σyy[x,y,z] = σyy[x,y,z] + dtds*(mats[mID].c[2,1]*dvxx + mats[mID].c[2,2]*dvyy + mats[mID].c[2,3]*dvzz + 
            mats[mID].c[2,4] * vi4 + mats[mID].c[2,5] * vi5 + mats[mID].c[2,6] * vi6)

        σzz[x,y,z] = σzz[x,y,z] + dtds*(mats[mID].c[3,1]*dvxx + mats[mID].c[3,2]*dvyy + mats[mID].c[3,3]*dvzz + 
            mats[mID].c[3,4] * vi4 + mats[mID].c[3,5] * vi5 + mats[mID].c[3,6] * vi6)

        #Shear stresses
        #These still need the additional off-diagonal terms. Omitted for testing.


        #13 direction
        vxa = sum(@view vx[x:x+1,y,z:z+1])-sum(@view vx[x-1:x,y,z:z+1])
        vya = sum(@view vy[x:x+1,y,z:z+1])-sum(@view vy[x:x,y-1,z:z+1])
        vza = sum(@view vz[x:x+1,y,z:z+1])-sum(@view vz[x:x+1,y,z-1:z])

        dvydz = (vy[x,y,z] + vy[x+1,y,z] + vy[x,y-1,z] + vy[x+1,y-1,z])-(vy[x,y,z-1] + vy[x+1,y,z-1] + vy[x,y-1,z-1] + vy[x+1,y-1,z-1])
        dvzdy = (vz[x,y+1,z] + vz[x+1,y+1,z]) - (vz[x,y-1,z] + vz[x+1,y-1,z])
        dvxdy = (vx[x,y+1,z] + vx[x,y+1,z+1]) - (vx[x,y-1,z] + vx[x,y-1,z+1])
        dvydx = (vy[x+1,y,z] + vy[x+1,y,z+1] +  vy[x+1,y-1,z] + vy[x+1,y-1,z+1])-(vy[x,y,z] + vy[x,y,z+1] +  vy[x,y-1,z] + vy[x,y-1,z+1])

        diag = dtds*c13[5,5]*(vx[x,y,z+1]-vx[x,y,z]+vz[x+1,y,z]-vz[x,y,z])

        σxz[x,y,z] = σxz[x,y,z] + 0.25*dtds * (c13[5,1]*vxa + c13[5,2]*vya + c13[5,3]*vza + c13[5,4]*(dvydz + dvzdy) + c13[5,6]*(dvxdy + dvydx)) + diag

        
        #23
        vxa = sum(@view vx[x,y:y+1,z:z+1])-sum(@view vx[x-1,y:y+1,z:z+1])
        vya = sum(@view vy[x,y:y+1,z:z+1])-sum(@view vy[x,y-1:y,z:z+1])
        vza = sum(@view vz[x,y:y+1,z:z+1])-sum(@view vz[x,y:y+1,z-1:z])

        dvxdz = (vx[x,y,z+1] + vx[x-1,y,z+1]+vx[x,y+1,z+1] + vx[x-1,y+1,z+1])-(vx[x,y,z] + vx[x-1,y,z]+vx[x,y+1,z] + vx[x-1,y+1,z])
        dvzdx = (vz[x+1,y,z] + vz[x+1,y+1,z])-(vz[x-1,y,z] + vz[x-1,y+1,z])
        dvxdy = (vx[x,y+1,z] + vx[x-1,y+1,z] + vx[x-1,y+1,z+1] + vx[x,y+1,z+1]) - (vx[x,y,z] + vx[x-1,y,z] + vx[x-1,y,z+1] + vx[x,y,z+1])
        dvydx = (vy[x+1] + vy[x+1,y,z+1]) - (vy[x-1,y,z] + vy[x-1,y,z+1])

        diag = dtds*c23[4,4]*(vy[x,y,z+1]-vy[x,y,z]+vz[x,y+1,z]-vz[x,y,z])
        
        σyz[x,y,z] = σyz[x,y,z] + 0.25*dtds * (c23[4,1]*vxa + c23[4,2]*vya + c23[4,3]*vza + c23[4,5] * (dvxdz + dvzdx) + c23[4,6]*(dvxdy + dvydx)) + diag

        #12
        vxa = sum(@view vx[x:x+1,y:y+1,z])-sum(@view vx[x-1:x,y:y+1,z])
        vya = sum(@view vy[x:x+1,y:y+1,z])-sum(@view vy[x:x+1,y-1:y,z])
        vza = sum(@view vz[x:x+1,y:y+1,z])-sum(@view vz[x:x+1,y:y+1,z-1])

        dvydz = (vy[x,y,z+1] + vy[x+1,y,z+1]) - (vy[x,y,z-1] + vy[x+1,y,z-1])
        dvzdy = (vz[x,y+1,z] +vz[x+1,y+1,z]) - (vz[x,y-1,z] + vz[x+1,y-1,z])
        dvxdz = (vx[x,y,z+1] + vx[x,y+1,z+1])-(vx[x,y,z-1] + vx[x,y+1,z-1])
        dvzdx = (vz[x+1,y,z]+vz[x+1,y+1,z] + vz[x+1,y,z+1] + vz[x+1,y+1,z+1])-(vz[x,y,z]+vz[x,y+1,z] + vz[x,y,z+1] + vz[x,y+1,z+1])

        diag = dtds*c12[6,6]*(vx[x,y+1,z]-vx[x,y,z]+vy[x+1,y,z]-vy[x,y,z])

        σxy[x,y,z] = σxy[x,y,z] + 0.25*dtds * (c12[6,1]*vxa + c12[6,2]*vya + c12[6,3]*vza + c12[6,4]*(dvydz + dvzdy) + c12[6,5]*(dvxdz + dvzdx)) + diag
        return
    end

    #Stress calculations for isostropic case
    @parallel_indices (x,y,z) function computeσIso!(vx::Data.Array,vy::Data.Array,vz::Data.Array, σxx::Data.Array,
        σyy::Data.Array,σzz::Data.Array,σxy::Data.Array,σxz::Data.Array,σyz::Data.Array,
        dtds::Data.Number, matGrid::Array{Int16,3}, mats::AbstractArray)
        
        μN = mats[(matGrid[x,y,z])].μ
        λN = mats[(matGrid[x,y,z])].λ
        λ2μN = mats[(matGrid[x,y,z])].λ2μ

        #Diagonal stresses
        σxx[x,y,z] = σxx[x,y,z] + dtds * (λ2μN * (vx[x,y,z]-vx[x-1,y,z]) + λN*(vy[x,y,z] - vy[x,y-1,z] + vz[x,y,z] - vz[x,y,z-1]))
        σyy[x,y,z] = σyy[x,y,z] + dtds * (λ2μN * (vy[x,y,z]-vy[x,y-1,z]) + λN*(vx[x,y,z] - vx[x-1,y,z] + vz[x,y,z] - vz[x,y,z-1]))
        σzz[x,y,z] = σzz[x,y,z] + dtds * (λ2μN * (vz[x,y,z]-vz[x,y,z-1]) + λN*(vy[x,y,z] - vy[x,y-1,z] + vx[x,y,z] - vx[x-1,y,z]))

        #Shear stresses
        σxy[x,y,z] = σxy[x,y,z] + dtds * (vx[x,y+1,z] - vx[x,y,z] + vy[x+1,y,z]-vy[x,y,z]) * 4/(
            1/μN+1/mats[matGrid[x+1,y,z]].μ+1/mats[matGrid[x,y+1,z]].μ+1/mats[matGrid[x+1,y+1,z]].μ
        )

        σxz[x,y,z] = σxz[x,y,z] + dtds * (vx[x,y,z+1] - vx[x,y,z] + vz[x+1,y,z]-vz[x,y,z]) * 4/(
            1/μN+1/mats[matGrid[x+1,y,z]].μ+1/mats[matGrid[x,y,z+1]].μ+1/mats[matGrid[x+1,y,z+1]].μ
        )

        σyz[x,y,z] = σyz[x,y,z] + dtds * (vy[x,y,z+1] - vy[x,y,z] + vz[x,y+1,z]-vz[x,y,z]) * 4/(
            1/μN+1/mats[matGrid[x,y+1,z]].μ+1/mats[matGrid[x,y,z+1]].μ+1/mats[matGrid[x,y+1,z+1]].μ
        )

        return
    end
    @parallel_indices (x,y,z) function applySource!(vx::Data.Array,vy::Data.Array,vz::Data.Array, 
    sx::Data.Number,sy::Data.Number,sz::Data.Number)
        vx[x,y,z]+=sx
        vy[x,y,z]+=sy
        vz[x,y,z]+=sz
        return
    end

    function writeToBOV(data, t::Number,nt::Int,grid::EFITGrid; directory::AbstractString="", filePrefix::AbstractString="data")
        headerName = "$filePrefix-$nt.bov"
        dataName = "$filePrefix-$nt.bin"
        nx = size(data,1)
        ny = size(data,2)
        nz = size(data,3)
        #Write the BOV header
        open("$directory/$headerName", "w") do headerFile
            println(headerFile, "TIME: $t")
            println(headerFile, "DATA_FILE: $dataName")
            println(headerFile, "DATA_SIZE: $nx $ny $nz")
            println(headerFile, "DATA_FORMAT: FLOAT")
            println(headerFile, "VARIABLE: Pressure")
            println(headerFile, "DATA_ENDIAN: LITTLE")
            println(headerFile, "CENTERING: ZONAL")
            println(headerFile, "BRICK_ORIGIN: 0. 0. 0.")
            println(headerFile, "BRICK_SIZE: $nx $ny $nz")

        end
        dataFile = open("$directory/$dataName","w")
        write(dataFile, data)
        close(dataFile)

    end
    function writeToBOV(t::Number,nt::Int,grid::EFITGrid; directory::AbstractString="", filePrefix::AbstractString="data")
        velMag = Threads.@spawn Array(sqrt.(grid.vx[1:2:end, 1:2:end,1:2:end].^2 .+ grid.vy[1:2:end, 1:2:end,1:2:end].^2 .+ grid.vz[1:2:end, 1:2:end,1:2:end].^2))
        
        writeToBOV(fetch(velMag),t,nt,grid,directory=directory,filePrefix=filePrefix)
    end
    function readBOV(path::AbstractString)
        size = (0,0,0)
        dataPath = ""
        dir = join(split(path,"/")[1:end-1],"/")

        open(path) do bovFile
            for l in readlines(path)
                words = split(l)
                if words[1] == "DATA_SIZE:"
                    size = (parse(Int, words[2]), parse(Int, words[3]), parse(Int, words[4]))
                elseif words[1] == "DATA_FILE:"
                    dataPath = String(words[2])
                end
            end
        end
        data = Array{Float32,3}(undef, size[1],size[2],size[3])
        read!("$dir/$dataPath",data)

        return data
    end
    function readBOV!(data::AbstractArray, path::AbstractString)
        size = (0,0,0)
        dataPath = ""
        dir = join(split(path,"/")[1:end-1],"/")

        open(path) do bovFile
            for l in readlines(path)
                words = split(l)
                if words[1] == "DATA_SIZE:"
                    size = (parse(Int, words[2]), parse(Int, words[3]), parse(Int, words[4]))
                elseif words[1] == "DATA_FILE:"
                    dataPath = String(words[2])
                end
            end
        end
        read!("$dir/$dataPath",data)

    end

    
    #Hash the material ids to get a dict key
    function hashIDX(a::Integer,b::Integer,c::Integer,d::Integer)
        if a < b
            low1 = a
            high1 = b
        else 
            low1 = b
            high1 = a
        end
        if c < d
            low2 = c
            high2 = d
        else
            low2 = d
            high2 = c
        end
        if low1 < low2
            lowest = low1
            middle1 = low2
        else
            lowest = low2
            middle1 = low1
        end
        if high1 > high2
            highest = high1
            middle2 = high2
        else
            highest = high2
            middle2 = high1
        end
        if middle1 < middle2
            return Int64(lowest) + Int64(middle1) << 16 + Int64(middle2) << 32 + Int64(highest) << 48

        else
            return Int64(lowest) + Int64(middle2) << 16 + Int64(middle1) << 32 + Int64(highest) << 48
        end
    end

    const offsets = [(CartesianIndex(1,0,0),CartesianIndex(0,1,0)),(CartesianIndex(0,1,0),CartesianIndex(0,0,1)),(CartesianIndex(1,0,0),CartesianIndex(0,0,1))]
    #Calculate the material BC matrix permutations for dictionary storage
    function setAveragePerms!(dict::Dict{Int64,Matrix{Float32}}, matIDs::Array{Int16,3}, mats::Array{AnisoMat,1})
        xSize,ySize,zSize = size(matIDs)

        for N in CartesianIndices((2:xSize-1,2:ySize-1,2:zSize-1))
            for dir in offsets
                a = matIDs[N]
                b = matIDs[N+dir[1]]
                c = matIDs[N+dir[2]]
                d = matIDs[N+dir[1]+dir[2]]
                id = hashIDX(a,b,c,d)

                if !haskey(dict, id)
                    
                    dict[id] = inv((inv(mats[a].c) + inv(mats[b].c) + inv(mats[c].c) + inv(mats[d].c))/4)
                end
            end
        end
        n = length(dict)
        println("")
        println("Initialized averaging dict, hashed $n material combinations")
        println("")

        display(dict)
    end

end
