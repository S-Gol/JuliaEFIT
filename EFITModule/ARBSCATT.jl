module ARBSCATT
    function loadARBSCATT(dir::AbstractString)
        #Load the header file
        f = open("$dir/in.file")
        inData = readline(f)
        close(f)

        #Readable names for header params
        paramNames = ["nz","ny","nx","ds","dt","ρ","λ","μ","maxT","outputN"]
        
        header = split(inData," ")
        deleteat!(header, findall(x->x=="",header))
        paramDict = Dict()
        for i in 1:3
            paramDict[paramNames[i]]=parse(Int, string(header[i]))
        end
        for i in 4:length(paramNames)
            paramDict[paramNames[i]]=parse(Float32, string(header[i]))
        end
        
        nx = paramDict["nx"]
        ny = paramDict["ny"]
        nz = paramDict["nz"]

        #Load the ARBSCATT file itself
        v = Array{Int16}(undef, nx, ny, nz)
        @inbounds for (i, c) in enumerate(@view read("$dir/ARBSCATT.file")[2:2:end])
            v[i] = Int16(c - 0x30)
        end
        return (v,paramDict)
    end
    struct Transducer
        z::Int
        y::Int
        x::Int
        rad::Int
        drivelen::Int
        theta::Number
        phi::Number
        drivef
    end
    
    function loadTransducers(dir::AbstractString)
        #Load the header file
        f = open("$dir/trans.file")
        transData = split(readline(f))
        close(f)

        curStart = 2
        baseLength = 7
        
        nt = parse(Int, transData[1])
        transducers = Array{Transducer,1}(undef, nt)

        for i in 1:nt
            z=parse(Int,transData[curStart])
            y=parse(Int,transData[curStart+1])
            x=parse(Int,transData[curStart+2])
            rad=parse(Int,transData[curStart+3])
            len=parse(Int,transData[curStart+4])
            θ=parse(Float32,transData[curStart+5])
            ϕ=parse(Float32,transData[curStart+6])

            curStart += baseLength
            drivef = parse.(Float32, transData[curStart:curStart+len-1])
            curStart+=len

            transducers[i] = Transducer(z,y,x,rad,len,θ,ϕ,drivef)
            return transducers
        end


    end

end