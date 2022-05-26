using LinearAlgebra


"""
Rotation matrix for a 6x6 tensor rotated about x,y,z degrees
"""
function rotateMatrix(c,x,y,z)
    xT = deg2rad(x)
    yT = deg2rad(y)
    zT = deg2rad(z)

    Mx = xRotMatrix(xT)
    My = yRotMatrix(yT)
    Mz = zRotMatrix(zT)

    return Symmetric(Mz * (My * (Mx * c * transpose(Mx)) * Transpose(My)) * Transpose(Mz))
end
#http://solidmechanics.org/Text/Chapter3_2/Chapter3_2.php
function xRotMatrix(x::Number)
    c = cos(x)
    s = sin(x)
    M = zeros(Float32,6,6)
    M[1,:] .= (1,0,0,0,0,0)
    M[2,:] .= (0,c^2, s^2, -2*c*s,0,0)
    M[3,:] .= (0,s^2, c^2, 2*c*s,0,0)
    M[4,:] .= (0,c*s, -c*s, c^2-s^2,0,0)
    M[5,:] .= (0,0,0,0,c,s)
    M[6,:] .= (0,0,0,0,-s,c)
    return M
end
function yRotMatrix(y::Number)
    M = zeros(Float32,6,6)
    c = cos(y)
    s = sin(y)
    M[1,:] .= (c^2,0,s^2,0,2*c*s,0)
    M[2,:] .= (0, 1, 0, 0,0,0)
    M[3,:] .= (s^2,0,c^2,0,-2*c*s,0)
    M[4,:] .= (0,0,0,c,0,-s)
    M[5,:] .= (-c*s,0,c*s,0,c^2-s^2,0)
    M[6,:] .= (0,0,0,s,0,c)
    return M
end
function zRotMatrix(z::Number)
    M = zeros(Float32,6,6)
    c = cos(z)
    s = sin(z)
    M[1,:] .= (c^2,s^2,0,0,0,-2*c*s)
    M[2,:] .= (s^2,c^2,0,0,0,2*c*s)
    M[3,:] .= (0,0,1,0,0,0)
    M[4,:] .= (0,0,0,c,s,0)
    M[5,:] .= (0,0,0,-s,c,0)
    M[6,:] .= (c*s,-c*s,0,0,0,c^2-s^2)
    return M
end


voigtIndexes =[(1,1),(2,2),(3,3),(2,3),(1,3),(1,2)]
function voigtIDX(i,j)
    @assert(i<=3 && i > 0)
    @assert(j<=3 && j > 0)
    if i==j
        return i
    elseif min(i,j) == 2
        return 4
    elseif min(i,j) == 1 && max(i,j) == 3
        return 5
    elseif min(i,j) == 1 && max(i,j) == 2
        return 6
    else
        throw(error("Invalid voigt"))
        return 1
    end
end

function voigtToFull(C)
    cOut = zeros(3,3,3,3)
    for i in 1:3
        for j in 1:3
            for k in 1:3
                for l in 1:3
                    voigtI = voigtIDX(i,j)
                    voigtJ = voigtIDX(k,l)
                    cOut[i,j,k,l] = C[voigtI,voigtJ]
                end
            end
        end
    end
    return cOut
end

function fullToVoigt(C)
    @assert(size(C)==(3,3,3,3),"Incorrect size in call to fullToVoigt()")
    cOut = zeros(6,6)
    for i in 1:6
        for j in 1:6
            k,l = voigtIndexes[i]
            m,n = voigtIndexes[j]
            cOut[i,j] = C[k,l,m,n]
        end
    end
    return cOut
end

function rotMatrix(c,x,y,z;degrees=true)
    if !degrees
        tx = x
        ty = y
        tz = z
    else
        tx = deg2rad(x)
        ty = deg2rad(y)
        tz = deg2rad(z)
    end

    rx = [1 0 0; 0 cos(tx) (-sin(tx)); 0 sin(tx) cos(tx)]
    ry = [cos(ty) 0 sin(ty); 0 1 0; (-sin(ty)) 0 cos(ty) ]
    rz = [cos(tz) (-sin(tz)) 0; sin(tz) cos(tz) 0; 0 0 1]
    r = rz*ry*rx

    return rotMatrix(c,r)


end
#https://www.researchgate.net/profile/Abdalrhaman-Koko/post/How_to_rotate_an_anisotropic_stiffness_matrix_to_an_orthotropic_matrix_of_cancellous_bone/attachment/60b8c0eb5e24cd000161f769/AS%3A1030579663433730%401622720747470/download/Document1.pdf
function rotMatrix(C,r; checkEigen = false, tol = 1e-3)
    T = [
        r[1,1]^2 r[1,2]^2 r[1,3]^2 2*r[1,1]*r[1,2] 2*r[1,2]*r[1,3] 2*r[1,3]*r[1,1];
        r[2,1]^2 r[2,2]^2 r[2,3]^2 2*r[2,1]*r[2,2] 2*r[2,2]*r[2,3] 2*r[2,3]*r[2,1];
        r[3,1]^2 r[3,2]^2 r[3,3]^2 2*r[3,1]*r[3,2] 2*r[3,2]*r[3,3] 2*r[3,3]*r[3,1];
        r[1,1]*r[2,1] r[1,2]*r[2,2] r[1,3]*r[2,3] r[1,1]*r[2,2]+r[2,1]*r[1,2] r[1,2]*r[2,3] + r[1,3]*r[2,2] r[1,3]*r[2,1]+r[2,3]*r[1,1];
        r[2,1]*r[3,1] r[2,2]*r[3,2] r[3,3]*r[2,3] r[3,2]*r[2,1]+r[3,1]*r[2,2] r[2,2]*r[3,3] + r[2,3]*r[3,2] r[2,3]*r[3,1]+r[2,1]*r[3,3];
        r[1,1]*r[3,1] r[3,2]*r[1,2] r[3,3]*r[1,3] r[3,1]*r[1,2]+r[1,1]*r[3,2] r[3,2]*r[1,3] + r[1,2]*r[3,3] r[3,3]*r[1,1]+r[1,3]*r[3,1]]



    T[4:6,1:3] .= T[4:6,1:3].*sqrt(2)
    T[1:3,4:6] .= T[1:3,4:6]./sqrt(2)

    cPrime = inv(T)*C*T
    if checkEigen
        eigs = (eigvals(C),eigvals(cPrime))
        I1 = [eigs[i][1]+eigs[i][2]+eigs[i][3] for i in 1:2]
        I2 = [eigs[i][1]*eigs[i][2]+eigs[i][1]*eigs[i][3]+eigs[i][3]*eigs[i][2] for i in 1:2]
        I3 = [eigs[i][1]*eigs[i][2]*eigs[i][3] for i in 1:2]
        Is = (I1,I2,I3)
        for I in Is
            @assert(abs((I[1]-I[2])/I[1])<tol,"Eigenvalue error in rotation")
        end

        for i in 1:3
            for j in i:3
                @assert(abs((cPrime[i,j]-cPrime[j,i])/cPrime[i,j] < tol),"Symmetry error in rotation")
            end
        end
    end
    
    return cPrime
    #return fullToVoigt(rotTensor(voigtToFull(C),r))
end
function rotTensor(C,g)
    Tprime = zeros(3,3,3,3)
    
    for k = 1:3
        for l = 1:3
            for s = 1:3
                for t = 1:3

                    Tprime[k,l,s,t] = 0

                    for m = 1:3
                        for n = 1:3
                            for p = 1:3
                                for r = 1:3

                                    Tprime[k,l,s,t] += c[m,n,p,r] * T[k, m] * T[l, n] * T[s, p] * T[t, r]

                                end
                            end
                        end
                    end

                end
            end
        end
    end
    #=
    for i in 1:3
        for j in 1:3
            for k in 1:3
                for l in 1:3
                    for a in 1:3
                        for b in 1:3
                            for c in 1:3
                                for d in 1:3
                                    Tprime[i, j, k, l] +=  g[i, a] * g[j, b] * g[k, c] * g[l, d] * C[a, b, c, d]
                                end
                            end
                        end
                    end
                end
            end
        end
    end=#
    return Tprime
end


