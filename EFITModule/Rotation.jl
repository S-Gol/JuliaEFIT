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
    M[2,:] .= (0,c^2, s^2, 2*c*s,0,0)
    M[3,:] .= (0,s^2, c^2, -2*c*s,0,0)
    M[4,:] .= (0,-c*s, c*s, c^2-s^2,0,0)
    M[5,:] .= (0,0,0,0,c,-s)
    M[6,:] .= (0,0,0,0,s,c)
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
    M[1,:] .= (c^2,s^2,0,0,0,2*c*s)
    M[2,:] .= (s^2,c^2,0,0,0,-2*c*s)
    M[3,:] .= (0,0,1,0,0,0)
    M[4,:] .= (0,0,0,c,s,0)
    M[5,:] .= (0,0,0,-s,c,0)
    M[6,:] .= (-c*s,c*s,0,0,0,c^2-s^2)
    return M
end