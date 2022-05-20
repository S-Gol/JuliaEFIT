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
    Mz = yRotMatrix(zT)

    return Symmetric(Mz * My * Mx * c * transpose(Mx) * Transpose(My) * Transpose(Mz))
end
function xRotMatrix(x::Number)
    M = zeros(Float32,6,6)
    M[1,:] .= (1,0,0,0,0,0)
    M[2,:] .= (0,cos(x)^2, sin(x)^2, sin(2*x),0,0)
    M[3,:] .= (0,sin(x)^2, cos(x)^2, -sin(2*x),0,0)
    M[4,:] .= (0,-sin(2*x)/2, sin(2*x)/2, cos(2*x),0,0)
    M[5,:] .= (0,0,0,0,cos(x),-sin(x))
    M[6,:] .= (0,0,0,0,sin(x),cos(x))
    return M
end
function yRotMatrix(y::Number)
    M = zeros(Float32,6,6)
    M[1,:] .= (cos(y)^2,0,sin(y)^2,0,sin(2*y),0)
    M[2,:] .= (0, 1, 0, 0,0,0)
    M[3,:] .= (sin(y)^2, 0, cos(y)^2, 0, -sin(2*y),0)
    M[4,:] .= (0,0,0,cos(y),0,-sin(y))
    M[5,:] .= (-sin(2*y)/2,0,sin(2*y/2),0,cos(2*y),0)
    M[6,:] .= (0,0,0,sin(y),0,cos(y))
    return M
end
function zRotMatrix(z::Number)
    M = zeros(Float32,6,6)
    M[1,:] = (cos(z)^2, sin(z)^2, 0,0,0,-sin(2*z))
    M[2,:] = (sin(z)^2, cos(z)^2, 0,0,0, sin(2*z))
    M[3,:] = (0,0,1,0,0,0)
    M[4,:] = (0,0,0,cos(z),sin(z),0)
    M[5,:] = (0,0,0,-sin(z),cos(z),0)
    M[6,:] = (sin(2*z)/2, -sin(2*z)/2, 0,0,0,cos(2*z))
    return M
end