using Revise
using GLMakie
using Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using NPZ
include("EFITModule/EFIT.jl")

USE_GPU = false
ParallelStencil.@reset_parallel_stencil()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float32, 3)
else
    @init_parallel_stencil(Threads, Float32, 3)
end

#Halkjaer 5.6
dx = 10^-4
dt = 0.5*10^-8

L = 90e-3
H = 30e-3

sourceWidth = 26.6e-3

nx = round(Integer, L/dx)
nz = round(Integer, H/dx)


ny = nx

sx = round(Integer, L/(2*dx))
sy = round(Integer, 0.5*ny)
ns = round(Integer,0.5*sourceWidth/dx)

#Create a grid of integer material indices 
matGrid = ones(Int16, nx,ny,nz)
#Stress-free edges
th = 3
matGrid[1:th,:,:] .= 2
matGrid[(nx-th):nx,:,:] .= 2

matGrid[:,:,1:th] .= 2
matGrid[:,:,(nz-th):nz] .= 2


#Use anisotropic stiffness matrices for isotropic materials 
materials = [Main.EFIT.IsoMats["Austenite"],Main.EFIT.IsoMat(10,5,10)]

grid = Main.EFIT.EFITGrid(matGrid,materials,dt,dx);

f = 2e6
w = 2*pi*f
n = 2
#Create a source function from which input can be taken
function source(t)
    if t < n*2*pi/w
        v = cos(w*t)*(1-cos(w*t/n))*0.01
    else
        v=0
    end

    return Data.Number(v)
end

# animation settings0
framerate = 30

tIterator = 0:grid.dt:5e-6

fig = Figure(resolution = (900, 300))
fig[1,1] = Axis(fig)
plt = heatmap!(fig[1,1],Array(grid.vx[1:nx,2,1:nz]),colorrange = (-75, 0),colormap=:grayC)

#Step the simulation
function stepSim(t)
    println(t)
    #Isotropic step function
    Main.EFIT.SimStep!(grid)
    #Apply the source to the Z-direction
    @parallel ((sx-ns):(sx+ns),(sy-ns):(sy+ns),(nz-th-2):nz) Main.EFIT.applySource!(grid.vx,grid.vy,grid.vz, Data.Number(0.0), Data.Number(0.0), (source(t)))
    #=
    grid.vx[:,1,:] .= grid.vx[:,3,:] .= grid.vx[:,2,:]
    grid.vy[:,1,:] .= grid.vy[:,3,:] .= grid.vy[:,2,:]
    grid.vz[:,1,:] .= grid.vz[:,3,:] .= grid.vz[:,2,:]

    grid.σxx[:,1,:] .= grid.σxx[:,3,:] .= grid.σxx[:,2,:]
    grid.σyy[:,1,:] .= grid.σyy[:,3,:] .= grid.σyy[:,2,:]
    grid.σzz[:,1,:] .= grid.σzz[:,3,:] .= grid.σzz[:,2,:]

    grid.σxz[:,1,:] .= grid.σxz[:,3,:] .= grid.σxz[:,2,:]
    grid.σyz[:,1,:] .= grid.σyz[:,3,:] .= grid.σyz[:,2,:]
    grid.σxy[:,1,:] .= grid.σxy[:,3,:] .= grid.σxy[:,2,:]
    =#

    velMag = sqrt.(grid.vx[:,round(Int,end/2),:].^2 .+ grid.vy[:,round(Int,end/2),:].^2 .+ grid.vz[:,round(Int,end/2),:].^2)
    p0 = maximum(velMag)

    #Update the plot
    plt[1] = 10*log10.(velMag./p0)
end



record(stepSim, fig, "color_animation.mp4", tIterator; framerate = 30)

#Store the result as a brick of values file for LLNL VisIT
#writeResult = Main.EFIT.writeToBOV(1.0,100, grid,directory="A://")




