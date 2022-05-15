using GLMakie

cube = [Point3f(-0.5,-0.5,-0.5),Point3f(-0.5,0.5,-0.5),Point3f(-0.5,-0.5,0.5),Point3f(-0.5,0.5,0.5),
Point3f(0.5,-0.5,-0.5),Point3f(0.5,0.5,-0.5),Point3f(0.5,-0.5,0.5),Point3f(0.5,0.5,0.5)]
cubeLines = [[1,2],[1,3],[3,4],[2,4],[5,6],[5,7],[7,8],[8,6],[1,5],[2,6],[3,7],[4,8]]

ax, fig, plt = scatter(cube)


for p in cubeLines
    lines!(cube[p] )
end
text!(L"\sigma", position=Point3f(0,0,0))
display(ax)