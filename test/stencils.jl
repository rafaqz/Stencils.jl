using Stencils, Test, LinearAlgebra, StaticArrays, OffsetArrays, BenchmarkTools

init = [0 0 0 1 1 1
        1 0 1 1 0 1
        0 1 1 1 1 1
        0 1 0 0 1 0
        0 0 0 0 1 1
        0 1 0 1 1 0]

win1 = SA[0, 0, 0, 0, 1, 0, 0, 0, 0]
win2 = SA[1, 1, 1, 1, 0, 1, 1, 1, 1]
win3 = SA[1, 1, 1, 0, 0, 1, 0, 0, 1]

@testset "Moore" begin
    moore = Moore{1,2}(SVector(0,1,0,0,1,0,1,1))
    @test isbits(moore)

    # Stencils.distance_zones(moore)
    @test radius(moore) == 1
    @test diameter(moore) == 3
    @test moore[1] == 0
    @test moore[8] == 1
    @test length(moore) == 8
    @test eltype(moore) == Int
    @test neighbors(moore) === SVector(0, 1, 0, 0, 1, 0, 1, 1)
    @test sum(moore) == sum(neighbors(moore)) == 4
    @test offsets(moore) == SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    @test indices(moore, (1, 1)) ==
        SVector((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2))
    sq2 = sqrt(2)
    @test distances(moore) == SVector(sq2, 1.0, sq2, 1.0, 1.0, sq2, 1.0, sq2)
    @test (@allocated distances(moore)) == 0
end

@testset "Window" begin
    @test Window{1}() == Window{1,2}()
    window = Window{1}(SVector(init[1:3, 1:3]...))
    @test isbits(window)
    @test diameter(window) == 3
    @test window[1] == 0
    @test window[2] == 1
    @test length(window) == 9
    @test eltype(window) == Int
    @test neighbors(window) isa SVector
    @test sum(window) == sum(neighbors(window)) == 4
    @test offsets(window) == SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0),
                                     (1, 0), (-1, 1), (0, 1), (1, 1))

    window2 = Stencils.rebuild(window, SVector(win2...))
    @test neighbors(window2) == SVector(win2...)

    @test sum(Window{1}(win1)) == 1
    @test sum(Window{1}(win2)) == 8
    @test sum(Window{1}(win3)) == 5
end

@testset "VonNeumann" begin
    h = VonNeumann{1}()
    A = StencilArray(init, h)
    vonneumann = Stencils.rebuild(h, neighbors(A, (2, 2))) 
    @test offsets(vonneumann) == SVector((0, -1), (-1, 0), (1, 0), (0, 1))
    @test radius(vonneumann) == 1
    @test diameter(vonneumann) == 3
    @test vonneumann[1] == 1
    @test vonneumann[2] == 0
    @test length(vonneumann) == 4
    @test eltype(vonneumann) == Int
    @test neighbors(vonneumann) == SVector(1, 0, 1, 1)
    @test sum(neighbors(vonneumann)) == sum(vonneumann) == 3
    vonneumann2 = VonNeumann{2}()
    @test offsets(vonneumann2) == SVector((0, -2), (-1, -1), (0, -1), (1, -1),
         (-2 , 0), (-1, 0), (1, 0), (2, 0),
         (-1, 1), (0, 1), (1, 1), (0, 2))
end

@testset "Positional" begin
    win = [0 1 0 0 1
           0 0 1 0 0
           0 0 0 1 1
           0 0 1 0 1
           1 0 1 0 1]
    h1 = Positional(((-1, -1), (2, -2), (2, 2), (-1, 2), (0, 0)))
    @test isbits(h1)
    @test radius(h1) == 2
    @test length(h1) == 5
    res1 = stencil(StencilArray(win, h1), (3, 3)) 
    @test neighbors(res1) == SVector(0, 1, 1, 0, 0)
    @test sum(res1) == 2

    h2 = Positional{((-1,-1), (0,-1), (1,-1), (0,0))}()
    @test radius(h2) == 1
    @test length(h2) == 4
    res2 = stencil(StencilArray(win, h2), (3, 3)) 
    @test neighbors(res2) == SVector(0, 0, 0, 0)
    @test sum(res2) == 0
end

@testset "Layered" begin
    lhood = Layered(
        Positional(((-1, -1), (1, 1)), ), Positional(((-2, -2), (2, 2)), )
    )
    @test isbits(lhood)
    @test radius(lhood) == 2
    @test offsets(lhood) == (SVector((-1, -1), (1, 1)), SVector((-2, -2), (2, 2)))
    @test indices(lhood, (1, 1)) === (SVector((0, 0), (2, 2)), SVector((-1, -1), (3, 3)))
    lhood1 = stencil(StencilArray(reshape(1:25, 5, 5), lhood), (3, 3))
    @test neighbors(lhood1) == (SVector(7, 19), SVector(1, 25))
end

@testset "Kernel" begin
    @testset "Window" begin
        kern = SVector{9}(1:9)
        Kernel(Window{1}(), kern)
        @test_throws ArgumentError Kernel(Window{2}(), kern)
        @test Kernel(Window{1,2}(), SMatrix{3,3}(reshape(1:9, 3, 3))) == 
            Kernel(SMatrix{3,3}(reshape(1:9, 3, 3)))
            Kernel(reshape(1:9, 3, 3))
        k = Kernel(Window{1,2}(kern), SMatrix{3,3}(reshape(1:9, 3, 3)))
        @test kernelproduct(k) == sum((1:9).^2)
        @test neighbors(k) == SVector{9}(1:9)
        @test offsets(k) === SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0),
                              (1, 0), (-1, 1), (0, 1), (1, 1))
        @test indices(k, (2, 2)) === SVector((1, 1), (2, 1), (3, 1), (1, 2),
                                       (2, 2), (3, 2), (1, 3), (2, 3), (3, 3))
    end
    @testset "Moore" begin
        k = Kernel(Moore{1,2}(SVector(1:4..., 6:9...)), SVector(1:4..., 6:9...))
        @test kernelproduct(k) === sum((1:4).^2) + sum((6:9).^2)
    end
    @testset "Positional" begin
        win = reshape(1:9, 3, 3)
        off = ((0,-1),(-1,0),(1,0),(0,1))
        hood = Positional{off,1,2,4,}()
        vals = SVector(map(I -> win[I...], indices(hood, (2, 2))))
        k = Stencils.rebuild(Kernel(hood, 1:4), vals)
        @test kernelproduct(k) === 1 * 2 + 2 * 4 + 3 * 6 + 4 * 8 === 60
    end
end
