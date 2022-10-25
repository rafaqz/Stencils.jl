using Neighborhoods, Test, LinearAlgebra, StaticArrays, OffsetArrays, BenchmarkTools

init = [0 0 0 1 1 1
        1 0 1 1 0 1
        0 1 1 1 1 1
        0 1 0 0 1 0
        0 0 0 0 1 1
        0 1 0 1 1 0]

win1 = [0 0 0
        0 1 0
        0 0 0]
win2 = [1 1 1
        1 0 1
        1 1 1]
win3 = [1 1 1
        0 0 1
        0 0 1]

@testset "Moore" begin
    moore = Moore{1}(SVector(0,1,0,0,1,0,1,1))

    # Neighborhoods.distance_zones(moore)
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
    @test Window{1}() == Window(1) == Window(zeros(3, 3))
    window = Window{1}(SVector(init[1:3, 1:3]...))
    @test diameter(window) == 3
    @test window[1] == 0
    @test window[2] == 1
    @test length(window) == 9
    @test eltype(window) == Int
    @test neighbors(window) isa SVector
    @test sum(window) == sum(neighbors(window)) == 4
    @test offsets(window) == SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0),
                                     (1, 0), (-1, 1), (0, 1), (1, 1))

    window2 = Neighborhoods.setneighbors(window, SVector(win2...))
    @test neighbors(window2) == SVector(win2...)

    @test sum(Window{1}(win1)) == 1
    @test sum(Window{1}(win2)) == 8
    @test sum(Window{1}(win3)) == 5
end

@testset "VonNeumann" begin
    h = VonNeumann{1}()
    A = NeighborhoodArray(init, h)
    vonneumann = Neighborhoods.setneighbors(h, neighbors(A, (2, 2))) 
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
    h1 = Positional(((-1,-1), (2,-2), (2,2), (-1,2), (0, 0)))
    custom1 = neighborhood(NeighborhoodArray(win, h1), (3, 3)) 
    h2 = Positional{((-1,-1), (0,-1), (1,-1), (2,-1), (0,0))}()
    custom2 = neighborhood(NeighborhoodArray(win, h2), (3, 3)) 
    l = Layered((Positional((-1,1), (-2,2)), Positional((1,2), (2,2), (0, 2))))
    layered = neighborhood(NeighborhoodArray(win, l), (3, 3)) 

    @test neighbors(custom1) == SVector(0, 1, 1, 0, 0)
    @test sum(custom1) == 2
    @test sum(custom2) == 0
    @test map(sum, layered) == (1, 3)
    @test offsets(layered) == (SVector((-1, 1), (-2, 2)), SVector((1, 2), (2, 2), (0, 2)))
end

@testset "Layered" begin
    lhood = Layered(
        Positional(((-1, -1), (1, 1)), ), Positional(((-2, -2), (2, 2)), )
    )
    @test radius(lhood) == ((-2, 2), (-2, 2))
    @test offsets(lhood) == (SVector((-1, -1), (1, 1)), SVector((-2, -2), (2, 2)))
    @test indices(lhood, (1, 1)) === (SVector((0, 0), (2, 2)), SVector((-1, -1), (3, 3)))
    lhood1 = neighborhood(NeighborhoodArray(reshape(1:25, 5, 5), lhood), (3, 3))
    @test neighbors(lhood1) == (SVector(7, 19), SVector(1, 25))
end

@testset "Kernel" begin
    win = reshape(1:9, 3, 3)
    @testset "Window" begin
        mat = zeros(3, 3)
        @test Kernel(mat) == Kernel(Window(1), mat)
        @test_throws ArgumentError Kernel(Window(2), mat)
        k = Kernel(Window{1,2,9,typeof(win)}(win), SMatrix{3,3}(reshape(1:9, 3, 3)))
        @test kernelproduct(k) == sum((1:9).^2)
        @test neighbors(k) == reshape(1:9, 3, 3)
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
        off = ((0,-1),(-1,0),(1,0),(0,1))
        hood = Positional{off,1,2,4,}()
        vals = SVector(map(I -> win[I...], indices(hood, (2, 2))))
        k = Neighborhoods.setneighbors(Kernel(hood, 1:4), vals)
        @test kernelproduct(k) === 1 * 2 + 2 * 4 + 3 * 6 + 4 * 8 === 60
    end
end
