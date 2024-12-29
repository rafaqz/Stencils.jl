using Stencils, Test, LinearAlgebra, StaticArrays, BenchmarkTools, Adapt

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
    moore = Moore{1,2}(1, SVector(0,1,0,0,1,0,1,1))
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
    window = Window{1}(1, SVector(init[1:3, 1:3]...))
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

    window2 = Stencils.rebuild(window, 1, SVector(win2...))
    @test neighbors(window2) == SVector(win2...)

    @test sum(Window{1}(1, win1)) == 1
    @test sum(Window{1}(1, win2)) == 8
    @test sum(Window{1}(1, win3)) == 5
end

@testset "VonNeumann" begin
    h = VonNeumann{1}()
    A = StencilArray(init, h)
    vonneumann = Stencils.rebuild(h, 1, neighbors(A, (2, 2))) 
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

@testset "Rectangle" begin
    A = [0 1 0 0 1
         0 0 1 0 0
         0 0 0 1 1
         0 0 1 0 1
         1 0 1 0 1]
    h1 = Rectangle(((-1, 0), (-2, 1)))
    @test isbits(h1)
    @test radius(h1) == 2
    @test length(h1) == 8
    sa = StencilArray(A, h1)
    res1 = stencil(sa, (3, 3)) 
    indices(h1, (3, 3)) 
    @test @inferred neighbors(res1) == SVector(0, 0, 0, 0, 1, 0, 0, 1)
    @test sum(res1) == 2

    A3 = cat(A, A, A; dims=3)
    h3 = Rectangle{((-1, 0), (0, 1), (1, 1))}()
    @test radius(h3) == 1
    @test length(h3) == 2 * 2 * 1
    sa = StencilArray(A3, h3)
    res3 = stencil(sa, (2, 2, 1)) 
    @test @inferred neighbors(res3) == SVector(1, 0, 0, 1)
    @test sum(res3) == 2
end

@testset "NamedStencil" begin
    win = [0 1 0 0 1
           0 0 1 0 0
           0 0 0 1 1
           0 0 1 0 1
           1 0 1 0 1]

    h1 = NamedStencil(n=(-1, 0), e=(0, -1), w=(1, 0), s=(0, 1))
    @test isbits(h1)
    @test radius(h1) == 1
    @test length(h1) == 4
    res1 = stencil(StencilArray(win, h1), (3, 3)) 
    @test neighbors(res1) === SVector(1, 0, 1, 1)
    @test sum(res1) == 3
    @test res1.n == 1
    @test res1.e == 0

    @test mapstencil(StencilArray(win, h1)) do s
        s.n + s.w
    end == [
        0 0 1 0 0
        0 1 0 1 2
        0 0 2 0 1
        1 0 1 1 2
        0 0 1 0 1
    ]
    # Shorthand syntax for naming another stencil
    @test h1 == NamedStencil{(:n,:e,:w,:s)}(VonNeumann())
    @test_throws ArgumentError NamedStencil{(:n,:s)}(VonNeumann())
end

@testset "Layered" begin
    layered = Layered(
        Positional(((-1, -1), (1, 1)), ), Positional(((-2, -2), (2, 2)), )
    )
    @test isbits(layered)
    @test radius(layered) == 2
    @test offsets(layered) == (SVector((-1, -1), (1, 1)), SVector((-2, -2), (2, 2)))
    @test indices(layered, (1, 1)) === (SVector((0, 0), (2, 2)), SVector((-1, -1), (3, 3)))
    A = StencilArray(collect(reshape(1:25, 5, 5)), layered)
    layered_filled = stencil(A, (3, 3))
    @test neighbors(layered_filled) == (SVector(7, 19), SVector(1, 25))
    mapstencil(A) do l
        sum(l[1]) - sum(l[2])
    end
    l1 = Layered(; a=Positional(((-1, -1), (1, 1)), ), b=Positional(((-2, -2), (2, 2))))
    l2 = Layered(; a=Positional(((-1, -1), (1, 1)), ), b=Positional(((-2, -2), (2, 2))))
    multi_layered = Layered(; l1, l2)
    A = StencilArray(collect(reshape(1:25, 5, 5)), multi_layered)
    ml_filled = stencil(A, (3, 3))
    @test ml_filled.l2.a == [7, 19]
    mapstencil(A) do l
        sum(l.l1.b) - sum(l.l2.a)
    end
end

@testset "Kernel" begin
    @testset "Window" begin
        kern = SVector{9}(1:9)
        Kernel(Window{1}(), kern)
        @test_throws ArgumentError Kernel(Window{2}(), kern)
        @test Kernel(Window{1,2}(), SMatrix{3,3}(reshape(1:9, 3, 3))) == 
            Kernel(SMatrix{3,3}(reshape(1:9, 3, 3)))
            Kernel(reshape(1:9, 3, 3))
        k = Kernel(Window{1,2}(5, kern), SMatrix{3,3}(reshape(1:9, 3, 3)))
        @test kernelproduct(k) == sum((1:9).^2)
        @test neighbors(k) == SVector{9}(1:9)
        @test offsets(k) === SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0),
                              (1, 0), (-1, 1), (0, 1), (1, 1))
        @test indices(k, (2, 2)) === SVector((1, 1), (2, 1), (3, 1), (1, 2),
                                       (2, 2), (3, 2), (1, 3), (2, 3), (3, 3))
    end
    @testset "Moore" begin
        vals = SVector(1:4..., 6:9...)
        k = Kernel(Moore{1,2}(4, vals), vals)
        @test kernelproduct(k) === sum(vals .^ 2)
        # Nested arrays work too
        vals2 = map(x -> SVector((x, 2x)), vals)
        k2 = Kernel(Moore{1,2}(4, vals2), vals)
        @test kernelproduct(k2) === sum(map((v2, v) -> v2 .* v, vals2, vals))
    end
    @testset "Positional" begin
        win = reshape(1:9, 3, 3)
        off = ((0,-1),(-1,0),(1,0),(0,1))
        hood = Positional{off,1,2,4,}()
        vals = SVector(map(I -> win[I...], indices(hood, (2, 2))))
        k = Stencils.rebuild(Kernel(hood, 1:4), 1, vals)
        @test kernelproduct(k) === 1 * 2 + 2 * 4 + 3 * 6 + 4 * 8 === 60
    end
    @testset "Adapt" begin
        kern = collect(1:9)
        st = Kernel(Window{1}(), kern)
        # Adapt to a SVector, which will be `isbits`
        @test !isbits(st)
        @test isbits(Adapt.adapt(SVector{9}, st))
    end
end
