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
    moore = Moore{1,2}(SVector(0,1,0,0,1,0,1,1), 1)
    @test isbits(moore)

    # Stencils.distance_zones(moore)
    @test radius(moore) == 1
    @test diameter(moore) == 3
    @test moore[1] == 0
    @test moore[8] == 1
    @test length(moore) == 8
    @test eltype(moore) == Int
    @test neighbors(moore) === SVector(0, 1, 0, 0, 1, 0, 1, 1)
    @test center(moore) == 1
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
    window = Window{1}(SVector(init[1:3, 1:3]...), 1)
    @test isbits(window)
    @test diameter(window) == 3
    @test window[1] == 0
    @test window[2] == 1
    @test length(window) == 9
    @test eltype(window) == Int
    @test neighbors(window) isa SVector
    @test center(window) == 1
    @test sum(window) == sum(neighbors(window)) == 4
    @test offsets(window) == SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0),
                                     (1, 0), (-1, 1), (0, 1), (1, 1))

    window2 = Stencils.rebuild(window, SVector(win2...), 1)
    @test neighbors(window2) == SVector(win2...)
    @test center(window2) == 1
    @test sum(Window{1}(win1, 1)) == 1
    @test sum(Window{1}(win2, 1)) == 8
    @test sum(Window{1}(win3, 1)) == 5
end

@testset "VonNeumann" begin
    h = VonNeumann{1}()
    A = StencilArray(init, h)
    vonneumann = Stencils.rebuild(h, neighbors(A, (2, 2)), 1) 
    @test offsets(vonneumann) == SVector((0, -1), (-1, 0), (1, 0), (0, 1))
    @test radius(vonneumann) == 1
    @test diameter(vonneumann) == 3
    @test vonneumann[1] == 1
    @test vonneumann[2] == 0
    @test length(vonneumann) == 4
    @test eltype(vonneumann) == Int
    @test neighbors(vonneumann) == SVector(1, 0, 1, 1)
    @test center(vonneumann) == 1
    @test sum(neighbors(vonneumann)) == sum(vonneumann) == 3
    vonneumann2 = VonNeumann{2}()
    @test offsets(vonneumann2) == SVector((0, -2), (-1, -1), (0, -1), (1, -1),
         (-2 , 0), (-1, 0), (1, 0), (2, 0),
         (-1, 1), (0, 1), (1, 1), (0, 2))
end

@testset "Annulus" begin
    h = Annulus{1}()
    A = StencilArray(init, h)
    annulus = Stencils.rebuild(h, neighbors(A, (2, 2)), 0)
    @test offsets(annulus) == SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1),
        (0, 1), (1, 1))
    @test radius(annulus) == 1
    @test diameter(annulus) == 3
    @test annulus[1] == 0
    @test annulus[2] == 1
    @test length(annulus) == 8
    @test eltype(annulus) == Int
    @test neighbors(annulus) == SVector(0, 1, 0, 0, 1, 0, 1, 1)
    @test center(annulus) == 0
    @test sum(neighbors(annulus)) == sum(annulus) == 4
    annulus2 = Annulus{2}()
    @test offsets(annulus2) == SVector((-1, -2), (0, -2), (1, -2), (-2, -1), (2, -1),
        (-2, 0), (2, 0), (-2, 1), (2, 1), (-1, 2), (0, 2), (1, 2))
end

@testset "Ordinal" begin
    h = Ordinal{1}()
    A = StencilArray(init, h)
    ordinal = Stencils.rebuild(h, neighbors(A, (2, 2)), 0)
    @test offsets(ordinal) == SVector((-1, -1), (1, -1), (-1, 1), (1, 1))
    @test radius(ordinal) == 1
    @test diameter(ordinal) == 3
    @test ordinal[1] == 0
    @test ordinal[4] == 1
    @test length(ordinal) == 4
    @test eltype(ordinal) == Int
    @test neighbors(ordinal) == SVector(0, 0, 0, 1)
    @test center(ordinal) == 0
    @test sum(neighbors(ordinal)) == sum(ordinal) == 1
    ordinal2 = Ordinal{2}()
    @test offsets(ordinal2) == SVector((-2, -2), (2, -2), (-2, 2), (2, 2))
end

@testset "Cardinal" begin
    h = Cardinal{1}()
    A = StencilArray(init, h)
    cardinal = Stencils.rebuild(h, neighbors(A, (2, 2)), 0)
    @test offsets(cardinal) == SVector((0, -1), (-1, 0), (1, 0), (0, 1))
    @test radius(cardinal) == 1
    @test diameter(cardinal) == 3
    @test cardinal[1] == 1
    @test cardinal[2] == 0
    @test length(cardinal) == 4
    @test eltype(cardinal) == Int
    @test neighbors(cardinal) == SVector(1, 0, 1, 1)
    @test center(cardinal) == 0
    @test sum(neighbors(cardinal)) == sum(cardinal) == 3
    cardinal2 = Cardinal{2}()
    @test offsets(cardinal2) == SVector((0, -2), (-2, 0), (2, 0), (0, 2))
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
    @test center(res1) == 0
    @test sum(res1) == 2

    h2 = Positional{((-1,-1), (0,-1), (1,-1), (0,0))}()
    @test radius(h2) == 1
    @test length(h2) == 4
    res2 = stencil(StencilArray(win, h2), (3, 3)) 
    @test neighbors(res2) == SVector(0, 0, 0, 0)
    @test center(res2) == 0
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
    @test @inferred center(res1) == 0
    @test sum(res1) == 2

    A3 = cat(A, A, A; dims=3)
    h3 = Rectangle{((-1, 0), (0, 1), (1, 1))}()
    @test radius(h3) == 1
    @test length(h3) == 2 * 2 * 1
    sa = StencilArray(A3, h3)
    res3 = stencil(sa, (2, 2, 1)) 
    @test @inferred neighbors(res3) == SVector(1, 0, 0, 1)
    @test @inferred center(res3) == 0
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
    @test center(res1) == 0
    @test sum(res1) == 3
    @test res1.n == 1
    @test res1.e == 0

    @test mapstencil(StencilArray(win, h1)) do s
        s.n + s.w + center(s)
    end == [
        0 0 1 0 0
        0 1 0 1 2
        0 0 2 0 1
        1 0 1 1 2
        0 0 1 0 1
    ] + win
    # Shorthand syntax for naming another stencil
    @test h1 == NamedStencil{(:n,:e,:w,:s)}(VonNeumann())
    @test_throws ArgumentError NamedStencil{(:n,:s)}(VonNeumann())

    # Cardinal and Ordinal (fixed wind directions) can be used as NamedStencils
    ns = NamedStencil(Cardinal(1))
    @test mapstencil(StencilArray(win, ns)) do s
        s.W + s.S
    end == [
        1 0 0 1 0
        0 2 0 0 1
        0 0 2 1 0
        0 1 0 2 1
        0 1 1 1 1
    ]

    ns = NamedStencil(Ordinal(1))
    @test mapstencil(StencilArray(win, ns)) do s
        s.NE + s.NW
    end == [
        0 1 0 1 0
        0 0 1 1 1
        0 1 0 2 0
        0 2 0 2 0
        0 0 0 0 0
    ]
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
    @test center(layered_filled) == (13, 13)
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
        k = Kernel(Window{1,2}(kern, 5), SMatrix{3,3}(reshape(1:9, 3, 3)))
        @test kernelproduct(k) == sum((1:9).^2)
        @test neighbors(k) == SVector{9}(1:9)
        @test center(k) == 5
        @test offsets(k) === SVector((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0),
                              (1, 0), (-1, 1), (0, 1), (1, 1))
        @test indices(k, (2, 2)) === SVector((1, 1), (2, 1), (3, 1), (1, 2),
                                       (2, 2), (3, 2), (1, 3), (2, 3), (3, 3))
    end
    @testset "Moore" begin
        vals = SVector(1:4..., 6:9...)
        k = Kernel(Moore{1,2}(vals, 4), vals)
        @test center(k) == 4
        @test kernelproduct(k) === sum(vals .^ 2)
        # Nested arrays work too
        vals2 = map(x -> SVector((x, 2x)), vals)
        k2 = Kernel(Moore{1,2}(vals2, SVector(4,8)), vals)
        @test center(k2) == SVector(4, 8)
        @test kernelproduct(k2) === sum(map((v2, v) -> v2 .* v, vals2, vals))
    end
    @testset "Positional" begin
        win = reshape(1:9, 3, 3)
        off = ((0,-1),(-1,0),(1,0),(0,1))
        hood = Positional{off,1,2,4,}()
        vals = SVector(map(I -> win[I...], indices(hood, (2, 2))))
        k = Stencils.rebuild(Kernel(hood, 1:4), vals, 1)
        @test center(k) == 1
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

@testset "Merge" begin
    @testset "Positional" begin
        p = merge(Horizontal())
        @test p === Horizontal()

        p = merge(Horizontal(), Vertical())
        @test issorted(offsets(p))
        @test length(p) == 5  # center was merged

        @test_throws ArgumentError merge(Horizontal(1,3), Horizontal())
    end
    @testset "Named" begin
        ns1 = NamedStencil(; west=(0, -1), north=(1, 0))
        @test merge(ns1) === ns1

        ns2 = NamedStencil(; south=(-1, 0), east=(0, 1))
        ns = merge(ns1, ns2)
        @test length(ns) == 4

        windrose = merge(NamedStencil(Cardinal(3)), NamedStencil(Ordinal(2)))
        @test length(windrose) == 8

        ns3 = NamedStencil(; center=(0, 0))
        ns = merge(ns1, ns2, ns3)
        @test length(ns) == 5

        @test_throws ArgumentError merge(ns1, ns1)
    end
end
