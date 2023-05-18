using Stencils, Test, LinearAlgebra, StaticArrays, OffsetArrays

@testset "StencilArray" begin

    @testset "1d" begin
        r = rand(100)
        A = StencilArray(r, VonNeumann{3,1}(); padding=Conditional(), boundary=Remove(0.0));
        B = StencilArray(r, Window{10,1}(); padding=Halo{:out}(), boundary=Remove(0.0));
        C = StencilArray(r, Moore{10,1}(); padding=Halo{:in}(), boundary=Wrap());
        @test size(A) == size(parent(A)) == (100,)
        @test size(B) == (100,)
        @test size(parent(B)) == (120,)
        @test size(C) == (80,)
        @test size(parent(C)) == (100,)
        D = StencilArray(r, Moore{10,1}(); padding=Halo{:in}(), boundary=Remove(0.0));
        D .= 0.0
        @test all(==(0.0), D)
    end
    @testset "2d" begin
        r = rand(100, 100)
        A = StencilArray(copy(r), VonNeumann{10}(), padding=Conditional(), boundary=Remove(0.0));
        B = StencilArray(copy(r), Window{10}(), padding=Halo{:out}(), boundary=Remove(0.0));
        C = StencilArray(copy(r), Moore{10}(), padding=Halo{:in}(), boundary=Remove(0.0));
        @test size(A) == size(parent(A)) == (100, 100)
        @test size(B) == (100, 100)
        @test size(parent(B)) == (120, 120)
        @test axes(parent(B)) == (-9:110, -9:110)
        @test size(C) == (80, 80)
        @test size(parent(C)) == (100, 100)
        @test axes(parent(C)) == (-9:90, -9:90)
        @test typeof(similar(A)) == Matrix{Float64}
        @test typeof(similar(B)) == Matrix{Float64}
        @test typeof(similar(C)) == Matrix{Float64}
        @test size(similar(A)) == size(A)
        @test size(similar(B)) == size(B)
        @test size(similar(C)) == size(C)
        A .= 0
        B .= 0
        C .= 0
        @test all(==(0.0), A)
        @test all(==(0.0), B)
        @test all(==(0.0), C)
    end
    @testset "3d" begin
        r = rand(100, 100, 100)
        A = StencilArray(r, VonNeumann{10,3}(), padding=Conditional(), boundary=Remove(0.0));
        B = StencilArray(r, Window{10,3}(), padding=Halo{:out}(), boundary=Remove(0.0));
        C = StencilArray(r, Moore{10,3}(), padding=Halo{:in}(), boundary=Remove(0.0));
        @test size(A) == size(parent(A)) == (100, 100, 100)
        @test size(B) == (100, 100, 100)
        @test size(parent(B)) == (120, 120, 120)
        @test size(C) == (80, 80, 80)
        @test size(parent(C)) == (100, 100, 100)
        D = StencilArray(r, Moore{10,3}(); padding=Halo{:in}(), boundary=Wrap());
        D .= 0.0
        axes(parent(D))
        @test all(==(0.0), D)
    end
end

@testset "mapstencil" begin
    r = rand(100, 100)
    A = StencilArray(r, Moore{1,2}(); padding=Conditional(), boundary=Remove(zero(eltype(r))));
    B = StencilArray(r, Moore{1,2}(); padding=Halo{:out}(), boundary=Remove(zero(eltype(r))));
    C = StencilArray(r, Moore{1,2}(); padding=Halo{:in}(), boundary=Remove(zero(eltype(r))));
    @time A1 = mapstencil(mean, A)
    @time B1 = mapstencil(mean, B)
    @time C1 = mapstencil(mean, C)
    @test A1[3:end-2, 3:end-2] == B1[3:end-2, 3:end-2] == C1[2:end-1, 2:end-1]
    # Something is wrong with boundary contions
    @test_broken A1[2:end-1, 2:end-1] == B1[2:end-1, 2:end-1] == C1
end

@testset "pad/unpad axes" begin
    A = zeros(6, 7)
    @test Stencils.outer_axes(A, 2) == (-1:8, -1:9)
    @test Stencils.outer_axes(A, Moore(3)) == (-2:9, -2:10)
    @test Stencils.inner_axes(A, 2) == (3:4, 3:5)
    @test Stencils.inner_axes(A, VonNeumann(1)) == (2:5, 2:6)
end

