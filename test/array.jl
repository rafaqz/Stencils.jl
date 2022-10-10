using Neighborhoods, Test, LinearAlgebra, StaticArrays, OffsetArrays

@testset "NeighborhoodArray" begin

    @testset "1d" begin
        r = rand(100)
        A = NeighborhoodArray(r; neighborhood=VonNeumann{3,1}(), padding=Conditional(), boundary_condition=Remove(0.0));
        B = NeighborhoodArray(r; neighborhood=Window{10,1}(), padding=Halo{:out}(), boundary_condition=Remove(0.0));
        C = NeighborhoodArray(r; neighborhood=Moore{10,1}(), padding=Halo{:in}(), boundary_condition=Wrap());
        @test size(A) == size(parent(A)) == (100,)
        @test size(B) == (100,)
        @test size(parent(B)) == (120,)
        @test size(C) == (80,)
        @test size(parent(C)) == (100,)
        D = NeighborhoodArray(r; neighborhood=Moore{10,1}(), padding=Halo{:in}(), boundary_condition=Remove(0.0));
        D .= 0.0
        @test all(==(0.0), D)
    end
    @testset "2d" begin
        r = rand(500, 500)
        A = NeighborhoodArray(r; neighborhood=VonNeumann{3}(), padding=Conditional(), boundary_condition=Remove(0.0));
        B = NeighborhoodArray(r; neighborhood=Window{10}(), padding=Halo{:out}(), boundary_condition=Remove(0.0));
        C = NeighborhoodArray(r; neighborhood=Moore{10}(), padding=Halo{:in}(), boundary_condition=Wrap());
        @test size(A) == size(parent(A)) == (100, 100)
        @test size(B) == (100, 100)
        @test size(parent(B)) == (120, 120)
        @test axes(parent(B)) == (-9:110, -9:110)
        @test size(C) == (80, 80)
        @test size(parent(C)) == (100, 100)
        @test axes(parent(C)) == (-9:90, -9:90)
        @test typeof(similar(A)) == typeof(A)
        @test typeof(similar(B)) == typeof(B)
        @test typeof(similar(C)) == typeof(C)
        @test size(similar(A)) == size(A)
        @test size(similar(B)) == size(B)
        @test size(similar(C)) == size(C)
        D = NeighborhoodArray(r; neighborhood=Moore{10,2}(), padding=Halo{:in}(), boundary_condition=Remove(0.0));
        D .= 0.0
        @test 
        using BenchmarkTools, ProfileView
        f(A) = map(==(0.0), A)
        @benchmark f($A)
        @benchmark f(parent(parent($A)))

        @profview for i in 1:100 f(A) end
        @profview for i in 1:100 f(parent(A)) end
        @test all(==(0.0), parent(D))
    end
    @testset "3d" begin
        r = rand(100, 100, 100)
        A = NeighborhoodArray(r; neighborhood=VonNeumann{3,3}(), padding=Conditional(), boundary_condition=Remove(0.0));
        B = NeighborhoodArray(r; neighborhood=Window{10,3}(), padding=Halo{:out}(), boundary_condition=Remove(0.0));
        C = NeighborhoodArray(r; neighborhood=Moore{10,3}(), padding=Halo{:in}(), boundary_condition=Wrap());
        @test size(A) == size(parent(A)) == (100, 100, 100)
        @test size(B) == (100, 100, 100)a

        @test size(parent(B)) == (120, 120, 120)
        @test size(C) == (80, 80, 80)
        @test size(parent(C)) == (100, 100, 100)
        D = NeighborhoodArray(r; neighborhood=Moore{10,3}(), padding=Halo{:in}(), boundary_condition=Remove(0.0));
        D .= 0.0
    end

    using ProfileView
    using CUDA, KernelAbstractions, CUDAKernels
    using BenchmarkTools
    function f(hood)
        sum = zero(first(hood))
        for x in hood
            sum += 3x
        end
        return sum
    end
    function f2(hood, x)
        sum = zero(first(hood))
        for x in hood
            sum += 3x
        end
        return sum
    end
    # r = CuArray(r)
    @benchmark
    @time broadcast_neighborhood(f, A)
    @time broadcast_neighborhood(f2, A, A)
    @benchmark
    @time broadcast_neighborhood(f, B)
    C = NeighborhoodArray(r; neighborhood=Moore{10}(), padding=Unpadded(), boundary_condition=Wrap());
    @benchmark
    @profview
    @time broadcast_neighborhood(f, C)
    @benchmark
    @time broadcast_neighborhood(f, D)
    E = NeighborhoodArray(r; neighborhood=VonNeumann{50}(), padding=Padded{:out}(), boundary_condition=Remove(0.0));
    @time broadcast_neighborhood(f, E)
    @profview broadcast_neighborhood(f, E)
end

@testset "pad/unpad axes" begin
    A = zeros(6, 7)
    @test Neighborhoods.outer_axes(A, 2) == (-1:8, -1:9)
    @test Neighborhoods.outer_axes(A, Moore(3)) == (-2:9, -2:10)
    @test Neighborhoods.inner_axes(A, 2) == (3:4, 3:5)
    @test Neighborhoods.inner_axes(A, VonNeumann(1)) == (2:5, 2:6)
end

