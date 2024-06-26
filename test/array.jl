using Stencils, Test, LinearAlgebra, StaticArrays, Statistics, DimensionalData

@testset "StencilArray" begin

    @testset "1d" begin
        r = rand(100)
        A = StencilArray(r, VonNeumann{3,1}(); padding=Conditional(), boundary=Remove(0.0));
        B = StencilArray(r, Window{10,1}(); padding=Halo{:out}(), boundary=Remove(0.0));
        C = StencilArray(r, Moore{10,1}(); padding=Halo{:in}(), boundary=Wrap());

        S = SwitchingStencilArray(r, Window{10,1}(); padding=Halo{:out}(), boundary=Wrap());

        @test size(A) == size(parent(A)) == (100,)
        @test size(B) == (100,)
        @test size(parent(B)) == (120,)
        @test size(C) == (80,)
        @test size(parent(C)) == (100,)
        
        @test size(similar(A)) == (100,)
        @test size(similar(B)) == (100,)
        @test size(similar(S)) == (100,)
        @test size(similar(C,Int)) == (80,)

        @test size(similar(B, 40)) == (40,)
        @test size(similar(S, 40)) == (40,)
        @test size(similar(S, Int, 40)) == (40,)
        @test size(parent(similar(B, Int, 20))) == (40,)
        
        @test eltype(similar(S, Bool, 20)) == Bool
        @test eltype(similar(B, Int, 20)) == Int

        D = StencilArray(r, Moore{10,1}(); padding=Halo{:in}(), boundary=Remove(0.0));
        D .= 0.0
        @test all(==(0.0), D)
    end
    @testset "2d" begin
        r = rand(100, 100)
        A = StencilArray(copy(r), VonNeumann{10}(), padding=Conditional(), boundary=Remove(0.0));
        B = StencilArray(copy(r), Window{10}(), padding=Halo{:out}(), boundary=Remove(0.0));
        C = StencilArray(copy(r), Moore{10}(), padding=Halo{:in}(), boundary=Remove(0.0));
        S = SwitchingStencilArray(r, Window{10}(); padding=Halo{:out}(), boundary=Wrap());
        @test size(A) == size(parent(A)) == (100, 100)
        @test size(B) == (100, 100)
        @test size(parent(B)) === (120, 120)
        @test axes(parent(B)) === (Base.OneTo(120), Base.OneTo(1:120))
        @test size(C) === (80, 80)
        @test size(parent(C)) === (100, 100)
        @test axes(parent(C)) === (Base.OneTo(1:100), Base.OneTo(1:100))
        @test typeof(similar(A)) == typeof(A)
        @test typeof(similar(B)) == typeof(B)
        @test typeof(similar(C)) == typeof(C)
        @test size(similar(A)) == size(A)
        @test size(similar(B)) == size(B)
        @test size(similar(C)) == size(C)
        @test size(similar(S)) == size(S)
        @test size(similar(A, 40, 40)) == (40,40)
        @test size(similar(B, 40, 40)) == (40,40)
        @test eltype(similar(B, Int, 20, 20)) == Int
        A .= 0
        B .= 0
        C .= 0
        @test all(==(0.0), A)
        @test all(==(0.0), B)
        @test all(==(0.0), C)
    end

    @testset "1d Window on 2d match 2d line on 2d" begin
        r = rand(100, 100)
        A2d1d = StencilArray(copy(r), Window{1,1}(), padding=Conditional(), boundary=Remove(0.0));
        A2dline = StencilArray(copy(r), Vertical{1,2}(), padding=Conditional(), boundary=Remove(0.0));
        @test mapstencil(sum, A2dline) == mapstencil(sum, A2d1d)
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
        @test size(similar(B)) == size(B)
        @test size(similar(B,(30,30,30))) == ((30,30,30))
        D = StencilArray(r, Moore{10,3}(); padding=Halo{:in}(), boundary=Wrap());
        D .= 0.0
        @test all(==(0.0), D)
    end

    @testset "2d Window on 2d match 2d line on 2d" begin
        r = rand(100, 100, 100)
        window_2d = Window{1,2}()
        pos_3d = Positional((-1, -1, 0), (0, -1, 0), (1, -1, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0))
        @test indices(window_2d, (10, 10, 10)) == indices(pos_3d, (10, 10, 10))
        A3d_window_2d = StencilArray(copy(r), window_2d, padding=Conditional(), boundary=Remove(0.0));
        A3d_pos_3d = StencilArray(copy(r), pos_3d, padding=Conditional(), boundary=Remove(0.0));
        @test neighbors(A3d_window_2d, (10, 10, 10)) == neighbors(A3d_pos_3d, (10, 10, 10))
        @test mapstencil(sum, A3d_window_2d) == mapstencil(sum, A3d_pos_3d)
    end
end

@testset "mapstencil" begin
    # Remove / Use
    r = (1.0:5.0) * (100.0:105.0)'
    A = StencilArray(r, Window{1,2}(); padding=Conditional(), boundary=Remove(zero(eltype(r))));
    B = StencilArray(r, Window{1,2}(); padding=Halo{:out}(), boundary=Remove(zero(eltype(r))));
    C = StencilArray(copy(r), Window{1,2}(); padding=Halo{:in}(), boundary=Stencils.Use())
    @time A1 = mapstencil(mean, A)
    @time B1 = mapstencil(mean, B)
    @time C1 = mapstencil(mean, C)
    @test A1 == B1
    # `mean` just cancels out to give the same answer, inside the padding at least
    @test A1[2:end-1, 2:end-1] == B1[2:end-1, 2:end-1] == C1 == r[2:end-1, 2:end-1]
    @time A1 = mapstencil(sum, A)
    @time B1 = mapstencil(sum, B)
    @time C1 = mapstencil(sum, C)
    @test A1 == B1
    # `sum` gives a different array
    @test A1[2:end-1, 2:end-1] == B1[2:end-1, 2:end-1] == C1 == [
        1818.0  1836.0  1854.0  1872.0
        2727.0  2754.0  2781.0  2808.0
        3636.0  3672.0  3708.0  3744.0
    ]

    # Wrap
    r = (1.0:5.0) * (100.0:105.0)'
    A = StencilArray(r, Moore{1,2}(); padding=Conditional(), boundary=Wrap());
    B = StencilArray(r, Moore{1,2}(); padding=Halo{:out}(), boundary=Wrap());
    C = StencilArray(copy(r), Moore{1,2}(); padding=Halo{:in}(), boundary=Wrap());
    @time A1 = mapstencil(mean, A)
    @time B1 = mapstencil(mean, B)
    @time C1 = mapstencil(mean, C)
    @test A1[3:end-2, 3:end-2] == B1[3:end-2, 3:end-2] == C1[2:end-1, 2:end-1]
    # But C is smaller so the outer wring is different
    # from wrapping in a different place
    @test A1 == B1
    @test A1[2:end-1, 2:end-1] != C1
    parent(A)
    parent(B)
    parent(C)

    # Reflect
    r = (1.0:5.0) * (100.0:105.0)'
    A = StencilArray(r, Window{1,2}(); padding=Conditional(), boundary=Reflect());
    B = StencilArray(r, Window{1,2}(); padding=Halo{:out}(), boundary=Reflect());
    C = StencilArray(copy(r), Window{1,2}(); padding=Halo{:in}(), boundary=Reflect());
    @time A1 = mapstencil(mean, A)
    @time B1 = mapstencil(mean, B)
    @time C1 = mapstencil(mean, C)
    @test A1[3:end-2, 3:end-2] == B1[3:end-2, 3:end-2] == C1[2:end-1, 2:end-1]
    # # Not sure about this one yet
    @test A1[2:end-1, 2:end-1] == B1[2:end-1, 2:end-1]
    @test A1[2:end-1, 2:end-1] != C1
    parent(A)
    parent(B)
    parent(C)


    @testset "Wrapper array types propagate" begin
        A = DimArray(r, (X(10:10:50), Y(1.0:6.0)))
        res_cond = mapstencil(sum, Window(1), A)
        @test res_cond isa DimArray
        @test dims(A) === dims(res_cond)
        res_out = mapstencil(sum, Window(1), A; padding=Halo{:out}())
        @test res_out isa DimArray
        @test dims(A) === dims(res_out)
        res_in = mapstencil(sum, Window(1), A; padding=Halo{:in}())
        @test res_in isa DimArray
        @test map(d -> d[2:end-1], dims(A)) == dims(res_in)
    end
end

@testset "pad/unpad axes" begin
    A = zeros(6, 7)
    @test Stencils.outer_axes(A, 2) == (-1:8, -1:9)
    @test Stencils.outer_axes(A, Moore(3)) == (-2:9, -2:10)
    @test Stencils.inner_axes(A, 2) == (3:4, 3:5)
    @test Stencils.inner_axes(A, VonNeumann(1)) == (2:5, 2:6)
end

@testset "broadcast and copyto!" begin
    A = zeros(6, 7)
    S = StencilArray(ones(6, 7), Moore{1,2}(); padding=Halo{:out}(), boundary=Wrap());
    A .= S
    @test A == S
    S .= zeros(6, 7)
    @test S == zeros(6, 7)
    S2 = StencilArray(ones(6, 7), Moore{1,2}(); padding=Halo{:out}(), boundary=Wrap());
    S .= S2
    @test S == S2
end
##
@testset "copy" begin
    S = StencilArray(ones(6, 7), Moore{1,2}());
    Scopy = copy(S)
    @test S == Scopy

    S = StencilArray(ones(6, 7), Moore{1,2}(); padding=Halo{:out}(), boundary=Wrap());
    Scopy = copy(S)
    @test S == Scopy

    S = SwitchingStencilArray(ones(6, 7), Moore{1,2}(); padding=Halo{:out}(), boundary=Wrap());
    Scopy = copy(S)
    @test S == Scopy
    
end
