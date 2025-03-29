using Stencils, Test, LinearAlgebra, Statistics, DimensionalData


@testset "StencilArray" begin
    @testset "indices" begin
        A = StencilArray(zeros(4, 4), Moore(), boundary=Remove(), padding=Conditional())
        @test indices(A, (1, 1)) == [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
        A = StencilArray(zeros(4, 4), Moore(1), boundary=Wrap(), padding=Conditional())
        @test indices(A, (1, 1)) == [(4, 4), (1, 4), (2, 4), (4, 1), (2, 1), (4, 2), (1, 2), (2, 2)]
        A = StencilArray(zeros(4, 4), Moore(), boundary=Reflect(), padding=Conditional())
        @test indices(A, (1, 1)) == [(2, 2), (1, 2), (2, 2), (2, 1), (2, 1), (2, 2), (1, 2), (2, 2)]
        S = SwitchingStencilArray(zeros(4, 4), Moore(), boundary=Reflect(), padding=Conditional())
        @test indices(S, (1, 1)) == [(2, 2), (1, 2), (2, 2), (2, 1), (2, 1), (2, 2), (1, 2), (2, 2)]
    end

    @testset "1d" begin
        r = rand(100)
        A = StencilArray(r, VonNeumann{3,1}(); padding=Conditional(), boundary=Remove(0.0));
        B = StencilArray(r, Window{10,1}(); padding=Halo{:out}(), boundary=Remove(0.0));
        C = StencilArray(r, Moore{10,1}(); padding=Halo{:in}(), boundary=Remove(0.0));
        sA = SwitchingStencilArray(r, Window{10,1}(); padding=Conditional(), boundary=Remove(0.0));
        sB = SwitchingStencilArray(r, Window{10,1}(); padding=Halo{:out}(), boundary=Remove(0.0));
        sC = SwitchingStencilArray(r, Window{10,1}(); padding=Halo{:in}(), boundary=Remove(0.0));

        @test sA.source !== sA.dest

        @test size(A) == size(parent(A)) == size(sA) == size(parent(sA)) == (100,)
        @test size(B) == size(sB) == (100,)
        @test size(parent(B)) == size(parent(sB)) == (120,)
        @test size(C) == size(sC) == (80,)
        @test size(parent(C)) == size(parent(sC)) == (100,)
        
        @test size(similar(A)) == size(similar(sA)) == (100,)
        @test size(similar(B)) == size(similar(sB)) == (100,)
        @test size(similar(C)) == size(similar(sC)) == (80,)

        @test size(similar(B, 40)) == size(similar(sB, 40)) == (40,)
        @test eltype(similar(B, Int)) == eltype(similar(sB, Int)) == Int

        D = StencilArray(r, Moore{10,1}(); padding=Halo{:in}(), boundary=Remove(0.0));
        D .= 0.0
        @test all(==(0.0), D)
    end

    @testset "2d" begin
        r = rand(100, 100)
        A = StencilArray(copy(r), VonNeumann{10}(), padding=Conditional(), boundary=Remove(0.0));
        B = StencilArray(copy(r), Window{10}(), padding=Halo{:out}(), boundary=Remove(0.0));
        C = StencilArray(copy(r), Moore{10}(), padding=Halo{:in}(), boundary=Remove(0.0));
        SA = SwitchingStencilArray(copy(r), VonNeumann{10}(), padding=Conditional(), boundary=Remove(0.0));
        SB = SwitchingStencilArray(copy(r), Window{10}(), padding=Halo{:out}(), boundary=Remove(0.0));
        SC = SwitchingStencilArray(copy(r), Moore{10}(), padding=Halo{:in}(), boundary=Remove(0.0));

        @test size(A) == size(SA) == size(parent(A)) == (100, 100)
        @test size(B) == size(SB) == (100, 100)
        @test size(parent(B)) === size(parent(SB)) === (120, 120)
        @test axes(parent(B)) === axes(parent(SB)) === (Base.OneTo(120), Base.OneTo(1:120))
        @test size(C) === size(SC) === (80, 80)
        @test size(parent(C)) === size(parent(SC)) === (100, 100)
        @test axes(parent(C)) === axes(parent(SC)) === (Base.OneTo(1:100), Base.OneTo(1:100))
        @test typeof(similar(A)) == typeof(A)
        @test typeof(similar(B)) == typeof(B)
        @test typeof(similar(C)) == typeof(C)
        @test typeof(similar(SA)) == typeof(SA)
        @test typeof(similar(SB)) == typeof(SB)
        @test typeof(similar(SC)) == typeof(SC)
        @test size(similar(A)) == size(similar(SA)) == size(A)
        @test size(similar(B)) == size(similar(SB)) == size(B)
        @test size(similar(C)) == size(similar(SC)) == size(C)
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
        SA = SwitchingStencilArray(r, VonNeumann{10,3}(), padding=Conditional(), boundary=Remove(0.0));
        SB = SwitchingStencilArray(r, Window{10,3}(), padding=Halo{:out}(), boundary=Remove(0.0));
        SC = SwitchingStencilArray(r, Moore{10,3}(), padding=Halo{:in}(), boundary=Remove(0.0));

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

    @testset "2d Window matches 3d line on 3d" begin
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
    @testset "Remove / Use" begin
        r = (1.0:5.0) * (100.0:105.0)'
        A = StencilArray(r, Window{1,2}(); padding=Conditional(), boundary=Remove(zero(eltype(r))));
        B = StencilArray(r, Window{1,2}(); padding=Halo{:out}(), boundary=Remove(zero(eltype(r))));
        C = StencilArray(copy(r), Window{1,2}(); padding=Halo{:in}(), boundary=Stencils.Use())
        SA = SwitchingStencilArray(copy(r), Window{1,2}(); padding=Conditional(), boundary=Remove(zero(eltype(r))));
        SB = SwitchingStencilArray(copy(r), Window{1,2}(); padding=Halo{:out}(), boundary=Remove(zero(eltype(r))));
        SC = SwitchingStencilArray(copy(r), Window{1,2}(); padding=Halo{:in}(), boundary=Stencils.Use());

        # `mean` just cancels out to give the same answer, inside the padding at least
        @time A1 = mapstencil(mean, A)
        @time B1 = mapstencil(mean, B)
        @time C1 = mapstencil(mean, C)
        @time SA1 = mapstencil!(mean, SA)
        @time SB1 = mapstencil!(mean, SB)
        @time SC1 = mapstencil!(mean, SC)
        @test A1 == B1 == SA1 == SB1 ≈ [
             67.0 101.0 102.0 103.0 104.0  69.66666666667
            134.0 202.0 204.0 206.0 208.0 139.33333333333
            201.0 303.0 306.0 309.0 312.0 209.0
            268.0 404.0 408.0 412.0 416.0 278.66666666667
            201.0 303.0 306.0 309.0 312.0 209.0
        ]
        @test C1 == SC1 
        @test A1[2:end-1, 2:end-1] == C1 == SC1 == r[2:end-1, 2:end-1]

        # `sum` gives a different array
        @time A1 = mapstencil(sum, A)
        @time B1 = mapstencil(sum, B)
        @time C1 = mapstencil(sum, C)
        @time SA1 = mapstencil!(sum, SA)
        @time SB1 = mapstencil!(sum, SB)
        @time SC1 = mapstencil!(sum, SC)
        @test A1 == B1 == SA1 == SB1
        @test A1[2:end-1, 2:end-1] == C1 == SC1 == [
            1818.0  1836.0  1854.0  1872.0
            2727.0  2754.0  2781.0  2808.0
            3636.0  3672.0  3708.0  3744.0
        ]
    end

    @testset "Wrap" begin
        @testset "1d" begin
            s = Window{1,1}()
            x = collect(1.0:5.0)
            A = StencilArray(x, s; padding=Conditional(), boundary=Wrap());
            B = StencilArray(x, s; padding=Halo{:out}(), boundary=Wrap());
            C = StencilArray(copy(x), s; padding=Halo{:in}(), boundary=Wrap());
            SA = SwitchingStencilArray(copy(x), s; padding=Conditional(), boundary=Wrap());
            SB = SwitchingStencilArray(copy(x), s; padding=Halo{:out}(), boundary=Wrap());
            SC = SwitchingStencilArray(copy(x), s; padding=Halo{:in}(), boundary=Wrap());

            @time A1 = mapstencil(mean, A)
            @time B1 = mapstencil(mean, B)
            @time C1 = mapstencil(mean, C)
            @time SA1 = mapstencil!(mean, SA)
            @time SB1 = mapstencil!(mean, SB);
            @time SC1 = mapstencil!(mean, SC)
            @test A1 == B1 == SA1 == [2.6666666666666665, 2.0, 3.0, 4.0, 3.3333333333333335]
            @test C1 == SC1 == [3.0, 3.0, 3.0]
        end
        @testset "2d" begin
            s = Window{1,2}()
            r = (1.0:5.0) * (100.0:105.0)'
            A = StencilArray(r, s; padding=Conditional(), boundary=Wrap());
            B = StencilArray(r, s; padding=Halo{:out}(), boundary=Wrap());
            C = StencilArray(copy(r), s; padding=Halo{:in}(), boundary=Wrap());
            SA = SwitchingStencilArray(copy(r), s; padding=Conditional(), boundary=Wrap());
            SB = SwitchingStencilArray(copy(r), s; padding=Halo{:out}(), boundary=Wrap());
            SC = SwitchingStencilArray(copy(r), s; padding=Halo{:in}(), boundary=Wrap());

            @time A1 = mapstencil(mean, A)
            @time B1 = mapstencil(mean, B)
            @time C1 = mapstencil(mean, C)
            @time SA1 = mapstencil!(mean, SA)
            @time SB1 = mapstencil!(mean, SB)
            @time SC1 = mapstencil!(mean, SC)
            @test A1 == B1 == SA1 == SB1 ≈ [
                272.0 269.33333333 272.0  274.66666666 277.33333333 274.66666666
                204.0 202.0        204.0  206.0        208.0        206.0
                306.0 303.0        306.0  309.0        312.0        309.0
                408.0 404.0        408.0  412.0        416.0        412.0
                340.0 336.66666666 340.0  343.33333333 346.66666666 343.33333333
            ]
            @test C1 == SC1 
        end
    end

    @testset "Reflect" begin
        @testset "1d" begin
            r = collect(1.0:5.0)
            s = Window{1,1}()
            A = StencilArray(r, s; padding=Conditional(), boundary=Reflect());
            B = StencilArray(r, s; padding=Halo{:out}(), boundary=Reflect());
            C = StencilArray(copy(r), s; padding=Halo{:in}(), boundary=Reflect());
            SA = SwitchingStencilArray(copy(r), s; padding=Conditional(), boundary=Reflect());
            SB = SwitchingStencilArray(copy(r), s; padding=Halo{:out}(), boundary=Reflect());
            SC = SwitchingStencilArray(copy(r), s; padding=Halo{:in}(), boundary=Reflect());

            @time A1 = mapstencil(mean, A)
            @time B1 = mapstencil(mean, B) 
            @time C1 = mapstencil(mean, C)
            @time SA1 = mapstencil!(mean, SA)
            @time SB1 = mapstencil!(mean, SB)
            @time SC1 = mapstencil!(mean, SC)
            @test A1 == B1 == SA1 == SB1 ≈ [1.66666666, 2.0, 3.0, 4.0, 4.33333333]
            @test C1 == SC1 ≈ [2.66666666, 3.0, 3.33333333]
        end
        @testset "2d" begin
            r = (1.0:5.0) * (100.0:105.0)'
            s = Window{1,2}()
            A = StencilArray(r, s; padding=Conditional(), boundary=Reflect());
            B = StencilArray(r, s; padding=Halo{:out}(), boundary=Reflect());
            C = StencilArray(copy(r), s; padding=Halo{:in}(), boundary=Reflect());
            SA = SwitchingStencilArray(copy(r), s; padding=Conditional(), boundary=Reflect());
            SB = SwitchingStencilArray(copy(r), s; padding=Halo{:out}(), boundary=Reflect());
            SC = SwitchingStencilArray(copy(r), s; padding=Halo{:in}(), boundary=Reflect());

            @time A1 = mapstencil(mean, A)
            @time B1 = mapstencil(mean, B) 
            @time C1 = mapstencil(mean, C)
            @time SA1 = mapstencil!(mean, SA)
            @time SB1 = mapstencil!(mean, SB)
            @time SC1 = mapstencil!(mean, SC)
            @test A1 == B1 == SA1 == SB1 ≈ [
                167.77777777 168.33333333 170.0 171.66666666 173.33333333 173.888888888
                201.33333333 202.0        204.0 206.0        208.0        208.666666666
                302.0        303.0        306.0 309.0        312.0        313.0
                402.66666666 404.0        408.0 412.0        416.0        417.333333333
                436.22222222 437.66666666 442.0 446.33333333 450.66666666 452.111111111
            ]
            @test C1 == SC1 
            @test A1[3:end-2, 3:end-2] == C1[2:end-1, 2:end-1] == SC1[2:end-1, 2:end-1]

            @test A1[2:end-1, 2:end-1] == B1[2:end-1, 2:end-1] == SA1[2:end-1, 2:end-1] == SB1[2:end-1, 2:end-1] == r[2:end-1, 2:end-1]
            @test A1[2:end-1, 2:end-1] != C1
        end
    end

    @testset "Wrapper array types propagate" begin
        r = (1.0:5.0) * (100.0:105.0)'
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
        @test mapstencil(sum, Stencils.Circle(2), A) == mapstencil(sum, StencilArray(A, Stencils.Circle(2)))
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
