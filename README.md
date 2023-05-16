# Stencils

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://rafaqz.github.io/Stencils.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rafaqz.github.io/Stencils.jl/dev)
[![CI](https://github.com/rafaqz/Stencils.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaqz/Stencils.jl/actions/workflows/ci.yml)
[![codecov.io](http://codecov.io/github/rafaqz/Stencils.jl/coverage.svg?branch=master)](http://codecov.io/github/rafaqz/Stencils.jl?branch=master)
[![Aqua.jl Quality Assurance](https://img.shields.io/badge/Aqua.jl-%F0%9F%8C%A2-aqua.svg)](https://github.com/JuliaTesting/Aqua.jl)

Stencils.jl streamlines working with stencils and neighborhoods - 
cellular automata, convolutions and filters, for neighborhoods of any 
(smallish) size and shape.

Stencils.jl defines only direct kernels, no FFTs. But it's fast at 
broadcasting direct kernels. Stencils are StaticArrays.jl vectors 
and are constucted with generated code for performance.

Exeample: mean blur, benchmarked on an 8-core thinkpad:

```julia
using Stencils, Statistics, BenchmarkTools
r = rand(1000, 1000)
A = StencilArray(r, Window(1))
@benchmark broadcast_stencil(mean, A)

BenchmarkTools.Trial: 1058 samples with 1 evaluation.
 Range (min … max):  2.755 ms … 9.693 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.373 ms             ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.718 ms ± 1.326 ms  ┊ GC (mean ± σ):  2.92% ± 5.78%

    ▆▂                                   ▁█▅                 
  ▂▇██▄▁▂▂▁▂▂▄▅▂▂▁▂▁▁▁▂▁▁▁▂▂▁▂▂▂▂▂▂▂▂▂▅▄▂███▆▄▁▁▁▁▂▁▁▂▁▁▂▄▄ ▂
  2.75 ms        Histogram: frequency by time       6.82 ms <

 Memory estimate: 7.64 MiB, allocs estimate: 110.
```

And on the Thinkpads tiny onboard Nvidia GeForce MX330:

```
using CUDA, CUDAKernels
r = CuArray(rand(1000, 1000))
A = StencilArray(r, Window(1))
@benchmark broadcast_stencil(mean, A)

BenchmarkTools.Trial: 3256 samples with 1 evaluation.
 Range (min … max):  916.833 μs …  10.147 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):       1.414 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):     1.521 ms ± 553.722 μs  ┊ GC (mean ± σ):  0.51% ± 2.45%

      ▂█▂                                                        
  ▂▃▄▆██████████▇▆▆▅▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▁▁▂▂▂▂▂ ▃
  917 μs           Histogram: frequency by time         4.11 ms <

 Memory estimate: 4.06 KiB, allocs estimate: 74.
```

Stencils.jl will:

- Run on parallel CPUs and GPUs using KernelAbstractions.jl
- Use any stencil neighborhood shapes and sizes (e.g. Moore and Von Neumann, but also ad-hoc custom offsets).
- Have an easy to use, concise syntax.
- Allow broadcasting stencil operations (not with broadcast syntax tho).
- Allow using stencil in arbitrary loops both to read and write, 
  such as in spatial cost-distance models.
- Provide tools for array switching: where two-layered arrays can be used for
  multiple steps of a simulations, or applying filters repeatedly using the same memory.

It should be usefull for image filtering and convolutions, but has no explicit image or color dependencies.

Expect occasional API breakages, Stencils.jl is being extracted from DynamicGrids.jl, and some coordination
may be required over 2023.
