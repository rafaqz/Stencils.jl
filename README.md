# Stencils

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://rafaqz.github.io/Stencils.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rafaqz.github.io/Stencils.jl/dev)
[![CI](https://github.com/rafaqz/Stencils.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaqz/Stencils.jl/actions/workflows/ci.yml)
[![codecov.io](http://codecov.io/github/rafaqz/Stencils.jl/coverage.svg?branch=master)](http://codecov.io/github/rafaqz/Stencils.jl?branch=master)
[![Aqua.jl Quality Assurance](https://img.shields.io/badge/Aqua.jl-%F0%9F%8C%A2-aqua.svg)](https://github.com/JuliaTesting/Aqua.jl)

Stencils.jl streamlines working with stencils and neighborhoods - 
cellular automata, convolutions and filters, for neighborhoods of any 
(smallish) size and shape. 

Stencils.jl builds on StaticArrays.jl and KernelAbstractions.jl to provide high 
performance tools on CPUs and most GPUs, but keep a simple code base.


## What is a `Stencil` in Stencils.jl?

A `Stencil` is a StaticArrays.jl `StaticVector` of values cut from an array in the
specified shape, initially it is filled with `nothing`s. Stencils.jl provides methods to 
update stenctil values (by rebuilding) for any center index in an array, handling boundary conditions
by either padding or checking bounds.

Stencils.jl also provides functions to retreive the `offsets`, `indices`, `distances`
from the center pixel, and other information about the stencil. These are all compile time
operations, usable in fast inner loops or in GPU kernels. `@generated`
functions are used in most cases to guarantee compiling performant, type-stable
code for all arbitrary stencil shapes and sizes.

Stencils are defined with radius and number of dimensions: 
```julia
using Stencils
radius = 1
ndims = 2

julia> Moore(radius, ndims)
Moore{1, 2, 8, Nothing}
█▀█
▀▀▀
```

The third number - here `8` - is the calculated length of the Stencil. Note
that no matter what you use for `ndims`, the stencil is still `<: StaticVector`,
because the potential for missing positions mean we need to collapse the 
dimensions into one to keep things generic.

You can also define stencils using the type parameters directly:

```julia
julia> Circle{3,2}()
Circle{3, 2, 37, Nothing}
 ▄███▄ 
███████
▀█████▀
  ▀▀▀  
```

There are a lot of default stencils built-in, with customisable size and
dimensionality, for example:

```julia
# Most shapes look pretty similar in 1d (and they all look better in a terminal!)
julia> Moore(2, 1)
Moore{2, 1, 4, Nothing}
█
▄
▀

# 2D is the default
julia> Moore(1)
Moore{1, 2, 8, Nothing}
█▀█
▀▀▀

# Some shapes include the center
julia> Window(3)
Window{3, 2, 49, Nothing}
███████
███████
███████
▀▀▀▀▀▀▀

# This shape is 3d, we just cant easily show that...
julia> VonNeumann(2, 3)
VonNeumann{2, 3, 24, Nothing}
 ▄█▄ 
▀█▄█▀
  ▀  

# This is 4d
julia> Cross(3, 4)
Cross{3, 4, 25, Nothing}
   █   
▄▄▄█▄▄▄
   █   
   ▀   
```

But you can also make up arbitrary shapes:

```julia
julia> Positional((-1, 1), (-2, -1), (1, 0), (-2, 2))
Positional{((-1, 1), (-2, -1), (1, 0), (-2, 2)), 2, 2, 4, Nothing}
 ▀ ▄▀
  ▄  
```

You can even give each positions a name,
so you can use the stencil like a `NamedTuple`:

```julia
julia> ns = NamedStencil(n=(-1, 0), w=(0, -1), s=(1, 0), e=(0, 1))
NamedStencil{(:n, :w, :s, :e), ((-1, 0), (0, -1), (1, 0), (0, 1)), 1, 2, 4, Nothing}
▄▀▄
 ▀ 

julia> A = StencilArray(
           [0 2 0 0 4
            0 0 6 0 0
            0 0 0 8 7
            0 0 3 0 1
            5 0 1 0 9],
           ns
       );

julia> stencil(A, (4, 4)).n
8

julia> stencil(A, (4, 4)).w
3
```


## How can we use stencils?

Stencils.jl defines only direct methods, no FFTs. But it's very fast at 
mapping direct kernels over arrays. The `StencilArray` provides a wrapper
for these operations that will properly handle boundary conditions.

### Example: mean blur

_benchmarked on an 8-core thinkpad T14:_

```julia
using Stencils, Statistics, BenchmarkTools

# Define a random array
r = rand(1000, 1000)

# Use a square 3x3/radius 1 stencil
stencil = Window(1)

# Wrap them both as a StencilArray
A = StencilArray(r, stencil)

# Map `mean` over all stencils in the array. You can use any function here - 
# `identity` would return an array of `Window` stencils.
@benchmark mapstencil(mean, A)

BenchmarkTools.Trial: 1058 samples with 1 evaluation.
 Range (min … max):  2.755 ms … 9.693 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.373 ms             ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.718 ms ± 1.326 ms  ┊ GC (mean ± σ):  2.92% ± 5.78%

    ▆▂                                   ▁█▅                 
  ▂▇██▄▁▂▂▁▂▂▄▅▂▂▁▂▁▁▁▂▁▁▁▂▂▁▂▂▂▂▂▂▂▂▂▅▄▂███▆▄▁▁▁▁▂▁▁▂▁▁▂▄▄ ▂
  2.75 ms        Histogram: frequency by time       6.82 ms <

 Memory estimate: 7.64 MiB, allocs estimate: 110.
```

_and on the Thinkpads tiny onboard Nvidia GeForce MX330:_

```julia
using CUDA
r = CuArray(rand(1000, 1000))
A = StencilArray(r, Window(1))
@benchmark mapstencil(mean, A)

BenchmarkTools.Trial: 3256 samples with 1 evaluation.
 Range (min … max):  916.833 μs …  10.147 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):       1.414 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):     1.521 ms ± 553.722 μs  ┊ GC (mean ± σ):  0.51% ± 2.45%

      ▂█▂                                                        
  ▂▃▄▆██████████▇▆▆▅▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▁▁▂▂▂▂▂ ▃
  917 μs           Histogram: frequency by time         4.11 ms <

 Memory estimate: 4.06 KiB, allocs estimate: 74.
```

(That is less than a nanosecond per operation - reading each 3x3 stencil, calling `mean` on it, and writing the result)

And on a GTX 1080:

```julia
using CUDA
r = CuArray(rand(1000, 1000))
A = SwitchingStencilArray(r, Window(1))
CUDA.@time mapstencil(mean, A);
  0.000216 seconds (64 CPU allocations: 3.875 KiB) (1 GPU allocation: 7.629 MiB, 5.80% memmgmt time)
# Or 216.0 μs
```

### Example: convolution
Stencels can also be used to convolve a Kernel with each stencil of a StencilArrray using `mapstencil` and `kernelproduct`:

```julia
using Stencils

# Define a random array that the kernel will be convolved with
r = rand(1000, 1000)

# Define kernel array
sharpen = [0 -1 0;
           -1 5 -1;
           0 -1 0]

# Define a stencil that is the same size as the kernel array
stencil = Window(1)

# Create a stencil Kernel from the stencil and the kernel array
k = Kernel(stencil, sharpen)

# Wrap the random array and the Kernel in a StencilArray
A = StencilArray(r, k)

# use `mapstencil` with the `kernelproduct` function to convolve the Kernel with array. 
# Note: `kernelproduce is similar to `Base.dot` but `kernelproduct` 
# lets you use an array of StaticArray and it will still work (dot is recursive).
mapstencil(kernelproduct, A) 
```

Or 

Stencils can be used standalone, outside of `mapstencil`. For example
in iterative cost-distance models. Stencils provides a `stencil` method
to fill a stencil for a particular location `stencil(A, I)`. It also has an 
`indices(A, I)` method to retreive array indices for the stencil, so you can use them to 
write values into an array for the stencil shape around your specified center index.

### SwitchingsStencilArray Example: rock-paper-scissors on a grid

Stencils.jl also provides a `SwitchingStencilArray` for use in contexts where multiple passes are 
required. This may be a simple iterative filter, or a simplation like the below.

<!-- This example adapted from https://discourse.julialang.org/t/seven-lines-of-julia-examples-sought/50416/166 -->

```julia
using Stencils, CairoMakie
function nextvalue_rps(neighborhood)
    dominant = mod1(neighborhood.center + 1, 3) # 3 beats 2, 2 beats 1, 1 beats 3
    return count(==(dominant), neighborhood.neighbors) >= 3 ? dominant : neighborhood.center
end

ssa = SwitchingStencilArray(rand(1:3, 100, 100), Window(1); boundary = Wrap())

f, a, p = heatmap(ssa; colormap = Makie.wong_colors()[1:3], axis = (; aspect = true))

record(f, "animation.gif", 1:100) do i
  global ssa
  # Switch the stencil array and run the mapping function
  ssa = mapstencil!(nextvalue_rps, ssa)
  # Update the plot, since the SwitchingStencilArray _is_ an array
  p.args[1][] = ssa
end
```
![animation](https://github.com/user-attachments/assets/d45e334e-9d4c-415d-89a3-b4c144b7a505)
