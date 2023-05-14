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
