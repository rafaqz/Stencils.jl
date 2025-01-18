var documenterSearchIndex = {"docs":
[{"location":"#Stencils","page":"Stencils","title":"Stencils","text":"","category":"section"},{"location":"","page":"Stencils","title":"Stencils","text":"Main functions","category":"page"},{"location":"","page":"Stencils","title":"Stencils","text":"mapstencil\nmapstencil!","category":"page"},{"location":"#Stencils.mapstencil","page":"Stencils","title":"Stencils.mapstencil","text":"mapstencil(f, A::StencilArray, args::AbstractArray...)\nmapstencil(f, stencil::Stencil, A::AbstractArray, args::AbstractArray...; kw...)\n\nStencil mapping where f is passed a Stencil centered at each index in A, followed by the values from args at each stencil center.\n\nKeywords\n\nboundary: a BoundaryCondition like Wrap.\npadding: Padding like Conditional or Halo{:in}.\n\nThe result is returned as a new array.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.mapstencil!","page":"Stencils","title":"Stencils.mapstencil!","text":"mapstencil!(f, dest::AbstractArray, source::StencilArray, args::AbstractArray...)\nmapstencil!(f, A::SwitchingStencilArray, args::AbstractArray...)\n\nStencil mapping where f is passed a stencil centered at each index in src, followed by the values from args at each stencil center. The result of f is written to dest.\n\nFor SwitchingStencilArray the internal source and dest arrays are used, returning a switched version of the array.\n\ndest must either be smaller than src by the stencil radius on all sides, or be the same size, in which case it is assumed to also be padded.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils-2","page":"Stencils","title":"Stencils","text":"","category":"section"},{"location":"","page":"Stencils","title":"Stencils","text":"Stencil\nNamedStencil\nAngledCross\nBackSlash\nCircle\nCross\nDiamond\nForwardSlash\nKernel\nHorizontal\nLayered\nMoore\nPositional\nRectangle\nVertical\nVonNeumann\nWindow\nAnnulus\nCardinal\nOrdinal","category":"page"},{"location":"#Stencils.Stencil","page":"Stencils","title":"Stencils.Stencil","text":"Stencil <: StaticVector\n\nStencils define a pattern of neighboring cells around the current cell. They reduce the structure and dimensions of the neighborhood into a StaticVector of values.\n\nStencil objects are updated to contain the neighbors for an array index.\n\nThis design is so that user functions can be passed a single object from which they can retrieve center, neighbors, offsets, distances to neighbors and other information.\n\nStencils also provide a range of compile-time utility funcitons like distances and offsets.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.NamedStencil","page":"Stencils","title":"Stencils.NamedStencil","text":"NamedStencil <: AbstractStencil\n\nNamedStencil(; kw...)\nNamedStencil(values::NamedTuple)\nNamedStencil{Keys}(values)\n\nA named stencil that can take arbitrary shapes where each offset position is named. This can make stencil code easier to read by  removing magic numbers.\n\nExample\n\njulia> using Stencils\n\njulia> ns = NamedStencil(; west=(0, -1), north=(1, 0), south=(-1, 0), east=(0, 1)) \nNamedStencil{(:west, :north, :south, :east), ((0, -1), (1, 0), (-1, 0), (0, 1)), 1, 2, 4, Nothing}\n▄▀▄\n ▀ \n\njulia> A = StencilArray((1:10) * (1:10)', ns);\n\njulia> stencil(A, (5, 5)).east # we can access values by name\n30\n\njulia> mapstencil(s -> s.east + s.west, A); # and use them in `mapstencil` functions\n\nWe can also take some shortcuts, and just name an existing stencil:\n\njulia> ns = NamedStencil{(:w,:n,:s,:e)}(VonNeumann(1)) \n\nThe stencil radius is calculated from the most distant coordinate, and the dimensionality N of the stencil is taken from the length of the first coordinate, e.g. 1, 2 or 3.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.AngledCross","page":"Stencils","title":"Stencils.AngledCross","text":"AngledCross <: Stencil\n\nAngledCross(; radius=1, ndims=2)\nAngledCross(radius, ndims)\nAngledCross{R,N}()\n\nA neighboorhood where all diagonals are included, for 2:N dimensions\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.BackSlash","page":"Stencils","title":"Stencils.BackSlash","text":"BackSlash <: Stencil\n\nBackSlash(; radius=1, ndims=2)\nBackSlash(radius, ndims)\nBackSlash{R,N}()\n\nA neighboorhood along the 'backwards' diagonal. Contains 2R+1 neighbors for for 2:N dimensions\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Circle","page":"Stencils","title":"Stencils.Circle","text":"Circle <: Stencil\n\nCircle(; radius=1, ndims=2)\nCircle(radius, ndims)\nCircle{R,N}()\n\nA circular or spherical stencil\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Cross","page":"Stencils","title":"Stencils.Cross","text":"Cross <: Stencil\n\nCross(; radius=1, ndims=2)\nCross(radius, ndims)\nCross{R,N}()\n\nA cross-shaped neighboorhood where positions with offsets of 0 on least N-1 axes are included\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Diamond","page":"Stencils","title":"Stencils.Diamond","text":"Diamond <: Stencil\n\nDiamond(; radius=1, ndims=2)\nDiamond(radius, ndims)\nDiamond{R,N}()\n\nA diamond or regular octahedron\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.ForwardSlash","page":"Stencils","title":"Stencils.ForwardSlash","text":"ForwardSlash <: Stencil\n\nForwardSlash(; radius=1, ndims=2)\nForwardSlash(radius, ndims)\nForwardSlash{R,N}()\n\nA neighboorhood where only 'forward' diagonals are included. Contains 2R+1 neighbors for 2:N dimensions\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Kernel","page":"Stencils","title":"Stencils.Kernel","text":"Kernel <: AbstractKernelStencil\n\nKernel(stencil::Stencil, kernel::AbstractArray)\nKernel(f::Function, stencil::Stencil)\n\nWrap any other stencil object, and includes a kernel array of the same length and positions as the stencil. A function of the stencil and kernel, like kernelproduct can be used in  mapstencil.\n\nA function f may be passed as the first argument, and a kernel array will be calculated with map(f, distances(stencil)).\n\nAs an example, Kernels can be convolved with an Array\n\nusing Stencils\n\n# Define a random array that the kernel will be convolved with\nr = rand(1000, 1000)\n\n# Define kernel array\nsharpen = [0 -1 0;\n           -1 5 -1;\n           0 -1 0]\n\n# Define a stencil that is the same size as the kernel array\nstencil = Window(1)\n\n# Create a stencil Kernel from the stencil and the kernel array\nk = Kernel(stencil, sharpen)\n\n# Wrap the random array and the Kernel in a StencilArray\nA = StencilArray(r, k)\n\n# use `mapstencil` with the `kernelproduct` function to convolve the Kernel with array. \n# Note: `kernelproduce is similar to `Base.dot` but `kernelproduct` \n# lets you use an array of StaticArray and it will still work (dot is recursive).\nmapstencil(kernelproduct, A) \n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Horizontal","page":"Stencils","title":"Stencils.Horizontal","text":"Horizontal <: Stencil\n\nHorizontal(; radius=1, ndims=2)\nHorizontal(radius, ndims)\nHorizontal{R,N}()\n\nA horizontal bar or plane\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Layered","page":"Stencils","title":"Stencils.Layered","text":"Layered <: Abstract\n\nLayered(layers::Union{Stencil,Layered}...)\nLayered(; layer_keywords...)\nLayered(layers::Union{Tuple,NamedTuple})\n\nTuple or NamedTuple of stencils that can be used together.\n\nneighbors for Layered returns a tuple of iterators for each stencil layer.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Moore","page":"Stencils","title":"Stencils.Moore","text":"Moore <: Stencil\n\nMoore(; radius=1, ndims=2)\nMoore(radius, ndims)\nMoore{R,N}()\n\nMoore stencils define the stencil as all cells within a horizontal or vertical distance of the central cell. The central cell is omitted.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Positional","page":"Stencils","title":"Stencils.Positional","text":"Positional <: AbstractPositionalStencil\n\nPositional(offsets::Tuple{Vararg{Int}}...)\nPositional(offsets::Tuple{Tuple{Vararg{Int}}})\nPositional{O}()\n\nStencils that can take arbitrary shapes by specifying each coordinate, as Tuple{Int,Int} of the row/column distance (positive and negative) from the central point.\n\nThe stencil radius is calculated from the most distant coordinate, and the dimensionality N of the stencil is taken from the length of the first coordinate, e.g. 1, 2 or 3.\n\nSee NamedStencil for a similar stencil with named offsets.\n\nExample\n\njulia> p = Positional((0, -1), (2, 1), (-1, 1), (0, 1)) \nPositional{((0, -1), (2, 1), (-1, 1), (0, 1)), 2, 2, 4, Nothing}\n   ▄ \n ▀ ▀ \n   ▀ \n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Rectangle","page":"Stencils","title":"Stencils.Rectangle","text":"Rectangle <: Stencil\n\nRectangle(offsets::Tuple{Tuple}...)\nRectangle{O}()\n\nRectanglar stencils of arbitrary shapes. These are  specified with pulles of offsets around the center point,  one for each dimension.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Vertical","page":"Stencils","title":"Stencils.Vertical","text":"Vertical <: Stencil\n\nVertical(; radius=1, ndims=2)\nVertical(radius, ndims)\nVertical{R,N}()\n\nA vertical bar or plane\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.VonNeumann","page":"Stencils","title":"Stencils.VonNeumann","text":"VonNeumann <: Stencil\n\nVonNeumann(; radius=1, ndims=2)\nVonNeumann(radius, ndims)\nVonNeumann{R,N}()\n\nDiamond-shaped neighborhood (in 2 dimensions), without the central cell In 1 dimension it is identical to Moore.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Window","page":"Stencils","title":"Stencils.Window","text":"Window <: Stencil\n\nWindow(; radius=1, ndims=2)\nWindow(radius, ndims)\nWindow{R,N}()\n\nA neighboorhood of radius R that includes the central cell.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Annulus","page":"Stencils","title":"Stencils.Annulus","text":"Annulus <: Stencil\n\nAnnulus(; outerradius=2, innerradius=1, ndims=2) Annulus(outerradius, innerradius, ndims) Annulus{RO,RI,N}()\n\nA donut or hollowed spherical stencil\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Cardinal","page":"Stencils","title":"Stencils.Cardinal","text":"Cardinal <: Stencil\n\nCardinal(; radius=1, ndims=2)\nCardinal(radius, ndims)\nCardinal{R,N}()\n\nCardinal (as in N,S,W,E compass directions) stencil\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Ordinal","page":"Stencils","title":"Stencils.Ordinal","text":"Ordinal <: Stencil\n\nOrdinal(; radius=1, ndims=2)\nOrdinal(radius, ndims)\nOrdinal{R,N}()\n\nOrdinal (as in NE,SE,SW,NW wind directions) stencil\n\n\n\n\n\n","category":"type"},{"location":"#Stencil-functions","page":"Stencils","title":"Stencil functions","text":"","category":"section"},{"location":"","page":"Stencils","title":"Stencils","text":"These can be called on any <: Stencil object, and Layered.","category":"page"},{"location":"","page":"Stencils","title":"Stencils","text":"stencil\nneighbors\ncenter\noffsets\nindices\ndistances\nradius\ndiameter\nStencils.distance_zones\nStencils.kernel\nStencils.kernelproduct\nStencils.unsafe_neighbors\nStencils.unsafe_stencil\nStencils.rebuild\nStencils.getneighbor","category":"page"},{"location":"#Stencils.stencil","page":"Stencils","title":"Stencils.stencil","text":"stencil(x) -> Stencil\n\nReturns a stencil object.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.neighbors","page":"Stencils","title":"Stencils.neighbors","text":"neighbors(x::Stencil) -> iterable\n\nReturns a basic SVector of all cells in the stencil.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.center","page":"Stencils","title":"Stencils.center","text":"center(x::Stencil)\n\nReturn the value of the central cell a stencil is offset around. It may or may not be part of the stencil itself.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.offsets","page":"Stencils","title":"Stencils.offsets","text":"offsets(x)\n\nReturn an SVector of NTuple{N,Int}, containing all positions in the stencil as offsets from the central cell.\n\nCustom Stencils must define this method.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.indices","page":"Stencils","title":"Stencils.indices","text":"indices(x::Stencil, I::Union{Tuple,CartesianIndex})\nindices(x::AbstractStencilArray, I::Union{Tuple,CartesianIndex})\n\nReturns an SVector of CartesianIndices for each neighbor around I.\n\nindices for Stencil do not know about array boundaries and wil not wrap or reflect. On AbstractStencilArray they will wrap and reflect depending on the boundary condition  of the array.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.distances","page":"Stencils","title":"Stencils.distances","text":"distances(hood::Stencil)\n\nReturns an SVector of center-to-center distance of each stencil position from the central cell, so that horizontally or vertically adjacent cells have a distance of 1.0, and a diagonally adjacent cell has a distance of sqrt(2.0).\n\nValues are calculated at compile time, so distances can be used with little overhead.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.radius","page":"Stencils","title":"Stencils.radius","text":"radius(stencil) -> Int\n\nReturn the radius of a stencil.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.diameter","page":"Stencils","title":"Stencils.diameter","text":"diameter(rule) -> Int\n\nThe diameter of a stencil is 2r + 1 where r is the radius.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.distance_zones","page":"Stencils","title":"Stencils.distance_zones","text":"distance_zones(hood::Stencil)\n\nReturns an SVector of Int distance zones for each offset in the Stencil.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.kernel","page":"Stencils","title":"Stencils.kernel","text":"kernel(hood::AbstractKernelStencil) => iterable\n\nReturns the kernel object, an array or iterable matching the length of the stencil.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.kernelproduct","page":"Stencils","title":"Stencils.kernelproduct","text":"kernelproduct(hood::AbstractKernelStencil)\nkernelproduct(hood::Stencil, kernel)\n\nReturns the vector dot product of the stencil and the kernel, although differing from dot in that it is not taken iteratively for members of the stencil - they are treated as scalars.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.unsafe_neighbors","page":"Stencils","title":"Stencils.unsafe_neighbors","text":"unsafe_neighbors([hood::Stencil,] A::AbstractStencilArray, I::CartesianIndex) => SArray\n\nGet stencil neighbors from A around center I as an SVector, without checking bounds of I.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.unsafe_stencil","page":"Stencils","title":"Stencils.unsafe_stencil","text":"unsafe_stencil(x, A::AbstractArray, I) => Stencil\n\nUpdate the neighbors of a stencil to values from the array A around index I, without checking bounds of I. Bounds checking of neighbors still occurs, but with the assumption that I is inbounds.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.rebuild","page":"Stencils","title":"Stencils.rebuild","text":"rebuild(x::Stencil, neighbors::StaticArray)\n\nRebuild a Stencil, returning an stencil of the same size and shape, with new neighbor values.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.getneighbor","page":"Stencils","title":"Stencils.getneighbor","text":"getneighbor(A::AbstractStencilArray, I::CartesianIndex)\n\nGet an array value from a stencil neighborhood.\n\nThis method handles boundary conditions.\n\n\n\n\n\n","category":"function"},{"location":"#Boundary-Conditions","page":"Stencils","title":"Boundary Conditions","text":"","category":"section"},{"location":"","page":"Stencils","title":"Stencils","text":"Stencils.BoundaryCondition\nRemove\nUse\nWrap\nReflect","category":"page"},{"location":"#Stencils.BoundaryCondition","page":"Stencils","title":"Stencils.BoundaryCondition","text":"BoundaryCondition\n\nAbstract supertype for flags that specify the boundary conditions. These determine what happens when a stencil extends outside of the grid.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Remove","page":"Stencils","title":"Stencils.Remove","text":"Remove <: BoundaryCondition\n\nRemove()\n\nBoundaryCondition flag that specifies to assign padval to cells that overflow grid boundaries.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Use","page":"Stencils","title":"Stencils.Use","text":"Use <: BoundaryCondition\n\nUse()\n\nBoundaryCondition flag that specifies to use the existing  padding, which is only possible when Halo{:in} is used for padding.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Wrap","page":"Stencils","title":"Stencils.Wrap","text":"Wrap <: BoundaryCondition\n\nWrap()\n\nBoundaryCondition flag to wrap cordinates that boundary boundaries  back to the opposite side of the grid.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Reflect","page":"Stencils","title":"Stencils.Reflect","text":"Reflect <: BoundaryCondition\n\nReflect()\n\nBoundaryCondition flag to mirror cordinates that boundary boundaries  back to the source cell of the grid.\n\n\n\n\n\n","category":"type"},{"location":"#Padding","page":"Stencils","title":"Padding","text":"","category":"section"},{"location":"","page":"Stencils","title":"Stencils","text":"Stencils.Padding\nConditional\nHalo","category":"page"},{"location":"#Stencils.Padding","page":"Stencils","title":"Stencils.Padding","text":"Padding\n\nAbstract supertype for padding modes, e.g.  Conditional and Halo.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Conditional","page":"Stencils","title":"Stencils.Conditional","text":"Conditional <: Padding\n\nPadding that doesn't change the array size, but checks getindex for out-of-bounds indexing, and inserts padval with Remove or values from the other side of the array with Wrap.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.Halo","page":"Stencils","title":"Stencils.Halo","text":"Halo{X} <: Padding\n\nPadding that uses an in-memory halo around the array so that parts of a stencil that go off the edge of the array can index directly into it without a bounds check or any conditional. This has the benefit of possibly better performance during window broadcasts, but some downsides.\n\nIn :out mode, a whole new array is alocated, larger than the original. This may not be worth doing unless you are using it multiple times. with :in mode, the outside edge of the array is used as padding. This may be more accurate  as there are no boundary effects from using a padding value.:w\n\nExample\n\nhalo_in = Halo(:in)\nhalo_out = Halo(:out)\n\n\n\n\n\n","category":"type"},{"location":"#Stencil-Arrays","page":"Stencils","title":"Stencil Arrays","text":"","category":"section"},{"location":"","page":"Stencils","title":"Stencils","text":"Stencils.AbstractStencilArray\nStencilArray\nStencils.AbstractSwitchingStencilArray\nSwitchingStencilArray","category":"page"},{"location":"#Stencils.AbstractStencilArray","page":"Stencils","title":"Stencils.AbstractStencilArray","text":"AbstractStencilArray <: StaticArray\n\nSupertype for arrays with a Stencil, a BoundaryCondition, and Padding.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.StencilArray","page":"Stencils","title":"Stencils.StencilArray","text":"StencilArray <: AbstractStencilArray\n\nStencilArray(A::AbstractArray, stencil::Stencil; kw...)\n\nAn array with a Stencil and a BoundaryCondition, and Padding.\n\nFor most uses a StencilArray works exactly the same as a regular array.\n\nExcept it can be indexed at any point with stencil to return a filled Stencil object, or neighbors to return an SVector of neighbors.\n\nArguments\n\nA: an AbstractArray\nstencil: a Stencil.\n\nKeywords\n\nboundary: a BoundaryCondition like Wrap.\npadding: Padding like Conditional or Halo{:in}.\n\nExample\n\nusing Stencils, Statistics\nsa = StencilArray((1:10) * (10:20)', Moore(1); boundary=Wrap())\nsa .*= 2 # Broadcast works as usual\nmeans = mapstencil(mean, sa) # mapstencil works\nstencil(sa, 5, 6) # manually reading a stencil works too\n\n# output\n\nMoore{1, 2, 8, Int64}\n█▀█\n▀▀▀\n\nwith neighbors:\n8-element StaticArraysCore.SVector{8, Int64} with indices SOneTo(8):\n 112\n 140\n 168\n 120\n 180\n 128\n 160\n 192\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.AbstractSwitchingStencilArray","page":"Stencils","title":"Stencils.AbstractSwitchingStencilArray","text":"AbstractSwitchingStencilArray\n\nAbstract supertype for AbstractStencilArray that wrap two arrays that switch places with each broadcast.\n\n\n\n\n\n","category":"type"},{"location":"#Stencils.SwitchingStencilArray","page":"Stencils","title":"Stencils.SwitchingStencilArray","text":"SwitchingStencilArray <: AbstractSwitchingStencilArray\n\nAn AbstractArray with a Stencil, a BoundaryCondition, Padding, and two array layers that are switched with each broadcast_stencil operation.\n\nThe use case for this operation is in simulations where stencil operations are repeatedly run over the same data, or where a filter (such as a blur) needs to be applied many times.\n\nFor most uses a SwitchingStencilArray works exactly the same as a regular array - the dest array can be safely ignored.\n\nHowever, when using mapstencil! you need to use the output, not the original array. Switching does not happen in-place, but as a new returned array.\n\nExample\n\nusing Stencils, Statistics\n\nsa = SwitchingStencilArray(rand(10, 10), Moore(1); boundary=Wrap())\nsa .*= 2 # Broadcast works as usual\nmapstencil(mean, sa) # As does runing `mapstencils\nhood = stencil(sa, 5, 10) # And retreiving a stencil\n# But we can also run it in-place, here doing 10 iterations of mean blur:\n# Note: if you dont assign new variable with `A =`, the array will\n# not switch and will not be blurred.\nlet sa = sa\n    for i in 1:10\n        sa = mapstencil!(mean, sa)\n    end\nend\n# output\n\n\n\n\n\n\n","category":"type"},{"location":"","page":"Stencils","title":"Stencils","text":"Methods on stencil arrays:","category":"page"},{"location":"","page":"Stencils","title":"Stencils","text":"Stencils.boundary\nStencils.padding\nStencils.switch","category":"page"},{"location":"#Stencils.boundary","page":"Stencils","title":"Stencils.boundary","text":"boundary(A::AbstractStencilArray)\n\nGet the BoundaryCondition object from an AbstractStencilArray.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.padding","page":"Stencils","title":"Stencils.padding","text":"padding(A::AbstractStencilArray)\n\nGet the Padding object from an AbstractStencilArray.\n\n\n\n\n\n","category":"function"},{"location":"#Stencils.switch","page":"Stencils","title":"Stencils.switch","text":"switch(A::SwitchingStencilArray)\n\nSwap the source and dest of a SwitchingStencilArray.\n\n\n\n\n\n","category":"function"}]
}
