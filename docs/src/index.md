# Stencils

```@autodocs
Modules = [Stencils]
Order = [:module, :type, :function]
```

## Stencils

```@docs
Stencil
AngledCross
BackSlash
Circle
Cross
ForwardSlash
Horizontal
Moore
Vertical
VonNeumann
Layout
```

## Stencil functions

These can be called on any `<: Stencil` object, and `Layered`.

```@docs
neighbors
setneighbors
offsets
indices
distances
distance_zones
radius
diameter
kernelproduct
```

## Boundary Conditions

`````@docs`
BoundaryCondition
Wrap
Remove
```

## Padding

```@docs
Padding
Conditional
Halo
```

## Stencil Arrays

```@docs
AbstractStencilArray
StencilArray
SwitchingStencilArray
```

## Array methods

```@docs
broadcast_stencil
broadcast_stencil!
switch
```


export Stencil, Window, Kernel, Moore, VonNeumann, Positional, Layered, 
    Circle, Cross, AngledCross, BackSlash, ForwardSlash, Vertical, Horizontal
export StencilArray
export BoundaryCondition, Wrap, Remove
export Padding, Conditional, Halo

export stencil, neighbors, offsets, indices, distances, radius, diameter, kernel, kernelproduct
export broadcast_stencil, broadcast_stencil!
