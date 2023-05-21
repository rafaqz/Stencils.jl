# Stencils

Main functions

```@docs
mapstencil
mapstencil!
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
Layered
Moore
Vertical
VonNeumann
Window
```

## Stencil functions

These can be called on any `<: Stencil` object, and `Layered`.

```@docs
neighbors
offsets
indices
distances
distance_zones
radius
diameter
kernelproduct
update_stencil
unsafe_update_stencil
setneighbors
```

## Boundary Conditions

`````@docs
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
AbstractSwitchStencilArray
SwitchingStencilArray
```
