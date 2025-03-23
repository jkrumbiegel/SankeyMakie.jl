<div align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" 
        srcset="docs/src/assets/logo_with_text_dark.svg" >
      <img alt="SankeyMakie.jl logo" 
        src="docs/src/assets/logo_with_text.svg" height="50">
    </picture>
</div>

<div align="center">

A Makie port of [https://github.com/daschw/SankeyPlots.jl](https://github.com/daschw/SankeyPlots.jl)

## Example

```julia
using SankeyMakie
using CairoMakie
using Random
Random.seed!(123)

connections = [
    (1, 2, 1100),
    (2, 4, 300),
    (6, 2, 1400),
    (2, 3, 500),
    (2, 5, 300),
    (5, 7, 100),
    (2, 8, 100),
    (3, 9, 150),
    (2, 10, 500),
    (10, 11, 50),
    (10, 12, 80),
    (5, 13, 150),
    (3, 14, 100),
    (10, 15, 300),
]

labels = [
    "Salary",
    "Income",
    "Rent",
    "Insurance",
    "Car",
    "Salary 2",
    "Depreciation",
    "Internet",
    "Electricity",
    "Food & Household",
    "Fast Food",
    "Drinks",
    "Gas",
    "Water",
    "Groceries",
]

f, ax, s = sankey(connections,
    nodelabels = labels,
    nodecolor = rand(RGBf, length(labels)),
    linkcolor = SankeyMakie.TargetColor(0.2),
    figure = (; resolution = (1000, 500)))
hidedecorations!(ax)
hidespines!(ax)

save("sankey.svg", f)
```

![sankey example](sankey.svg)