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

f = Figure(size=(500,500))
ax = Axis(f[1, 1])

sankey!(
    connections,
    nodelabels = labels,
    nodecolor = rand(RGBf, length(labels)),
    linkcolor = SankeyMakie.TargetColor(0.2),
    fontsize = 20,
)
xlims!(0.6, 4.6)
ylims!(-0.1, 1.15)
#hidedecorations!(ax)
#hidespines!(ax)
display(f)
