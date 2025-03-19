function test_connections_and_labels()
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
    connections, labels
end

reftest("basic sankey") do
    connections, labels = test_connections_and_labels()
    fig, ax, plt = sankey(
        connections,
        nodelabels = labels,
        nodecolor = Makie.to_colormap(:tab20)[1:length(labels)],
        linkcolor = SankeyMakie.TargetColor(0.2),
    )
    hidespines!(ax); hidedecorations!(ax)
    return fig
end

reftest("sankey standard `forceorder`") do
    connections, labels = test_connections_and_labels()
    fig, ax, plt = sankey(
        connections,
        nodelabels = labels,
        nodecolor = Makie.to_colormap(:tab20)[1:length(labels)],
        linkcolor = SankeyMakie.TargetColor(0.2),
        forceorder = [6 => 1],
    )
    hidespines!(ax); hidedecorations!(ax)
    return fig
end

reftest("sankey `forceorder = :reverse`") do
    connections, labels = test_connections_and_labels()
    fig, ax, plt = sankey(
        connections,
        nodelabels = labels,
        nodecolor = Makie.to_colormap(:tab20)[1:length(labels)],
        linkcolor = SankeyMakie.TargetColor(0.2),
        forceorder = :reverse,
    )
    hidespines!(ax); hidedecorations!(ax)
    return fig
end

reftest("larger fontsize") do
  connections, labels = test_connections_and_labels()
    fig, ax, plt = sankey(
        connections,
        nodelabels = labels,
        nodecolor = Makie.to_colormap(:tab20)[1:length(labels)],
        linkcolor = SankeyMakie.TargetColor(0.2),
        fontsize = 24,
    )
    hidespines!(ax); hidedecorations!(ax)
    return fig
end

reftest("masked nodes") do
    links = [
        (1, 2, 150),
        (1, 3, 100),
        (2, 3, 200),
        (2, 4, 50),
        (1, 4, 50),
        (3, 4, 50),
    ]
        
    fig, ax, plt = sankey(
        links;
        nodelabels = ["A", "B", "C", "D"],
        linkcolor = Makie.to_colormap([:tomato, :bisque, :teal, :blue, :pink, :orange])
    )
    hidespines!(ax); hidedecorations!(ax)
    return fig
end
