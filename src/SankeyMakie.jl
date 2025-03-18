module SankeyMakie

using LayeredLayouts
using Graphs, MetaGraphs
using SparseArrays
using Makie

export sankey, sankey!

# """
#     sankey(src, dst, weights; kwargs..., plotattributes...)
#     sankey(g::AbstractMetaGraph; kwargs..., plotattributes...)

# Plot a sankey diagram.

# In addition to [Plots.jl attributes](http://docs.juliaplots.org/latest/attributes/) the following keyword arguments are supported.

# | Keyword argument | Default value | Options |
# |---|---|----|
# | `node_labels` | `nothing` | `AbstractVector{<:String}` |
# | `node_colors` | `nothing` | Vector of [color specifications supported by Plots.jl](http://docs.juliaplots.org/latest/colors/) or [color palette](http://docs.juliaplots.org/latest/generated/colorschemes/#ColorPalette) |
# | `edge_color` | `:gray` | Plots.jl supported [color](http://docs.juliaplots.org/latest/colors/), color selection from connected nodes with `:src`, `:dst`, `:gradient`, or an `AbstractDict{Tuple{Int, Int}, Any}` where `edge_color[(src, dst)]` maps to a color |
# | `label_position` | `:inside` | `:legend`, `:node`, `:left`, `:right`, `:top` or `:bottom` |
# | `label_size` | `8` | `Int` |
# | `compact` | `false` | `Bool` |
# | `force_layer` | `Vector{Pair{Int,Int}}()` | Vectors of Int pairs specifying the layer for every node e.g. `[4=>2]` to force node 4 in layer 3 |
# | `force_order` | `Vector{Pair{Int,Int}}()` | Vectors of Int pairs specifying the node ordering in each layer e.g. `[1=>2]` to specify node 1 preceeds node 2 in the same layer |"""

@recipe(Sankey) do scene
    Attributes(
        labelposition=:inside,
        compact = true,
        fontsize = theme(scene, :fontsize),
        nodelabels = nothing,
        nodecolor = :gray30,
        linkcolor = (:pink, 0.2),
        forceorder = Pair{Int,Int}[],
    )
end
#     node_labels=nothing,
#     node_colors=nothing,
#     edge_color=:gray,
#     
#     label_size=8,
#     compact=false,
#     force_layer::Vector{Pair{Int,Int}}=Vector{Pair{Int,Int}}(),
#     force_order::Vector{Pair{Int,Int}}=Vector{Pair{Int,Int}}(),
# )

function Makie.plot!(s::Sankey)
    g = sankey_graph(s[1][])
    linkindexdict = Dict(tuple.(first.(s[1][]), getindex.(s[1][], 2)) .=> eachindex(s[1][]))
    labels = sankey_names(g, s.nodelabels[])
    # if node_colors === nothing
    #     node_colors = palette(get(plotattributes, :color_palette, :default))
    # end

    scene = Makie.parent_scene(s)
    wbox = 0.03

    force_layer = Pair{Int,Int}[]

    x, y, mask = sankey_layout!(g, force_layer, s.forceorder[])
    perm = sortperm(y, rev=true)

    vw = vertex_weight.(Ref(g), vertices(g))
    m = maximum(vw)

    if s.compact[] == true
        y = make_compact(x, y, vw / m)
    end

    src_offsets = get_src_offsets(g, perm) ./ m
    dst_offsets = get_dst_offsets(g, perm) ./ m

    heights = vw ./ 2m

    # if label_position ∉ (:inside, :left, :right, :top, :bottom, :node, :legend)
    #     error("label_position :$label_position not supported")
    # elseif label_position !== :legend
    #     xlabs = Float64[]
    #     ylabs = Float64[]
    #     lab_orientations = Symbol[]
    # end

    for (i, v) in enumerate(vertices(g))
        h = heights[i]

        if !(mask[i])
            poly!(s, BBox(x[i]-wbox, x[i]+wbox, y[i]-h, y[i]+h), color = get_node_color(s.nodecolor[], i))

            for (j, w) in enumerate(vertices(g))
                if has_edge(g, v, w)
                    y_src = y[i] + h - src_offsets[i, j]
                    edge_it = Edge(v, w)
                    h_edge = get_prop(g, edge_it, :weight) / (2m)

                    sankey_y = Float64[]
                    x_start = x[i] + wbox
                    xvals = [x_start]
                    yvals = [y_src]
                    k = j
                    l = i
                    while mask[k]
                        y_dst = y[k] + vw[k] / (2m) - dst_offsets[k, l]
                        # x_coords = range(0, 1, length=length(x_start:0.01:x[k]))
                        x_coords = range(0, 1, length=3)
                        y_coords =
                            remap(1 ./ (1 .+ exp.(6 .* (1 .- 2 .* x_coords))), y_src, y_dst)

                        append!(sankey_y, y_coords)

                        x_start = x[k] + 0.01
                        push!(xvals, x_start)
                        y_src = y_dst
                        push!(yvals, y_src)
                        l = k
                        k = findfirst(==(first(outneighbors(g, k))), vertices(g))
                    end
                    push!(xvals, x[k]-wbox)

                    y_dst = y[k] + vw[k] / (2m) - dst_offsets[k, l]
                    push!(yvals, y_dst)
                    x_coords = range(0, 1, length=3)
                    # x_coords = range(0, 1, length=length(x_start:0.01:x[k]-wbox))
                    y_coords = remap(1 ./ (1 .+ exp.(6 .* (1 .- 2 .* x_coords))), y_src, y_dst)
                    append!(sankey_y, y_coords)
                    sankey_x = range(x[i]+wbox, x[k]-wbox, length = length(sankey_y))

                    pol = linkpoly(s, scene, xvals, yvals .- h_edge, 2h_edge)

                    poly!(s, pol, color = get_link_color(s.linkcolor[], i, j, linkindexdict[(i, j)], s.nodecolor[]), space = :pixel)
                    # band!(s, sankey_x, sankey_y.-2h_edge, sankey_y, color = (:black, 0.1))

    #                 missing_keys = Tuple{Int64, Int64}[]
    #                 @series begin
    #                     seriestype := :path
    #                     primary := false
    #                     linecolor := nothing
    #                     linewidth := false
    #                     fillrange := sankey_y .- 2h_edge
    #                     fillalpha --> 0.5
    #                     if edge_color === :gradient
    #                         fillcolor := getindex(
    #                             cgrad(node_colors[mod1.([i, k], end)]),
    #                             range(0, 1, length=length(sankey_x)),
    #                         )
    #                     elseif edge_color === :src
    #                         fillcolor := node_colors[mod1(i, end)]
    #                     elseif edge_color === :dst
    #                         fillcolor := node_colors[mod1(k, end)]
    #                     elseif typeof(edge_color) <: AbstractDict{Tuple{Int, Int}}
    #                         if haskey(edge_color, (i, k))
    #                             fillcolor := edge_color[(i, k)]
    #                         else
    #                             push!(missing_keys, (i, k))
    #                             fillcolor := :gray
    #                         end
    #                     else
    #                         fillcolor := edge_color
    #                     end
    #                     sankey_x, sankey_y
    #                 end
    #                 isempty(missing_keys) || @warn "The following missing keys in the edge_color dictionary defaulted to gray" missing_keys
                end
            end

    #         if label_position !== :legend
    #             xlab, orientation = if label_position in (:node, :top, :bottom)
    #                 olab = if label_position === :top
    #                     :bottom
    #                 elseif label_position === :bottom
    #                     :top
    #                 else
    #                     :center
    #                 end
    #                 x[i], olab
    #             elseif label_position === :right ||
    #                     (label_position === :inside && x[i] < maximum(x))
    #                 x[i] + 0.15, :left
    #             elseif label_position === :left ||
    #                     (label_position === :inside && x[i] == maximum(x))
    #                 x[i] - 0.15, :right
    #             else
    #                 error("label_position :$label_position not supported")
    #             end
    #             ylab = if label_position === :top
    #                 y[i] + h + 0.02
    #             elseif label_position === :bottom
    #                 y[i] - h - 0.0025 * label_size
    #             else
    #                 y[i]
    #             end
    #             push!(xlabs, xlab)
    #             push!(ylabs, ylab)
    #             push!(lab_orientations, orientation)
    #         end
        end
    end

    text!(s, x[.!mask], y[.!mask] .- heights[.!mask], text = labels, align = (:center, :top), fontsize = s.fontsize[])

    # if label_position !== :legend
    #     @series begin
    #         primary := :false
    #         seriestype := :scatter
    #         markeralpha := 0
    #         series_annotations := text.(names, lab_orientations, label_size)
    #         xlabs, ylabs
    #     end

    #     # extend axes for labels
    #     if label_position in (:left, :right)
    #         x_extra = label_position === :left ? minimum(xlabs) - 0.4 : maximum(xlabs) + 0.5
    #         @series begin
    #             primary := false
    #             seriestype := :scatter
    #             markeralpha := 0
    #             [x_extra], [ylabs[1]]
    #         end
    #     end
    # end

    # primary := false
    # framestyle --> :none
    # legend --> label_position === :legend ? :outertopright : false
    # ()

    return s
end

function add_weighted_edge!(g, src, dst, weight)
    new_edge = Edge(src, dst)
    add_edge!(g, new_edge)
    set_prop!(g, new_edge, :weight, weight)
    return g
end

function sankey_graph(src::Vector, dst::Vector, w)
    # get list of unique nodes
    unique_nodes = sort(unique([src; dst]))

    n_nodes = length(unique_nodes)
    n_edges = length(src)
    list_nodes = 1:n_nodes

    # process src and dst to avoid missing nodes

    # Parse src and dst to match all ids in unique_nodes
    parser_dict = Dict(unique_nodes[id]=>id for id = 1:length(unique_nodes))
    
    src = [parser_dict[src_val] for src_val in src]
    dst = [parser_dict[dst_val] for dst_val in dst]

    # verify length of vectors
    @assert length(src)==length(dst)==length(w)  "Mismatch in the lengths of input parameters"

    # initialize graph
    g = MetaDiGraph(n_nodes)

    # add edges iteratively
    for i = 1:n_edges
        add_weighted_edge!(g, src[i], dst[i], w[i])
    end

    return g
end
sankey_graph(g::AbstractMetaGraph) = copy(g)
function sankey_graph(v::Vector{<:Tuple{Int,Int,Real}})
    sankey_graph(getindex.(v, 1), getindex.(v, 2), getindex.(v, 3))
end

get_node_color(s::Symbol, i) = s
get_node_color(v::AbstractVector{<:Makie.Colors.Colorant}, i) = v[i]

get_link_color(v::AbstractVector{<:Makie.Colors.Colorant}, i, j, k) = v[k]
get_link_color(x, i, j, k) = x
get_link_color(x, i, j, k, nodecolor) = get_link_color(x, i, j, k)

struct SourceColor
    alpha::Float64
end
struct TargetColor
    alpha::Float64
end

get_link_color(t::TargetColor, i, j, k, nodecolor) = (get_node_color(nodecolor, j), t.alpha)
get_link_color(t::SourceColor, i, j, k, nodecolor) = (get_node_color(nodecolor, i), t.alpha)


sankey_names(g, names) = names
sankey_names(g, ::Nothing) = string.("Node", eachindex(vertices(g)))

function sankey_layout!(g, forcelayer, forceorder::Vector{Pair{Int,Int}})
    xs, ys, paths = solve_positions(
        Zarate(), g, force_layer=forcelayer, force_order=forceorder
    )
    
    mask = insert_masked_nodes!(g, xs, ys, paths)

    if !isempty(forceorder)
        reorder_nodes!(ys, xs, mask, forceorder)
    end

    return xs, ys, mask
end

function sankey_layout!(g, forcelayer, forceorder::Symbol)
    xs, ys, paths = solve_positions(
        Zarate(), g, force_layer=forcelayer, force_order=Pair{Int,Int}[]
    )

    mask = insert_masked_nodes!(g, xs, ys, paths)
    
    if forceorder == :reverse
        layers = nodes_by_layers(xs, mask)
        reverse_nodes!(ys, layers)
    end

    return xs, ys, mask
end

function insert_masked_nodes!(g, xs, ys, paths)
    mask = falses(length(xs))
    for (edge, path) in paths
        s = edge.src
        px, py = path
        weight_path = get_prop(g, edge, :weight)
        if length(px) > 2
            for i in 2:length(px)-1
                add_vertex!(g)
                v = last(vertices(g))
                add_weighted_edge!(g, s, v, weight_path)
                push!(xs, px[i])
                push!(ys, py[i])
                push!(mask, true)
                s = v
            end
            add_weighted_edge!(g, s, edge.dst, weight_path)
            rem_edge!(g, edge)
        end
    end
    return mask
end

# ys as first arg because it's mutated?
function reorder_nodes!(ys, xs, mask, forceorder)
    for (node1, node2) in forceorder
        idx1 = findfirst(i -> !mask[i] && i == node1, eachindex(xs))
        idx2 = findfirst(i -> !mask[i] && i == node2, eachindex(xs))

        if idx1 !== nothing && idx2 !== nothing
            x1, x2 = xs[idx1], xs[idx2]
            if x1 != x2
                @warn "`forceorder = [$node1 => $node2]` failed; nodes must be in the same layer."
                continue
            elseif ys[idx1] < ys[idx2]
                @warn "`forceorder = [$node1 => $node2]` failed; nodes are already in this order."
                continue
            end
            ys[idx1], ys[idx2] = ys[idx2], ys[idx1]
        end
    end
    return nothing
end

function reverse_nodes!(ys, layers)
    for (_, indices) in layers
        if length(indices) < 2
            continue
        end
        sorted_indices = sort(indices, by=i -> ys[i])
        y_positions = ys[sorted_indices]
        for (i, idx) in enumerate(sorted_indices)
            ys[idx] = y_positions[end-i+1]
        end
    end
    return nothing
end

function nodes_by_layers(xs, mask)
    layers = Dict{Float64,Vector{Int}}()
    for (i, x) in enumerate(xs)
        if !mask[i]
            if !haskey(layers, x)
                layers[x] = Int[]
            end
            push!(layers[x], i)
        end
    end
    return layers
end

function vertex_weight(g, v)
    max(
        sum0(x->get_prop(g, x, :weight), Iterators.filter(e -> src(e) == v, edges(g))),
        sum0(x->get_prop(g, x, :weight), Iterators.filter(e -> dst(e) == v, edges(g))),
    )
end
sum0(f, x) = isempty(x) ? 0.0 : sum(f, x)

in_edges(g, v) = Iterators.filter(e -> dst(e) == v, edges(g))
out_edges(g, v) = Iterators.filter(e -> src(e) == v, edges(g))

function get_src_offsets(g, perm)
    verts = vertices(g)
    n = nv(g)
    p = spzeros(n, n)
    for (i, v) in enumerate(verts)
        offset = 0.0
        for j in perm
            if has_edge(g, v, verts[j])
                if offset > 0
                    p[i, j] = offset
                end
                edge_it = Edge(v, verts[j])
                # add to offset if edge is available
                offset += get_prop(g, edge_it, :weight)
            end
        end
    end
    return p
end

function get_dst_offsets(g, perm)
    verts = vertices(g)
    n = nv(g)
    p = spzeros(n, n)
    for (i, v) in enumerate(verts)
        offset = 0.0
        for j in perm
            if has_edge(g, verts[j], i)
                if offset > 0
                    p[i, j] = offset
                end
                offset += get_prop(g, Edge(verts[j], i), :weight)
            end
        end
    end
    return p
end

function remap(x, lo, hi)
    xlo, xhi = extrema(x)
    lo .+ (x .- xlo) / (xhi - xlo) * (hi - lo)
end

function make_compact(x, y, w)
    x = round.(Int, x)
    ux = unique(x)
    heights = zeros(length(ux))
    uinds = [findall(==(uxi), x) for uxi in ux]
    for (i, inds) in enumerate(uinds)
        perm = sortperm(view(y, inds))
        start = 0
        for j in inds[perm]
            y[j] = start + w[j] / 2
            start += 0.1 + w[j]
        end
        heights[i] = start - 0.1
    end
    maxh = maximum(heights)
    for (i, inds) in enumerate(uinds)
        y[inds] .+= (maxh - heights[i]) / 2
    end
    return y
end

function linkpoly(plt, scene, xs, ys, lwidth)
    n = 30

    lift(scene.camera.projectionview, scene.viewport) do _, _

        nparts = length(xs)-1
        # points = Vector{Point2f}(undef, 2 * n * nparts)
        points = fill(Point2f(1, 1), 2 * n * nparts)

        for (ipart, (x0, x1, y0, y1)) in enumerate(zip(
                @view(xs[1:end-1]),
                @view(xs[2:end]),
                @view(ys[1:end-1]),
                @view(ys[2:end]),
            ))

            corners = Makie.plot_to_screen(
                plt,
                Point2f[
                    (x0, y0 - lwidth/2),
                    (x1, y1 - lwidth/2),
                    (x1, y1 + lwidth/2),
                    (x0, y0 + lwidth/2),
                ],
            )

            start = 0.5 * (corners[1] + corners[4])
            stop =  0.5 * (corners[2] + corners[3])
            thickness = corners[4][2] - corners[1][2]
            width = stop[1] - start[1]
            height = stop[2] - start[2]


            x0pix = start[1]
            x1pix = stop[1]

            for (i, x) in enumerate(range(x0pix, x1pix, length = n))

                y = height * (1 - cos(pi * (x - x0pix) / width)) / 2 + start[2]
                deriv = pi * height * sin((pi * (x - x0pix)) / width) / (2 * width)

                xortho = sqrt(1 / (1 + deriv^2))
                yortho = xortho * deriv

                ortho = Point2f(-yortho, xortho)
                _xy = Point2f(x, y)

                if nparts > 1
                    points[i + (ipart-1) * n] = _xy - ortho * thickness/2
                    points[end - (ipart-1) * n - i + 1] = _xy + ortho * thickness/2
                else
                    points[i + (ipart-1) * n] = _xy - ortho * thickness/2
                    points[end - (ipart-1) * n - i + 1] = _xy + ortho * thickness/2
                end
            end
        end

        Makie.Polygon(points)
    end
end

Makie.data_limits(s::Sankey) = reduce(union, [Makie.data_limits(p) for p in s.plots if !(haskey(p, :space) && p.space[] === :pixel)])
Makie.boundingbox(s::Sankey, space::Symbol = :data) = Makie.apply_transform_and_model(s, Makie.data_limits(s))


end
