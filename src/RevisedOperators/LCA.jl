"""
    rerooted_parent_map(net::AbstractNetwork, root::Tuple{Int,Int})

Returns a Dict mapping each node to its parent in the tree rooted at `root`.
"""
function rerooted_parent_map(net::AbstractNetwork, root::Tuple{Int,Int})
    visited = Set{Tuple{Int,Int}}()
    parent = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    stack = [root]

    while !isempty(stack)
        node = pop!(stack)
        push!(visited, node)

        for child in child_nodes(net, node) |> x -> x === nothing ? [] : x
            if !(child in visited)
                parent[child] = node
                push!(stack, child)
            end
        end

        p = parent_node(net, node)
        if p !== nothing && !(p in visited)
            parent[p] = node
            push!(stack, p)
        end
    end
    parent[root] = root  # Ensure the root maps to itself

    return parent
end

"""
    reverse_bfs_nodes(net::AbstractNetwork, root::Tuple{Int,Int})

Return a Vector of all nodes `(layer,node)` in the tree rooted at `root`,
ordered by *decreasing* distance from `root` (i.e. reverse‐BFS order).
If two nodes share the same distance, the one with the smaller `layer` comes first.
Only nodes with `layer ≥ 1` are included.
"""
function reverse_bfs_nodes(net::TTN.AbstractNetwork, root::Tuple{Int,Int})
    # 1) build the rerooted parent map
    parent = rerooted_parent_map(net, root)

    # 2) compute depth (distance) of every node to `root`
    depths = Dict{Tuple{Int,Int}, Int}()
    for node in keys(parent)
        d = 0
        cur = node
        while cur != root
            cur = parent[cur]
            d += 1
        end
        depths[node] = d
    end

    # 3) collect only layer ≥ 1
    allnodes = [n for n in keys(parent) if n[1] ≥ 1]

    # 4) sort by (–depth, layer)
    sort!(allnodes, by = n -> (-depths[n], n[1]))

    return allnodes
end

"""
    next_on_path(path::Vector{Tuple{Int,Int}}, n::Tuple{Int,Int})

Return the node that comes *after* `n` in `path`, or `nothing` if `n` is the last element.
Throws an error if `n ∉ path`.
"""
function next_on_path(path::Vector{Tuple{Int,Int}}, n::Tuple{Int,Int})
    idx = findfirst(==(n), path)
    @assert idx !== nothing "node $n is not in the path"
    return idx < length(path) ? path[idx+1] : nothing
end

"""
    subpath_from(path::Vector{Tuple{Int,Int}}, n::Tuple{Int,Int})

Return the slice of `path` starting at `n` (inclusive) through the end.
Throws an error if `n ∉ path`.
"""
function subpath_from(path::Vector{Tuple{Int,Int}}, n::Tuple{Int,Int})
    idx = findfirst(==(n), path)
    @assert idx !== nothing "node $n is not in the path"
    return path[idx:end]
end


"""
    lowest_common_ancestor_node_links(net, site1, site2, root)

Returns:
- `lca`: rerooted lowest common ancestor of site1 and site2
- `leg1`: which leg of the LCA leads to site1 (0=parent, 1=child1, 2=child2)
- `leg2`: same for site2
"""
function lowest_common_ancestor_node_links(net::AbstractNetwork, site1, site2, root)
    parent_map = rerooted_parent_map(net, root)

    # Build paths to root
    function path_to_root(site)
        path = Tuple[]
        current = site
        while current ≠ root
            push!(path, current)
            current = parent_map[current]
        end
        push!(path, root)
        return reverse(path)
    end

    path1 = path_to_root(site1)
    path2 = path_to_root(site2)

    # Find LCA
    lca = root
    for (p1, p2) in zip(path1, path2)
        if p1 == p2
            lca = p1
        else
            break
        end
    end

    # Find the next node after LCA on the path to site
    function leg_index_from_lca(net, path, lca)
        # Find the index of the LCA in the path
        idx = findfirst(x -> x == lca, path)
        next_node = path[idx + 1]  # always exists since site ≠ lca
        children = child_nodes(net, lca)
        if children !== nothing
            if next_node == children[1]
                return 1
            elseif next_node == children[2]
                return 2
            end
        end
        if parent_node(net, lca) == next_node
            return 0
        end
        error("Node $next_node is not a neighbor of LCA $lca in original tree")
    end

    leg1 = leg_index_from_lca(net, path1, lca)
    leg2 = leg_index_from_lca(net, path2, lca)

    return (lca, leg1, leg2)
end


function build_lca_id_map(net::AbstractNetwork, tpo::TPO_GPU)
    lca_id_map = Dict{Int, Dict{Tuple{Int,Int}, LCA}}()
    # Iterate over all pairs of sites in the TPO
    for idd in 1:tpo.terms[end].id
        op = get_id_terms(tpo, idd)
        if length(op) == 2
            site1, site2 = op[1].site, op[2].site
            ## Ensure site1 < site2 for consistent ordering, already done
            # site_pair = site1 < site2 ? (site1, site2) : (site2, site1)
            for l in 1:number_of_layers(net)
                for n in 1:number_of_tensors(net, l)
                    root = (l, n)
                    # Find the lowest common ancestor for the pair of sites
                    lca, link1, link2 = lowest_common_ancestor_node_links(net, site1, site2, root)

                    # Store the LCA information in a dictionary
                    get!(lca_id_map, idd, Dict())[root] = LCA(lca, (link1, link2))
                end
            end
        end
    end
    return lca_id_map
end

function build_lca_sites_map(net::AbstractNetwork, tpo::TPO_GPU)
    lca_sites_map = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, LCA}}()
    # Iterate over all pairs of sites in the TPO
    for idd in 1:tpo.terms[end].id
        op = get_id_terms(tpo, idd)
        if length(op) == 2
            site1, site2 = op[1].site, op[2].site
            ## Ensure site1 < site2 for consistent ordering, already done
            # site_pair = site1 < site2 ? (site1, site2) : (site2, site1)
            for l in 1:number_of_layers(net)
                for n in 1:number_of_tensors(net, l)
                    root = (l, n)
                    # Find the lowest common ancestor for the pair of sites
                    lca, link1, link2 = lowest_common_ancestor_node_links(net, site1, site2, root)

                    # Store the LCA information in a dictionary
                    get!(lca_sites_map, (site1[2], site2[2]), Dict())[root] = LCA(lca, (link1, link2))
                end
            end
        end
    end
    return lca_sites_map
end


#=

function build_lca_id_map_old(net::AbstractNetwork, tpo::TPO_GPU)
    lca_id_map = Dict{Int, Dict{Tuple{Int,Int}, LCA}}()
    # Iterate over all pairs of sites in the TPO
    for op in tpo
        if length(op.site) == 2
            site1, site2 = op.site
            ## Ensure site1 < site2 for consistent ordering, already done
            # site_pair = site1 < site2 ? (site1, site2) : (site2, site1)
            for l in 1:number_of_layers(net)
                for n in 1:number_of_tensors(net, l)
                    root = (l, n)
                    # Find the lowest common ancestor for the pair of sites
                    lca, link1, link2 = lowest_common_ancestor_node_links(net, site1, site2, root)

                    # Store the LCA information in a dictionary
                    get!(lca_id_map, op.id, Dict())[root] = LCA(lca, (link1, link2))
                end
            end
        end
    end
    return lca_id_map
end

function build_lca_sites_map_old(net::AbstractNetwork, tpo::TPO_GPU)
    lca_sites_map = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, LCA}}()
    # Iterate over all pairs of sites in the TPO
    for op in tpo
        if length(op.site) == 2
            site1, site2 = op.site
            ## Ensure site1 < site2 for consistent ordering, already done
            # site_pair = site1 < site2 ? (site1, site2) : (site2, site1)
            for l in 1:number_of_layers(net)
                for n in 1:number_of_tensors(net, l)
                    root = (l, n)
                    # Find the lowest common ancestor for the pair of sites
                    lca, link1, link2 = lowest_common_ancestor_node_links(net, site1, site2, root)

                    # Store the LCA information in a dictionary
                    get!(lca_sites_map, (site1[2], site2[2]), Dict())[root] = LCA(lca, (link1, link2))
                end
            end
        end
    end
    return lca_sites_map
end

=#

#=

function build_lca_id_map(net::AbstractNetwork, tpo::TPO_GPU)
    lca_id_map = Dict{Int, Dict{Tuple{Int,Int}, LCA}}()

    # Get sites map
    lca_sites_map = build_lca_sites_map(net, tpo)
    for op in tpo
        # op.site is the ((0,site1), (0,site2)) tuple
        # lin_sites = (op.site[1][2], op.site[2][2])
        if length(op) == 2
            lca_id_map[op.id] = lca_sites_map[op.site[1][2], op.site[2][2]]
        end
    end
    return lca_id_map
end

=#

#=

## Get list of all paired sites in the TPO
"""
    paired_sites(tpo::TPO_GPU)
Returns a Set of tuples representing pairs of sites in the TPO that are paired together.
"""
function paired_sites(tpo::TPO_GPU)
    pairs = Set{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}()
    for terms in tpo
        if length(terms.site) == 2
            site1, site2 = terms.site
            pair = site1 < site2 ? (site1, site2) : (site2, site1)
            push!(pairs, pair)
        end
    end
    return collect(pairs)
end

=#

#=
## Obsolete: Use LCA() constructor instead?

function lca_info(net::AbstractNetwork, lca_sites::Vector{Tuple{Int64, Int64}}, root::Tuple{Int,Int})
    lca_sites = paired_sites(tpo)
    lca_info = LCA[]
    # Iterate over all pairs of sites in the TPO
    for (s1, s2) in LCA_sites
        # Get the sites in the physical lattice
        # print("Processing sites: $s1 and $s2\n")
        site1 = (0, s1)
        site2 = (0, s2)

        # Find the lowest common ancestor for the pair of sites
        lca, link1, link2 = lowest_common_ancestor_node_links(net, site1, site2, root)

        # Store the LCA information in a dictionary or similar structure
        push!(lca_info,LCA(lca, (link1, link2)))
    end
    return lca_info
end

=#