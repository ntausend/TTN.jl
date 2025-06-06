## Get list of all paired sites in the TPO
"""
    paired_sites(tpo::TPO_group)
Returns a Set of tuples representing pairs of sites in the TPO that are paired together.
"""
function paired_sites(tpo::TPO_group)
    pairs = Set{Tuple{Int,Int}}()
    for term in tpo.terms
        if length(term.sites) == 2
            push!(pairs, (term.sites[1], term.sites[2]))
        end
    end
    return collect(pairs)
end

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

function build_lca_map(net::AbstractNetwork, tpo::TPO_group)
    lca_sites = paired_sites(tpo)
    lca_map = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, LCA}}()
    # Iterate over all pairs of sites in the TPO
    for l in 1:number_of_layers(net)
        for n in 1:number_of_tensors(net, l)
            root = (l, n)

            for (s1, s2) in lca_sites
                # Get the sites in the physical lattice
                site1 = (0, s1)
                site2 = (0, s2)

                # Find the lowest common ancestor for the pair of sites
                lca, link1, link2 = lowest_common_ancestor_node_links(net, site1, site2, root)

                # Store the LCA information in a dictionary
                get!(lca_map, (s1, s2), Dict())[root] = LCA(lca, (link1, link2))
            end
        end
    end
    return lca_map
end

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