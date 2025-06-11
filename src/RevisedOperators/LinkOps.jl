function populate_physical_link_ops(net, tpo)
    link_ops = Dict{Tuple{Int,Int,Int}, Vector{OpGroup}}()
    # Iterate over each node of first layer
    for n in eachindex(net,1)
        # Get child nodes of the first layer nodes
        childs = TTN.child_nodes(net, (1, n))
        # Iterate over each child node
        for (i, child) in enumerate(childs)
            link = (1,n,i)  # layer, node, child_index
            ops_on_site = get_site_terms(tpo, child)
            # Store the operators on the link in the dictionary
            # get!(link_ops, link, ops_on_site)
            link_ops[link] = ops_on_site
        end
    end 
    return link_ops
end