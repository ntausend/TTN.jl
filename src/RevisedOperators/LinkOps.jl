function populate_physical_link_ops(net::AbstractNetwork, tpo::TPO_group)
    link_ops = Dict{Tuple{Tuple{Int,Int},Int}, Vector{OpGroup}}()
    # Iterate over each node of first layer
    for n in eachindex(net,1)
        # Get child nodes of the first layer nodes
        childs = TTN.child_nodes(net, (1, n))
        # Iterate over each child node
        for (i, child) in enumerate(childs)
            link = ((1,n),i)  # layer, node, child_index
            ops_on_site = get_site_terms(tpo, child)
            # Store the operators on the link in the dictionary
            link_ops[link] = ops_on_site
        end
    end 
    return link_ops
end

function contract_first_layer_linkops(net::BinaryNetwork, tpo::TPO_group, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor})
    link_ops = populate_physical_link_ops(net, tpo);
    layer = 1
    println("layer: $layer")
    for node in eachindex(net,layer)
        println("Node: $node")
        # define where to save linkops
        link = (parent_node(net,(layer,node)),which_child(net,(layer,node)))
        coarse_ops = contract_ops_on_node(net, ttn0, link_ops, (layer,node))
        link_ops[link] = coarse_ops
    end
    return link_ops
end

function contract_upper_layer_linkops(net::BinaryNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor}, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}})
    for l in 2:number_of_layers(net)-1
        println("layer: $l")
        for n in eachindex(net,l)
            println("Node: $n")
            # define where to save linkops
            link = (parent_node(net,(l,n)),which_child(net,(l,n)))
            coarse_ops = contract_linkops_on_node(net, ttn0, link_ops, (l,n))
            link_ops[link] = coarse_ops
        end
    end
    return link_ops
end

function complete_contraction(net::BinaryNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor}, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}})
    (layer, node) = (number_of_layers(net), 1)
    
    tn = ttn0[(layer, node)]
    collaps_list = Vector{ITensor}()
    bucket = get_id_terms(net, link_ops, (layer,node))
    ## get sites of ITensor
    tn_sites = extract_layer_node.(inds(tn))
    ## find index of open / parent connection to prime

    for (idd, ops) in bucket
        tn_p = tn
        tensor_list = [tn]
        for op in ops            
            op_site_id = findfirst(x -> x == op.site, tn_sites)
            op_site_ind = ind(tn_p, op_site_id)
            tn_p = prime(tn_p, op_site_ind)
            push!(tensor_list, op.op)
        end
        push!(tensor_list, dag(tn_p))
        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
        tn_con = contract(tensor_list; sequence = opcon_seq)
        push!(collaps_list, tn_con)
    end
    tn_exp = sum(collaps_list)
    return tn_exp
end

function complete_contraction_rerooted(net::BinaryNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor}, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, root::Tuple{Int64, Int64})
    
    tn = ttn0[root]
    collaps_list = Vector{ITensor}()
    bucket = get_id_terms(net, link_ops, root)
    ## get sites of ITensor
    tn_sites = extract_layer_node.(inds(tn))
    ## find index of open / parent connection to prime

    for (_idd, ops) in bucket
        tn_p = tn
        tensor_list = [tn]
        for op in ops            
            op_site_id = findfirst(x -> x == op.site, tn_sites)
            op_site_ind = ind(tn_p, op_site_id)
            tn_p = prime(tn_p, op_site_ind)
            push!(tensor_list, op.op)
        end
        push!(tensor_list, dag(tn_p))
        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
        tn_con = contract(tensor_list; sequence = opcon_seq)
        push!(collaps_list, tn_con)
    end
    tn_exp = sum(collaps_list)
    return tn_exp
end

function upflow(net::BinaryNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor}, tpo::TPO_group)
    link_ops = populate_physical_link_ops(net, tpo)
    # link_ops = contract_first_layer_linkops(net, ttn0, link_ops)
    # link_ops = contract_upper_layer_linkops(net, ttn0, link_ops)
    
    for l in 1:number_of_layers(net)-1
        # println("layer: $l")
        
        for n in eachindex(net,l)
            # println("Node: $n")
            # define where to save linkops
            link = (parent_node(net,(l,n)),which_child(net,(l,n)))
            coarse_ops = contract_ops(net, ttn0, link_ops, (l,n))
            link_ops[link] = coarse_ops
        end
    end

    return link_ops
end

function upflow_rerooted(net::BinaryNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor}, tpo::TPO_group, newroot::Tuple{Int,Int})

    link_ops = populate_physical_link_ops(net, tpo)
    # get path from top node to oc = newroot
    top_node = (number_of_layers(net), 1)
    connect_path = pushfirst!(connecting_path(net, top_node, newroot), top_node)

    for pos in reverse_bfs_nodes(net, newroot)
        layer, node = pos
        # stop at the new root itself
        if pos == newroot
            return link_ops
        end
        # println("Layer: $layer, Node: $node")
        # determine the link on which to store your contracted ops
        if pos in connect_path
            next_node = next_on_path(connect_path, pos)
            link = (pos, which_child(net, next_node))
            # println("Layer: $layer, Node: $node, Link re: $link")
            if pos == top_node
                coarse_ops = contract_linkops_on_topnode(net, ttn0, link_ops, top_node, next_node)
            else
                coarse_ops = contract_ops(net, ttn0, link_ops, pos; open_link = next_node)
            end
                
        else
            link = (parent_node(net, pos), which_child(net, pos))
            # println("Layer: $layer, Node: $node, Link dev: $link")
            coarse_ops = contract_ops(net, ttn0, link_ops, pos)
        end
        # contract the ops on this node:
        
        link_ops[link] = coarse_ops
    end
    return link_ops
end

function contract_ops(net::TTN.AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, pos::Tuple{Int,Int}; open_link = pos)
    (layer, node) = pos
    if layer == 1
        contract_ops_on_node(net, ttn0, link_ops, pos)
    else
        contract_linkops_on_node(net, ttn0, link_ops, pos; open_link = open_link)
    end
end

function contract_ops_on_node(net::TTN.AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, pos::Tuple{Int,Int})
    (layer, node) = pos
    op_vec = Vector{OpGroup}()
    collaps_list = Vector{ITensor}()
    
    bucket = get_id_terms(net, link_ops, pos)
    tn = ttn0[pos]
    tn_dag = dag(prime(tn, ind(tn,3)))

    for (idd, ops) in bucket
        tn_p = tn_dag
        tensor_list = [tn]
        len_op = ops[1].length
        len_con = length(ops)
        for op in ops
            ## prime index of acting op
            i = which_child(net, op.site)
            indi = ind(tn_p, i)
            tn_p = prime(tn_p, indi)
            push!(tensor_list, op.op)
        end
        push!(tensor_list, tn_p)
        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
        tn_con = contract(tensor_list; sequence = opcon_seq)
        if len_con > 1
            len_op -= (len_con - 1)
        end
        if len_op > 1
            push!(op_vec, OpGroup(idd, (layer, node),tn_con, len_op))
        else
            push!(collaps_list, tn_con)
        end
    end
    # Collapse all tensors with length 1
    if length(collaps_list) > 0
        # assign a fresh unique id
        push!(op_vec, OpGroup(new_opgroup_id(), (layer, node), sum(collaps_list), 1))
    end
    return op_vec
end

function contract_linkops_on_node(net::TTN.AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, pos::Tuple{Int,Int}; open_link = pos)

    # (layer, node) = pos
    op_vec = Vector{OpGroup}()
    collaps_list = Vector{ITensor}()

    bucket = get_id_terms(net, link_ops, pos)

    tn = ttn0[pos]
    
    ## get sites of ITensor
    tn_sites = extract_layer_node.(inds(tn))
    ## find index of open / parent connection and prime
    ## only valid for upflows
    id_open = findfirst(x -> x == open_link, tn_sites)
    ## general: prime index in oc direction
    tn_dag = dag(prime(tn, ind(tn, id_open)))

    for (idd, ops) in bucket
        tn_p = tn_dag
        tensor_list = [tn]
        len_op = ops[1].length
        len_con = length(ops)
         for op in ops
            ## prime index of acting op
            op_site_id = findfirst(x -> x == op.site, tn_sites)
            op_site_ind = ind(tn_p, op_site_id)
            tn_p = prime(tn_p, op_site_ind)
            push!(tensor_list, op.op)
        end
        push!(tensor_list, tn_p)
        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
        tn_con = contract(tensor_list; sequence = opcon_seq)
        if len_con > 1
            len_op -= (len_con - 1)
        end
        if len_op > 1
            push!(op_vec, OpGroup(idd, open_link, tn_con, len_op))
        else
            push!(collaps_list, tn_con)
        end
    end
    # Collapse all tensors with length 1
    if length(collaps_list) > 0
        push!(op_vec, OpGroup(new_opgroup_id(), open_link, sum(collaps_list), 1))
    end
    return op_vec
end

function contract_linkops_on_topnode(net::TTN.AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, pos::Tuple{Int,Int}, open_link::Tuple{Int,Int})
    
    op_vec = Vector{OpGroup}()
    collaps_list = Vector{ITensor}()
    # pos is supposed to be the top node
    # pos = (number_of_layers(net), 1)

    bucket = get_id_terms(net, link_ops, pos)
    tn = ttn0[pos]
    ## get sites of ITensor
    tn_sites = extract_layer_node.(inds(tn))
    tn_dag = dag(prime(tn))
    # all ops have length one because they are on the top node
    for (idd, ops) in bucket
        tensor_list = [tn, tn_dag, ops[1].op]
        len_op = ops[1].length
        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
        tn_con = contract(tensor_list; sequence = opcon_seq)
        if len_op > 1
            push!(op_vec, OpGroup(idd, open_link, tn_con, len_op))
        else
            push!(collaps_list, tn_con)
        end
    end
    # Collapse all tensors with length 1
    if length(collaps_list) > 0
        push!(op_vec, OpGroup(new_opgroup_id(), open_link, sum(collaps_list), 1))
    end
    return op_vec
end

function extract_layer_node(index::Index)
    taglist = string.(collect(tags(index)))  # Vector{String}
    nl = nothing
    np = nothing
    for tag in taglist
        if startswith(tag, "nl=")
            nl = parse(Int, split(tag, "=")[2])
        elseif startswith(tag, "np=")
            np = parse(Int, split(tag, "=")[2])
        elseif startswith(tag, "n=")
            nl = 0
            np = parse(Int, split(tag, "=")[2])
        end
    end
    if isnothing(nl) || isnothing(np)
        error("Could not extract nl or np from index tags")
    end
    return nl, np
end