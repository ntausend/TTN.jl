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
            # get!(link_ops, link, ops_on_site)
            link_ops[link] = ops_on_site
        end
    end 
    return link_ops
end

function contract_first_layer_linkops(net::BinaryNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor})
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
    (layer, node) = (4,1)
    
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

function contract_ops_on_node(net::AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, (layer, node)::Tuple{Int,Int})
    op_vec = Vector{OpGroup}()
    collaps_list = Vector{ITensor}()
    id_collaps = Int[]
    bucket = get_id_terms(net, link_ops, (layer, node))
    tn = ttn0[(layer, node)]
    tn_dag = dag(prime(tn))

    for (idd, vec) in bucket
        if length(vec) == 2 # && is_lca(vec[1],node,lca_id_map)
            act_op1, act_op2 = vec[1].op, vec[2].op
            len = vec[1].length # == vec[2].length
            tensor_list = [tn, tn_dag, act_op1, act_op2]
            opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
            tn_con = contract(tensor_list; sequence = opcon_seq)
            len -= 1
            id_collaps = idd
            # push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
        else
        # one-site or still two separate legs
            act_op = vec[1].op
            len = vec[1].length
            i = which_child(net, vec[1].site) # 0, 1 or 2
            tn_dag_p = noprime(tn_dag, ind(tn_dag, 3-i)) # unprime leg without op
            tensor_list = [tn, tn_dag_p, act_op]
            opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
            tn_con = contract(tensor_list; sequence = opcon_seq)
            # push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
        end
        if len > 1
            push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
        else
            push!(collaps_list, tn_con)
        end
    end
    # Collapse all tensors with length 1
    if length(collaps_list) > 0
        tn_collaps = sum(collaps_list)
        # assign a fresh unique id
        uid = new_opgroup_id()
        push!(op_vec, OpGroup(uid, (layer, node), tn_collaps, 1))
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
        end
    end
    if isnothing(nl) || isnothing(np)
        error("Could not extract nl or np from index tags")
    end
    return nl, np
end

function contract_linkops_on_node(net::AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, (layer, node)::Tuple{Int,Int})
    op_vec = Vector{OpGroup}()
    collaps_list = Vector{ITensor}()
    bucket = get_id_terms(net, link_ops, (layer, node))
    tn = ttn0[(layer, node)]
    ## get sites of ITensor
    tn_sites = extract_layer_node.(inds(tn))
    ## find index of open / parent connection to prime
    id_open = findfirst(x -> x == (layer, node), tn_sites)
    tn_dag = dag(prime(tn, ind(tn, id_open)))

    for (idd, op) in bucket
        if length(op) == 2
            len = op[1].length
            op_site_id1 = findfirst(x -> x == op[1].site, tn_sites)
            op_site_ind1 = ind(tn_dag, op_site_id1)
            tn_dag_p1 = prime(tn_dag, op_site_ind1)

            op_site_id2 = findfirst(x -> x == op[2].site, tn_sites)
            op_site_ind2 = ind(tn_dag, op_site_id2)
            tn_dag_p2 = prime(tn_dag_p1, op_site_ind2)

            tensor_list = [tn, tn_dag_p2, op[1].op, op[2].op]
            opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
            tn_con = contract(tensor_list; sequence = opcon_seq)
            len -= 1
            # push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
        else
        # one-site or still two separate legs
            act_op = op[1].op
            len = op[1].length
            op_site_id = findfirst(x -> x == op[1].site, tn_sites)
            op_site_ind = ind(tn_dag, op_site_id)
            tn_dag_p = prime(tn_dag, op_site_ind)
            tensor_list = [tn, tn_dag_p, act_op]
            opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
            tn_con = contract(tensor_list; sequence = opcon_seq)
            # push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
        end
        if len > 1
            push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
        else
            push!(collaps_list, tn_con)
        end
    end
    # Collapse all tensors with length 1
    if length(collaps_list) > 0
        tn_collaps = sum(collaps_list)
        # assign a fresh unique id
        uid = new_opgroup_id()
        push!(op_vec, OpGroup(uid, (layer, node), tn_collaps, 1))
    end
    return op_vec
end


## Grouping by LCA

## Collect acting operators on leg 1 of node (1,1) i.e. (0,1) in the TPO
# pTPO.link_ops[Link(1,1,1)]
function contract_link_ops_by_lca(net::AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, lca_id_map::Dict{Int64, Dict{Tuple{Int64, Int64}, LCA}}, (layer,node)::Tuple{Int,Int})
    # Contract all operators acting on the first leg of the node (1,1)
    # and return a vector of OpGroup objects
    # OpGroup(id, node, tensor)
    # where id is the id of the operator in the TPO
    # node is the node where the operator acts on
    # tensor is the resulting tensor after contraction
    println("Contracting link operators for node ($layer,$node)")
    # Where to store next layer link operators
    link = (parent_node(net,(layer, node)), which_child(net, (layer, node)))
    op_vec = Vector{OpGroup}()
    tn = ttn0[(layer, node)]
    tn_dag = dag(prime(tn))

    visited_ids = Set{Int}()
    # Get all operators acting on the first leg of the node (1,1)
    for (i, child) in enumerate(child_nodes(net, (layer, node)))
        acting_ops = link_ops[(layer, node), i]
        for op in acting_ops
            idd = op.id
            len = op.length
            if idd in visited_ids
                continue
            else
                if haskey(lca_id_map, op.id) == false
                    println("ID: $idd: Contract directly")
                    act_op = op.op
                    tn_dag_p = noprime(tn_dag, ind(tn_dag, 3-i))
                    tensor_list = [tn, tn_dag_p, act_op]
                    opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
                    tn_con = contract(tensor_list; sequence = opcon_seq)
                    push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
                else
                    if (layer, node) == lca_id_map[op.id][pTPO.oc].lca_node
                        println("ID: $idd: Contract via LCA")
                        # act_op1, act_op2 = get_id_terms(tpo, op.id)[1].ops[1], get_id_terms(tpo, op.id)[2].ops[1]
                        act_op1, act_op2 = get_id_terms(net,link_ops, (layer, node), op.id)[1].op, get_id_terms(net,link_ops, (layer, node), op.id)[2].op
                        tensor_list = [tn, tn_dag, act_op1, act_op2]
                        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
                        tn_con = contract(tensor_list; sequence = opcon_seq)
                        push!(op_vec, OpGroup(idd, (layer, node),tn_con, len-1))
                    else
                        println("ID: $idd: Contract directly and keep open link")
                        act_op = op.op
                        tn_dag_p = noprime(tn_dag, ind(tn_dag, 3-i))
                        tensor_list = [tn, tn_dag_p, act_op]
                        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
                        tn_con = contract(tensor_list; sequence = opcon_seq)
                        push!(op_vec, OpGroup(idd, (layer, node),tn_con, len))
                    end
                end
                push!(visited_ids, idd)
            end
        end
    end
    return op_vec
end