
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

function contract_ops(net::TTN.AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, pos::Tuple{Int,Int}; open_link = pos)

    if layer == 1
        contract_ops_on_node(net, ttn0, link_ops, pos; open_link = open_link)
    else
        contract_ops_on_node(net, ttn0, link_ops, pos; open_link = open_link)
    end
end


function contract_linkops_on_node(net::TTN.AbstractNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
     link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, pos::Tuple{Int,Int}; open_link::Tuple{Int,Int} = pos)

    op_vec = Vector{OpGroup}()
    collaps_list = Vector{ITensor}()

    bucket = get_id_terms(net, link_ops, pos)

    tn = ttn0[pos]    
    # Construct the link tag of the open link
    open_tag = link_tag(open_link...)

    # Find the index in tn corresponding to the open link
    open_index = findfirst(i -> hastags(i, open_tag), inds(tn))
    if open_index === nothing
        error("Could not find open link index with tag $open_tag in node $pos")
    end
    
    ## general: prime index in oc direction
    tn_dag = dag(prime(tn, ind(tn, open_index)))

    for (idd, ops) in bucket
        tn_p = tn_dag
        tensor_list = [tn]
        len_op = ops[1].length
        len_con = length(ops)
         for op in ops
            ## prime index of acting op
            common_index = commonind(tn, op.op)
            tn_p = prime(tn_p, common_index)
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
        # assign a fresh unique id
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

"""
    LCA

Stores the rerooted lowest common ancestor for a pair of sites under a given OC,
along with the legs of the LCA node that each site connects through.

Fields:
- `lca_node`  : Tuple{Int,Int} — (layer, node) of the LCA
- `legs`      : Tuple{Int,Int} — each in {0, 1, 2} = (parent, child1, child2)
"""
struct LCA
  lca_node::Tuple{Int,Int}
  legs::Tuple{Int,Int}
end

## Grouping by LCA

## Collect acting operators on leg 1 of node (1,1) i.e. (0,1) in the TPO
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


