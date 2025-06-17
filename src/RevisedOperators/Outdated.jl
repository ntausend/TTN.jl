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


#=
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
=#

#=
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
=#
