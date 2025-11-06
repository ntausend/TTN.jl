"""
```julia
    correlations(ttn::TreeTensorNetwork, op1, op2, pos::NTuple)
```

Calculates the correlation function between the `op1` at position `pos` and `op2` varying for all sites of the system.  
The position is given as the d-dimensional coordinate of the system.
"""
function correlations_gpu(ttn::TreeTensorNetwork, op1, op2, pos::NTuple; use_gpu::Bool=false)
    pos_lin = linear_ind(physical_lattice(network(ttn)), pos)
    return correlations_gpu(ttn, op1, op2, pos_lin; use_gpu=use_gpu)
end

"""
```julia
    correlations(ttn::TreeTensorNetwork, op1, op2, pos::Int)
```

Calculates the correlation function between the `op1` at position `pos` and `op2` varying for all sites of the system.  
The position is given as the linearized position in the system.
"""
function correlations_gpu(ttn::TreeTensorNetwork, op1, op2, pos::Int; use_gpu::Bool=false)
    physlat = physical_lattice(network(ttn))
    res = map(eachindex(physlat)) do pp
        correlation_gpu(ttn, op1, op2, pos, pp; use_gpu=use_gpu)
    end
    dims = size(physlat)
    return reshape(res, dims)
end

"""
```julia
    correlation(ttn::TreeTensorNetwork, op1, op2, pos1::NTuple,pos1::NTuple)
```

Calculates the correlation function between the `op1` at position `pos1` and `op2` at position `pos2`.
The positions is given as the d-dimensional coordinate of the system.
"""
function correlation_gpu(ttn::TreeTensorNetwork, op1, op2, pos1::NTuple, pos2::NTuple; use_gpu::Bool=false)
    pos1_lin = linear_ind(physical_lattice(network(ttn)), pos1)
    pos2_lin = linear_ind(physical_lattice(network(ttn)), pos2)
    return correlation_gpu(ttn, op1, op2, pos1_lin, pos2_lin; use_gpu=use_gpu)
end

"""
```julia
    correlation(ttn::TreeTensorNetwork, op1, op2, pos1::Int,pos1::Int)
```
Intnts the correlation function between the `op1` at position `pos1` and `op2` at position `pos2`.
The positions is given as the linearized position in the system.
"""
function correlation_gpu(ttn::TreeTensorNetwork, op1::AbstractString, op2::AbstractString, pos1::Int, pos2::Int; use_gpu::Bool=false)
    if pos1 == pos2
        # fast exit using the expectation value
        op_new = "$op1 * $op2"
        return expect_gpu(ttn, op_new, pos1; use_gpu=use_gpu)
    end
    if pos1 < pos2
        return _correlation_pos1_le_pos2_gpu(ttn, op1, op2, pos1, pos2; use_gpu=use_gpu)
    else
        return conj(_correlation_pos1_le_pos2_gpu(ttn, op2, op1, pos2, pos1; use_gpu=use_gpu))
    end
end

# function for positions not being equal
# otherwise it is the expectation value of the product of operators
function _correlation_pos1_le_pos2_gpu(ttn::TreeTensorNetwork, op1::String, op2::String, pos1::Int, pos2::Int; use_gpu::Bool=false)
    @assert pos1 < pos2

    net = network(ttn)
    phys_sites = sites(ttn)
    # get the operators
    Opl = use_gpu ? gpu(op(op1, phys_sites[pos1])) : op(op1, phys_sites[pos1])
    Opr = use_gpu ? gpu(op(op2, phys_sites[pos2])) : op(op2, phys_sites[pos2])

    pos_parent1 = parent_node(net, (0,pos1))
    pos_parent2 = parent_node(net, (0,pos2))
    # get the minimal path connecting the two 
    path = vcat(pos_parent1, connecting_path(net, pos_parent1, pos_parent2))

    # finding the top node for the subtree
    _, topidx = findmax(first, path)
    top_pos = path[topidx]
    # split path into "left" and "right" part of the path
    # in general networks, they might not be equal
    
    path_l = path[1:topidx-1]
    path_r = path[end:-1:topidx+1]

    # move the orthocenter to the top node for having all other subtrees collapse
    ttnc = use_gpu ? move_ortho!(copy(ttn), top_pos, Dict()) : move_ortho!(copy(ttn), top_pos)
    # now calculate the flow of both operators to the end of the path
    for posl in path_l
        T = use_gpu ? gpu(ttnc[posl]) : ttnc[posl]
        # getting the index shared by the tensor and
        # the current left_rg operator
        idx_shr = commonind(T, Opl)
        # now getting the link to the parent node. This link is always
        # labeled by the current layer number
        # no need for optimal_contraction_sequence here.. Opl and Opr only
        # operates on one leg
        idx_prnt = only(inds(T; tags = "nl=$(posl[1])"))
        Opl = (T * Opl) * dag(prime(T, idx_prnt, idx_shr))
    end
    for posr in path_r
        T = use_gpu ? gpu(ttnc[posr]) : ttnc[posr]
        # getting the index shared by the tensor and
        # the current left_rg operator
        idx_shr = commonind(T, Opr)
        # now getting the link to the parent node. This link is always
        # labeled by the current layer number
        # no need for optimal_contraction_sequence here.. Opl and Opr only
        # operates on one leg
        idx_prnt = only(inds(T; tags = "nl=$(posr[1])"))
        Opr = (T * Opr) * dag(prime(T, idx_prnt, idx_shr)) 
    end

    T = use_gpu ? gpu(ttnc[top_pos]) : ttnc[top_pos]

    idx_shrl = commonind(T, Opl)
    idx_shrr = commonind(T, Opr)
    # no need for optimal_contraction_sequence here.. Opl and Opr only
    # operates on one leg
    return ITensors.scalar(((T*Opl) * Opr)*dag(prime(T, idx_shrl, idx_shrr)))
end


### general n point correlations ###
"""
```julia
    correlation(ttn::TreeTensorNetwork, ops::Vector{String}, pos::Vector{NTuple})
```

Calculates the n-point function of n operators `ops` at positions `pos`.
The position `pos[j]` corresponds to the operator `ops[j]`. The positions are given by the d-dimensional coordinates of the system.
"""
function correlation_gpu(ttn::TreeTensorNetwork, ops::Vector{String}, pos::Vector{NTuple}; use_gpu::Bool=false)
    pos_lin = [linear_ind(physical_lattice(network(ttn)), posi) for posi in pos]
    return correlation_gpu(ttn, ops, pos_lin; use_gpu=use_gpu)
end

"""
```julia
    correlation(ttn::TreeTensorNetwork, ops::Vector{String}, pos::Vector{Int})
```

Calculates the n-point function of n operators `ops` at positions `pos`.
The position `pos[j]` corresponds to the operator `ops[j]`. The positions are given by the linearized coordinates of the system.
"""
function correlation_gpu(ttn::TreeTensorNetwork, ops::Vector{String}, pos::Vector{Int}; use_gpu::Bool=false)
    net = network(ttn)
    phys_sites = sites(ttn)

    # find the top position
    pos_parent = [parent_node(net, (0,posi)) for posi in pos]
    paths = [vcat(pos_parent1, connecting_path(net, pos_parent1, pos_parent2)) for (pos_parent1, pos_parent2) in Iterators.product(pos_parent, pos_parent) if pos_parent1 != pos_parent2]

    # finding the top node for the subtree
    top_path_pos_idx = [findmax(first, path)[2] for path in paths]
    top_path_pos = [path[top_path_pos_i] for (path,top_path_pos_i) in zip(paths, top_path_pos_idx)]
    _,top_pos_idx = findmax(first, top_path_pos)
    top_pos = top_path_pos[top_pos_idx]

    ttnc = use_gpu ? move_ortho!(copy(ttn), top_pos, Dict()) : move_ortho!(copy(ttn), top_pos)

    ops_pos = [(use_gpu ? gpu(op(opsi, phys_sites[posi])) : op(opsi, phys_sites[posi]), (0,posi)) for (opsi,posi) in zip(ops,pos)]

    for ll in 1:top_pos[1]-1
      temp_ops_pos = []

      for pp in eachindex(net,ll)
        idx = findall(x -> parent_node(net, x[2]) == (ll,pp), ops_pos)
        isempty(idx) && continue

        T = ttnc[(ll,pp)]
        temp_ops = [ops_pos[i][1] for i in idx]

        idx_shr = [commonind(T, temp_opsi) for temp_opsi in temp_ops]
        idx_prnt = only(inds(T; tags = "nl=$(ll)"))
        append!(temp_ops_pos, [(reduce(*, temp_ops, init = T) * dag(prime(T, idx_prnt, idx_shr...)), (ll,pp))])
      end

      ops_pos = temp_ops_pos

    end

    T = ttnc[top_pos]
    idx = findall(x -> parent_node(net, x[2]) == top_pos, ops_pos)
    temp_ops = [ops_pos[i][1] for i in idx]
    idx_shr = [commonind(T, temp_opsi) for temp_opsi in temp_ops]

    return ITensors.scalar(reduce(*, temp_ops, init = T) * dag(prime(T, idx_shr...)))
end;

"""
```julia
    correlation_matrix(ttn::TreeTensorNetwork, ops::Vector{String}, pos::Vector{Int})
```
Calculates the correlation matrix between `op1`  and `op2` for all sites.
The positions within the matrix are given as the linearized position in the system.
"""
function all_correlations_gpu(ttn::TreeTensorNetwork, op1::String, op2::String; use_gpu::Bool=false)
    net = network(ttn)
    physlat = physical_lattice(net)
    phys_sites = TTN.sites(ttn)
    nsites = length(physlat)

    all_paths = map(eachindex(physlat)) do i
      res = map(eachindex(physlat)) do j
        j>=i && return nothing

        pos_parent1 = TTN.parent_node(net, (0,i))
        pos_parent2 = TTN.parent_node(net, (0,j))
        path = vcat(pos_parent1, TTN.connecting_path(net, pos_parent1, pos_parent2))

        _, topidx = findmax(first, path)
        top_pos = path[topidx]

        path_l = path[1:topidx-1]
        path_r = path[end:-1:topidx+1]

        return (pos1=i, pos2=j, top_index=topidx, top_position=top_pos, left_path=path_l, right_path=path_r)
      end

      filter!(x-> !isnothing(x), res)

      return group_by_top_idx(res)
    end[2:end]

    mat = Matrix{ComplexF64}(undef, nsites, nsites)

    for pos_path in all_paths
      layers = sort(collect(keys(pos_path)))

      pos1 = first(pos_path[first(layers)]).pos1

      for l in layers
        top_pos = first(pos_path[l]).top_position
        path_l  = first(pos_path[l]).left_path
        ttnc = use_gpu ? move_ortho!(copy(ttn), top_pos, Dict()) : move_ortho!(copy(ttn), top_pos)

        Opl = use_gpu ? gpu(op(op1, phys_sites[pos1])) : op(op1, phys_sites[pos1])

        # now calculate the flow of both operators to the end of the left path
        for posl in path_l
            T = ttnc[posl]
            idx_shr = commonind(T, Opl)
            idx_prnt = only(inds(T; tags = "nl=$(posl[1])"))
            Opl = (T * Opl) * dag(prime(T, idx_prnt, idx_shr))
        end
        for path in pos_path[l]
          pos2 = path.pos2
          path_r = path.right_path

          Opr = use_gpu ? gpu(op(op2, phys_sites[pos2])) : op(op2, phys_sites[pos2])

          # now calculate the flow of both operators to the end of the right path
          for posr in path_r
              T = ttnc[posr]
              idx_shr = commonind(T, Opr)
              idx_prnt = only(inds(T; tags = "nl=$(posr[1])"))
              Opr = (T * Opr) * dag(prime(T, idx_prnt, idx_shr)) 
          end

          T = ttnc[top_pos]
          idx_shrl = commonind(T, Opl)
          idx_shrr = commonind(T, Opr)

          matEl = ITensors.scalar(((T*Opl) * Opr)*dag(prime(T, idx_shrl, idx_shrr)))
          mat[pos1, pos2] = matEl
          mat[pos2, pos1] = matEl

        end
      end
    end

    for pos in 1:nsites
      op_new = "$op1 * $op2"
      mat[pos,pos] = expect_gpu(ttn, op_new, pos; use_gpu=use_gpu)
    end

    return mat
end
