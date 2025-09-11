
# GPU-aware expect with keyword use_gpu

"""
    expect(ttn::TreeTensorNetwork, op; use_gpu=false)

Return a matrix/array of ⟨op⟩ at every lattice position.
If `use_gpu=true`, the local contraction is performed on the GPU.
"""
function expect_gpu(ttn::TreeTensorNetwork, op; use_gpu::Bool=false)
    physlat = TTN.physical_lattice(TTN.network(ttn))
    vals = map(eachindex(physlat)) do pos
        expect_gpu(ttn, op, pos; use_gpu=use_gpu)
    end
    return reshape(vals, size(physlat))
end

"""
    expect(ttn::TreeTensorNetwork, op, pos::NTuple; use_gpu=false)

Return ⟨op⟩ at coordinate `pos` (d-dimensional tuple).
"""
function expect_gpu(ttn::TreeTensorNetwork, op, pos::NTuple; use_gpu::Bool=false)
    lin = TTN.linear_ind(TTN.physical_lattice(TTN.network(ttn)), pos)
    return expect_gpu(ttn, op, lin; use_gpu=use_gpu)
end

"""
    expect(ttn::TreeTensorNetwork, op, pos::Int; use_gpu=false)

Return ⟨op⟩ at linear position `pos`.
This moves the orthogonality center to the parent of that site, then contracts.
If `use_gpu=true`, only the local tensors/operator are moved to the GPU.
"""
function expect_gpu(ttn::TreeTensorNetwork, _op, pos::Int; use_gpu::Bool=false)
    net  = network(ttn)
    ttnc = copy(ttn) 
    idx = siteinds(net)[pos]
    O = use_gpu ? gpu(op(_op, idx)) : op(_op, idx)

    # parent of node (0, pos)
    parent_pos = parent_node(net, (0, pos))

    # move orthogonality center to parent
    use_gpu ? TTN.move_ortho!(ttnc, parent_pos, Dict()) : TTN.move_ortho!(ttnc, parent_pos)

    # parent tensor
    T = use_gpu ? gpu(ttnc[parent_pos]) : ttnc[parent_pos]

    # perform the contraction
    res = dot(T, noprime(O*T))
    return res
end

## _ops = Vector containing multiple operators to calculate expectation values from

# function expect_gpu(ttn::TreeTensorNetwork, _ops::AbstractVector, pos::Int; use_gpu::Bool=false)
#     net  = network(ttn)
#     ttnc = copy(ttn) 
#     idx = siteinds(net)[pos]
#     Os = use_gpu ? gpu.(op.(_op, idx)) : op.(_ops, idx)

#     # parent of node (0, pos)
#     parent_pos = parent_node(net, (0, pos))

#     # move orthogonality center to parent
#     use_gpu ? TTN.move_ortho!(ttnc, parent_pos, Dict()) : TTN.move_ortho!(ttnc, parent_pos)

#     # parent tensor
#     T = use_gpu ? gpu(ttnc[parent_pos]) : ttnc[parent_pos]

#     # perform the contraction per op in _ops
#     res = dot.(T, noprime.(Os.*T))
#     return res
# end


###################################################

# ------------------------------------------------------------
# 2) One-pass optimized version:
#    Traverses once and evaluates ALL ops at each site.
#    Much faster for ops like ["X","Z"] on large lattices.
# ------------------------------------------------------------

"""
    expect_multi_grid(ttn::TreeTensorNetwork, ops::AbstractVector; use_gpu=false)

Return a Dict `op => array` with ⟨op⟩ at every lattice position, computed
in a single orthocenter traversal (avoids O(L^2) move_ortho!).
`ops` are passed to `ITensors.op(op, idx)` (e.g. "X", "Z").

Example:
    grids = expect_gpu_multi(ttn, ["X","Z"]; use_gpu=false)
    X = grids["X"]; Z = grids["Z"]
"""
function expect_gpu_multi(ttn::TreeTensorNetwork, ops::AbstractVector; use_gpu::Bool=false)
    net = TTN.network(ttn)
    lat = TTN.physical_lattice(net)
    dims = size(lat)
    N = length(lat)

    # Pre-build result storage
    res_vecs = Dict{Any, Vector{Float64}}(op => Vector{Float64}(undef, N) for op in ops)

    # Work on one copy and move the orthocenter incrementally
    ttnc = copy(ttn)

    # Full minimal steps walk through whole tree that includes leaves (layer 0)
    ## Reduce to iteration through only physical_lattice with respect to order (just eachindex(lat) isnt optimal)
    route = ttn_traversal_least_steps(net; include_layer0=true, exclude_topnode=true).visit_order

    for node in route
        l, i = node
        l == 0 || continue             # only physical leaves
        lin = i                        # linear index 1..N
        # lin = linear_ind(lat, node)
        # println(lin)
        parent = parent_node(net, node)
        use_gpu ? move_ortho!(ttnc, parent, Dict()) : move_ortho!(ttnc, parent)

        idx = ITensors.siteinds(net)[lin]

        # Parent tensor once
        T = use_gpu ? gpu(ttnc[parent]) : ttnc[parent]

        # Evaluate all requested ops at this site

        for op_spec in ops
            O = use_gpu ? gpu(op(op_spec, idx)) : op(op_spec, idx)
            val = ITensors.scalar(ITensors.dag(T) * ITensors.noprime(O * T))
            res_vecs[op_spec][lin] = real(val)
        end
    end

    # Reshape vectors into grids
    return Dict(op => reshape(vec, dims) for (op, vec) in res_vecs)
end

function entanglement_entropy_plus_ops(ttn::TreeTensorNetwork, ops::AbstractVector; use_gpu = false, start=(number_of_layers(network(ttn))-1, 1), include_layer0 = true)
    net = network(ttn)
    lat = TTN.physical_lattice(net)
    dims = size(lat)
    N = length(lat)
    ttnc = copy(ttn)

    maxL = number_of_layers(net)
    entropies = Dict{Tuple{Int, Int}, Float64}()
    exp_ops = Dict{Any, Vector{Float64}}(op => Vector{Float64}(undef, N) for op in ops)

    route = ttn_traversal_least_steps(net; start, include_layer0, exclude_topnode=true);

    for pos_left in route.visit_order
        # first calculate entropies along cut above pos_left
        pos_left[1] == maxL && continue
        pos_right = parent_node(net, pos_left) 

        # checking if pos_right is contained in the network
        check_valid_position(net, pos_right)
        # check if layer number is not physical
        @assert pos_right[1] > 0
        # check if pos_left is either in the child nodes/ or being the parent node
        @assert (pos_left ∈ child_nodes(net, pos_right) || pos_left == parent_node(net, pos_left))

        # now move the orthogonality centrum
        ttnc = use_gpu ? move_ortho!(ttnc, pos_right, Dict()) : move_ortho!(ttnc, pos_right)
        T = use_gpu ? gpu(ttnc[pos_right]) : ttnc[pos_right]
        
        # getting the indices for decomposition, this only contains pos_left link
        if pos_left[1] == 0
            # pos_left is a physical site.. we need to filter differently
            idx_left = inds(T; tags = "Site,n=$(pos_left[2])")
        else
            idx_left = inds(T; tags = "Link,nl=$(pos_left[1]),np=$(pos_left[2])")
        end

        U,S,V,spec = svd(T, idx_left)
        entropies[pos_left] = entropy2(spec)
        
        ######################################

        # then calculate expectation values of all operators in ops at physical layer
        l, i = pos_left
        l == 0 || continue             # only physical leaves
        lin = i                        # linear index 1..N

        idx = ITensors.siteinds(net)[lin]

        # Evaluate all requested ops at this site

        for op_spec in ops
            O = use_gpu ? gpu(op(op_spec, idx)) : op(op_spec, idx)
            val = ITensors.scalar(ITensors.dag(T) * ITensors.noprime(O * T))
            exp_ops[op_spec][lin] = real(val)
        end
    end
    return entropies, Dict(op => reshape(vec, dims) for (op, vec) in exp_ops)
end