

"""
    apply!(ttn::TreeTensorNetwork, op::String, site::Int)

Apply a single-site operator to a specified site in a tree tensor network.

# Arguments
- `ttn::TreeTensorNetwork`: The tree tensor network to modify.
- `op::String`: The name of the operator to apply (e.g., "X", "Y", "Z").
- `site::Int`: The site index where the operator is applied.

# Details
This function performs the following steps:
1. Locates the orthogonality center position for the given site.
2. Moves the orthogonality center to that position.
3. Retrieves the ITensor operator from the site indices.
4. Contracts the operator with the tensor at the orthogonality center.
5. Updates the tensor in the network with the contracted result.
6. Reorthogonalizes the tree tensor network.

# Returns
- `ttn::TreeTensorNetwork`: The modified tree tensor network.
"""
function apply!(ttn::TreeTensorNetwork, op::String, site::Int)
    oc_position = TTN.parent_node(TTN.network(ttn), (0,site))
    move_ortho!(ttn, oc_position)
    op = ITensorMPS.op(op, TTN.siteinds(ttn)[site])
    tensor_new = noprime(contract(ttn[oc_position], op))
    ttn[oc_position] = tensor_new
    TTN._reorthogonalize!(ttn)
    return ttn
end

apply!(ttn::TreeTensorNetwork, op::String, site::Tuple{Int,Int}) = apply!(ttn, op, TTN.linear_ind(TTN.physical_lattice(TTN.network(ttn)), site))

apply(ttn::TreeTensorNetwork, op::String, site::Int) = begin
    ttn_new = copy(ttn)
    apply!(ttn_new, op, site)
end

apply(ttn::TreeTensorNetwork, op::String, site::Tuple{Int,Int}) = apply(ttn, op, TTN.linear_ind(TTN.physical_lattice(TTN.network(ttn)), site))