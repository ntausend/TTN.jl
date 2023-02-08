"""
   entanglement_entropy(ttn::TreeTensorNetwork, pos_right::Tuple{Int, Int}, pos_left::Tuple{Int, Int})

Calculates the entanglement entropy by cutting the edge between `pos_right` and `pos_left`.
This effectively creates a bipartion where region A is given by the subtree of `pos_left` and 
B by the subtree of `pos_right`. The algorithm cuts the leg by moving the orthogonality centrum to `pos_right`
and forming the SVD on that node

Inputs: 
    - ttn defining: the Tree Tensor Network
    - pos_right: position in the network defining the right node of the cut. The tensor at this position will
      be used for decomposition
    - pos_left: position in the network defining the left part of the partionining. 
"""
function entanglement_entropy(ttn::TreeTensorNetwork{N, ITensor}, pos_right::Tuple{Int, Int}, pos_left::Tuple{Int, Int}) where{N}
    net = network(ttn)
    # checking if pos_right is contained in the network
    check_valid_position(net, pos_right)
    # check if layer number is not physical
    @assert pos_right[1] > 0
    # check if pos_left is either in the child nodes/ or being the parent node
    @assert (pos_left ∈ child_nodes(net, pos_right) || pos_left == parent_node(net, pos_left))

    # now move the orthogonality centrum
    ttnc = move_ortho!(copy(ttn), pos_right)
    T = ttnc[pos_right]
    
    # this only works for ITensors...
    # getting the indices for decomposition, this only contains pos_left link
    if pos_left[1] == 0
        # pos_left is a physical site.. we need to filter differently
        idx_left = inds(T; tags = "Site,n=$(pos_left[2])")
    else
        idx_left = inds(T; tags = "Link,nl=$(pos_left[1]),np=$(pos_left[2])")
    end

    U,S,V,spec = svd(T, idx_left)
    return entropy(spec)
end