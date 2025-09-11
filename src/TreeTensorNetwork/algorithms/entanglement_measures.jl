 
function entropy2(s::Spectrum)
  S = 0.0
  eigs_s = Array(eigs(s))
  isnothing(eigs_s) && error("Spectrum does not contain any eigenvalues, cannot compute the entropy")
  for p in eigs_s
    p > 1e-13 && (S -= p * log(p))
  end
  return S
end
                    
"""
```julia
   entanglement_entropy(ttn::TreeTensorNetwork, pos_right::Tuple{Int, Int}, pos_left::Tuple{Int, Int})
```

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
function entanglement_entropy(ttn::TreeTensorNetwork, pos_right::Tuple{Int, Int}, pos_left::Tuple{Int, Int}; use_gpu = false)
    net = network(ttn)
    ttnc = copy(ttn)
    # checking if pos_right is contained in the network
    check_valid_position(net, pos_right)
    # check if layer number is not physical
    @assert pos_right[1] > 0
    # check if pos_left is either in the child nodes/ or being the parent node
    @assert (pos_left ∈ child_nodes(net, pos_right) || pos_left == parent_node(net, pos_left))

    # now move the orthogonality centrum
    ttnc = use_gpu ? move_ortho!(ttnc, pos_right, Dict()) : move_ortho!(ttnc, pos_right)
    T = use_gpu ? gpu(ttnc[pos_right]) : ttnc[pos_right]
    
    # this only works for ITensors...
    # getting the indices for decomposition, this only contains pos_left link
    if pos_left[1] == 0
        # pos_left is a physical site.. we need to filter differently
        idx_left = inds(T; tags = "Site,n=$(pos_left[2])")
    else
        idx_left = inds(T; tags = "Link,nl=$(pos_left[1]),np=$(pos_left[2])")
    end

    U,S,V,spec = svd(T, idx_left)
    return entropy2(spec)
end

"""
```julia
   entanglement_entropy(ttn::TreeTensorNetwork, pos_path::Vector{Tuple{Int, Int}})
```
Calculates the entanglement entropy along a path defined by `pos_path` in the network.
This is effectively a series of bipartionings where region A is given by the subtree of each node in `pos_path` and
B by the rest of the network. The algorithm cuts the leg by moving the orthogonality centrum to the parent node of each position
in `pos_path` and forming the SVD on that node.
Inputs: 
    - ttn defining: the Tree Tensor Network
    - pos_path: Vector of positions in the network defining the path. The tensor at the parent node of each position will
      be used for decomposition
"""

function entanglement_entropy(ttn::TreeTensorNetwork, pos_path::Vector{Tuple{Int, Int}}; use_gpu = false)
    net = network(ttn)
    ttnc = copy(ttn)

    maxL = number_of_layers(net)
    entropies = Float64[]
    # entropies = Dict{Tuple{Int, Int}, Float64}()

    for pos_left in pos_path
        pos_left[1] == maxL && continue
        pos_right = parent_node(network(ttn), pos_left) 

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
        push!(entropies, entropy2(spec))
        # entropies[pos_left] = entropy2(spec)
    end
    return entropies
end

function entanglement_entropy(ttn::TreeTensorNetwork; use_gpu = false, start=(number_of_layers(network(ttn))-1, 1), include_layer0 = true)
    net = network(ttn)
    ttnc = copy(ttn)

    maxL = number_of_layers(net)
    # entropies = Float64[]
    entropies = Dict{Tuple{Int, Int}, Float64}()

    route = ttn_traversal_least_steps(net; start, include_layer0, exclude_topnode=true);

    for pos_left in route.visit_order
        pos_left[1] == maxL && continue
        pos_right = parent_node(network(ttn), pos_left) 

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
        # push!(entropies, TTN.entropy2(spec))
        entropies[pos_left] = entropy2(spec)

    end
    return entropies
end