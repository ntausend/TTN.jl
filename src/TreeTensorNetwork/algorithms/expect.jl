# expectation value of a onsite operator i.e.: <n_j>

"""
```julia
   expect(ttn::TreeTensorNetwork, op)
```
Returns the expectation value of the local observable `op` evaluated at every position of the lattice.
"""
function ITensorMPS.expect(ttn::TreeTensorNetwork, op)
    physlat = physical_lattice(network(ttn))
    res = map(eachindex(physlat)) do (pos)
        expect(ttn, op, pos)
    end
    dims = size(physlat)
    return reshape(res, dims)
end
"""
```julia
   expect(ttn::TreeTensorNetwork, op, pos:Ntuple)
```
Returns the expectation value of the local observable `op` evaluated at the position `pos` given as the d-dimensional coordinate.
"""
function ITensorMPS.expect(ttn::TreeTensorNetwork, op, pos::NTuple)
    return expect(ttn, op, linear_ind(physical_lattice(network(ttn)),pos))
end

"""
```julia
   expect(ttn::TreeTensorNetwork, op, pos:Int)
```
Returns the expectation value of the local observable `op` evaluated at the position `pos` given as the linearized coordinate.
"""
function ITensorMPS.expect(ttn::TreeTensorNetwork, _op, pos::Int)

    net = network(ttn)
    ttnc = copy(ttn)

    idx = siteinds(net)[pos]
    O   = convert_cu(op(_op, idx), ttn[(1,1)])

    # linear position in the D-dimensional lattice
    ch_pos = (0,pos)
    # finding parent node position

    parent_pos = parent_node(net, ch_pos)

    # move ortho_center to parent pos
    move_ortho!(ttnc, parent_pos)

    # find the index of the child
    T = ttnc[parent_pos]
    # perform the contraction
    res = dot(T, noprime(O*T))

    return res
end
