"""
```julia
    contract_tensors(tensor_list::Vector{<:AbstractTensorMap}, index_list::Vector{Vector{Int}})
```

A function contracting the list of tensors `tensor_list` according to a `index_list` which defines
a index covering for every tensor in `tensor_list`. The function searches for double occuring indices
and calls th `@ncon` macro from `TensorKit` to contract these double occuring indices.

### Arguments:

---

- tensor_list: List of AbstractTensorMaps with arbitrary number of indices
- index_list: Vector of integer vectors labeling the legs of every tensor, i.e.
`index_list[j]` labels `tensor_list[j]`. All double occuring indices are contracted, the
single occuring indices are left open.

### Returns

---

The function returns as a first object a vector of all labels of the open indices. Their order also
defines the order of the legs in the resulting tensor. The second argument is the contracted tensor.

"""
# also return the unique_indices to have compatiblity with other methods... need to rethink about this
# whole construct..., for ITensors it is useless, since contract_tensors simply contruct the already
# equal legs
function contract_tensors(tensor_list::Vector{<:ITensor}, index_list::Vector{Vector{K}}) where {K}
    unique_indices = K[]
    double_indices = K[]
    flatIndexList = collect(Iterators.flatten(index_list))#vcat(indexList...)

    while !(isempty(flatIndexList))
        el = popfirst!(flatIndexList)
        if !(el in flatIndexList) && !(el in double_indices)
            append!(unique_indices, el)
        else
            append!(double_indices, el)
        end
    end

    opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
    return unique_indices, contract(tensor_list; sequence = opt_seq)
    #return unique_indices, reduce(*, tensor_list, init = ITensor(1))
end
