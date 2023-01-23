"""
    contract_tensors(tensor_list::Vector{<:AbstractTensorMap}, index_list::Vector{Vector{Int}})

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
function contract_tensors(tensor_list::Vector{<:AbstractTensorMap}, index_list::Vector{Vector{K}}) where K
    
    #= why is this slower than the version of Wladi? Would expect to be similar...
    fl_index_list = Iterators.flatten(index_list)
    n_count = StatsBase.countmap(fl_index_list)
    
    defects = findall(x -> x>2, n_count)
    isempty(defects) || error("Indices $(defects) occure more than twice.")
    
    double_occurence = findall(x -> x == 2, n_count)
    single_occurence = findall(x -> x == 1, n_count)
    =#
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

    contract_list = map(index_list) do list
        map(list) do pp
            return pp in unique_indices ? -findall(isequal(pp), unique_indices)[1] : findall(isequal(pp), double_indices)[1]
        end
    end
    return unique_indices, @ncon(tensor_list, contract_list)
end

# also return the unique_indices to have compatiblity with other methods... need to rethink about this
# whole construct..., for ITensors it is useless, since contract_tensors simply contruct the already
# equal legs
function contract_tensors(tensor_list::Vector{<:ITensor}, index_list::Vector{Vector{K}}) where{K}
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