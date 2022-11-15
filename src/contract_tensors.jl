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
function contract_tensors(tensor_list::Vector{<:AbstractTensorMap}, index_list::Vector{Vector{Int}})
    fl_index_list = Iterators.flatten(index_list)
    
    #= why is this slower than the version of Wladi? Would expect to be similar...
    n_count = StatsBase.countmap(fl_index_list)
    
    defects = findall(x -> x>2, n_count)
    isempty(defects) || error("Indices $(defects) occure more than twice.")
    
    double_occurence = findall(x -> x == 2, n_count)
    single_occurence = findall(x -> x == 1, n_count)
    =#
    unique_indices = Any[]
    double_indices = Any[]
    flatIndexList = collect(Iterators.flatten(index_list))#vcat(indexList...)  
    
    while !(isempty(flatIndexList))
        el = popfirst!(flatIndexList)
        if !(el in flatIndexList) && !(el in double_indices)
            append!(unique_indices, [el])
        else
            append!(double_indices, [el])
        end
    end

    
    index_list_c = deepcopy(index_list)
    foreach(enumerate(unique_indices)) do (jj,idx_open)
        foreach(index_list_c) do list
            replace!(list, idx_open => -jj)
        end
    end
    
    return unique_indices, @ncon(tensor_list, index_list_c)
    
end