
function _enlarge_two_leg_tensor(T::ITensor, id_n::Tuple{Index, Index}, use_random)
    id_t = inds(T)
    @assert length(id_t) == length(id_n)
    if !hasqns(T)
        dims_n = ITensors.dim.(id_n)
        dims_o = ITensors.dim.(id_t)
        Ttn = use_random ? randn(eltype(T), dims_n...) : zeros(eltype(T), dims_n...)
        view(Ttn, UnitRange.(1, dims_o)...) .= array(T)
        return ITensor(Ttn, id_n...)
    end

    # in case of qns we have to do more work
    # first create a dummy Tensor with the correct sectors
    Ttn = use_random ? randomITensor(eltype(T), flux(T), id_n...) : ITensor(eltype(T), 0, flux(T), id_n...)

    Tpt = ITensors.tensor(T)
    Tnt = ITensors.tensor(Ttn)

    itpt = inds(Tpt)
    itnt = inds(Tnt)
    
    sp_tnt_l = space.(itnt[1])
    sp_tnt_r = space.(itnt[2])

    foreach(ITensors.eachnzblock(Tpt)) do bl
        # qnnumbers
        sp_l = ITensors.getblock(itpt[1], bl[1])
        sp_r = ITensors.getblock(itpt[2], bl[2])
        qn_l = first(sp_l)
        qn_r = first(sp_r)

        id_bl_tnt_l = findfirst(q -> isequal(q, qn_l), first.(sp_tnt_l))
        id_bl_tnt_r = findfirst(q -> isequal(q, qn_r), first.(sp_tnt_r))
        bl_tnt = Block(id_bl_tnt_l, id_bl_tnt_r)

        Tntblv = ITensors.blockview(Tnt, bl_tnt)
        # sanity check
        #@assert last(sp_tnt_l[id_bl_tnt_l]) == last(sp_l)
        view(Tntblv, 1:last(sp_l), 1:last(sp_r)) .= ITensors.blockview(Tpt, bl)
    end

    # now rebuild the ITensor and split the left indices
    Tn = itensor(Tnt)

    return Tn
end


# enlarges the tensor with indices id_tu, and id_old -> id_n
# use_random = true ->  padding with random number otherwise with zeros
function _enlarge_tensor(T::ITensor, id_tu, id_old, id_n, use_random)
    
    @assert all(inds(T) .== [id_tu..., id_old])
    
    cl = combiner(id_tu...)
    Tp = cl*T
    Tn = _enlarge_two_leg_tensor(Tp, (combinedind(cl), id_n), use_random)

    return Tn*dag(cl)
    #=
    # simple enlargement
    if !hasqns(T)
        dims_n = vcat(ITensors.dim.(id_tu), ITensors.dim(id_n))
        dims_o = vcat(ITensors.dim.(id_tu), ITensors.dim(id_old))
        Ttn = use_random ? randn(eltype(T), dims_n...) : zeros(eltype(T), dims_n...)
        view(Ttn, UnitRange.(1, dims_o)...) .= array(T)
        return ITensor(Ttn, id_tu..., id_n)
    end

    # in case of qns we have to do more work
    # first create a dummy Tensor with the correct sectors
    Ttn = use_random ? randomITensor(eltype(T), flux(T), id_tu..., id_n) : ITensor(eltype(T), 0, flux(T), id_tu..., id_n)

    # now we want to set the subtensor defined by T
    # for this, we first build the combined left index
    cl = combiner(id_tu...)
    Tpt  = ITensors.tensor(cl*T)
    Tnt = ITensors.tensor(cl*Ttn)
    # Iterate through all blocks in Tpt and find the corresponding block in Tnt
    itpt = inds(Tpt)
    itnt = inds(Tnt)
    
    sp_tnt_l = space.(itnt[1])
    sp_tnt_r = space.(itnt[2])

    foreach(ITensors.eachnzblock(Tpt)) do bl
        # qnnumbers
        sp_l = ITensors.getblock(itpt[1], bl[1])
        sp_r = ITensors.getblock(itpt[2], bl[2])
        qn_l = first(sp_l)
        qn_r = first(sp_r)

        id_bl_tnt_l = findfirst(q -> isequal(q, qn_l), first.(sp_tnt_l))
        id_bl_tnt_r = findfirst(q -> isequal(q, qn_r), first.(sp_tnt_r))
        bl_tnt = Block(id_bl_tnt_l, id_bl_tnt_r)

        Tntblv = ITensors.blockview(Tnt, bl_tnt)
        # sanity check
        @assert ITensors.last(sp_tnt_l[id_bl_tnt_l]) == last(sp_l)
        view(Tntblv, 1:last(sp_l), 1:last(sp_r)) .= ITensors.blockview(Tpt, bl)
    end

    # now rebuild the ITensor and split the left indices
    Tn = itensor(Tnt) * dag(cl)
    =#

    return Tn
end

function complement(j1::Index, j2::Index; tags = "Complement", remove_trivial_blocks = false)
    
    dir(j1) ≠ dir(j2) && error(
"To form the complement of two indices, they must have the same direction. Trying to complement indices $(j1) and $(j2).",
)
    @assert hasqns(j1) == hasqns(j2)
    
    if !hasqns(j1)
        dim1 = dim(j1)
        dim2 = dim(j2)
        return Index(max(dim1 - dim2, 0); tags = tags, dir = dir(j1))
    end
    

    sp1 = first(ITensors.combineblocks(space(j1)))
    sp2 = first(ITensors.combineblocks(space(j2)))

    # how to handle multiple appearances of spaces in sp1 later??
    
    sec1 = first.(sp1)
    sec2 = first.(sp2)
    dim1  = last.(sp1)
    dim2  = last.(sp2)
    
    sec_int = intersect(sec1, sec2)
    
    ps1 = map(sec_int) do s
        findfirst(q ->  isequal(q, s), sec1)
    end
    ps2 = map(sec_int) do s
        findfirst(q ->  isequal(q, s), sec2)
    end
    
    
    dim1_red = dim1[ps1]#map(p -> dim1[p], ps1)
    dim2_red = dim2[ps2]#map(p -> dim2[p], ps2)
    
    dim_red  = map(minimum, zip(dim1_red, dim2_red)) 
    
    complement_set = map(zip(sec1, dim1)) do (sp, dd)
        idx_app = findall(q -> isequal(q, sp), sec_int)
        # sp does not appear in the intersection -> retrun the full dimension
        isempty(idx_app) && (return sp => dd)
        # sp does appear -> return the difference of the dimensions
        idx_app = only(idx_app)
        dim_crr = dd - dim_red[idx_app] 
        iszero(dim_crr) && remove_trivial_blocks && (return missing)
        return sp => dim_crr
    end

    complement_set = collect(skipmissing(complement_set))
    isempty(complement_set) && (return ITensor(0))

    return Index(complement_set; tags = tags, dir = dir(j1))
end


function _intersect_blocks(j1::Index, j2::Index)
    sp1 = space(j1)
    sp2 = space(j2)
    
    sec1 = first.(sp1)
    sec2 = first.(sp2)
    dim1  = last.(sp1)
    dim2  = last.(sp2)
    
    sec_int = intersect(sec1, sec2)
    
    ps1 = map(sec_int) do s
        findall(q ->  isequal(q, s), sec1)
    end
    ps2 = map(sec_int) do s
        findall(q ->  isequal(q, s), sec2)
    end
    
    
    dim1_red = map(p -> sum(dim1[p]), ps1)
    dim2_red = map(p -> sum(dim2[p]), ps2)
    
    dim_red  = map(minimum, zip(dim1_red, dim2_red)) 
    
    return map(qd -> Pair(qd...), zip(sec_int, dim_red))
end

function Base.intersect(j1::Index{Vector{Pair{QN,Int}}}, j2::Index{Vector{Pair{QN,Int}}}; tags = "Intersect")
    dir(j1) ≠ dir(j2) && error(
"To intersect two indices, they must have the same direction. Trying to intersect indices $j1 and $j2.",
)
    blcks_intersection = _intersect_blocks(j1, j2)
    return Index(blcks_intersection; dir=dir(j1), tags=tags)
end

function Base.intersect(j1::Index{Int64}, j2::Index{Int64}; tags = "Intersect")
    return Index(min(ITensors.dim(j1),ITensors.dim(j2)); tags = tags)
end

function _padding(j::Index{Vector{Pair{QN, Int}}}, jp::Index{Vector{Pair{QN, Int}}}, p::Real; tags = "Padded", kwargs...)
    @assert 0≤p≤1
    dir(j) ≠ dir(jp) && error(
    "To pad two indices, they must have the same direction. Trying to pad indices $j and $jp.",
    )
    cmbblocks_pd = map(first(ITensors.combineblocks(space(jp)))) do (q, d)
            return q => round(Int, d*p)
    end
    jp_new = ITensors.Index(cmbblocks_pd; dir=dir(jp))
    
    return directsum(j, jp_new; tags = tags)
    
end

function _padding(j::Index{Vector{Pair{QN, Int}}}, jp::Index{Vector{Pair{QN, Int}}}, p::Int; tags = "Padded", kwargs...)
    @assert 0≤p
    dir(j) ≠ dir(jp) && error(
    "To pad two indices, they must have the same direction. Trying to pad indices $j and $jp.",
    )
    cmbblocks_pd = map(first(ITensors.combineblocks(space(jp)))) do (q, d)
            return q => min(d, p)
    end
    jp_new = ITensors.Index(cmbblocks_pd; dir=dir(jp))
    
    return directsum(j, jp_new; tags = tags)
end

function _padding(j::Index{Int64}, jp::Index{Int64}, p::Real; tags = "Padded", kwargs...)
    @assert 0≤p≤1
    jp_new = Index(round(Int, d*p))
    return directsum(j, jp_new; tags = tags)
end
function _padding(j::Index{Int64}, jp::Index{Int64}, p::Int; tags = "Padded", kwargs...)
    @assert 0≤p
    jp_new = Index(min(ITensors.dim(jp), p))
    return directsum(j, jp_new; tags = tags)
end