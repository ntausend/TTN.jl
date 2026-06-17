
# Used only for the two-legged center node: expand both bond indices at once.
# Each bond index has one block per QN sector, so offset accumulation is not needed.
function _enlarge_two_leg_tensor(T::ITensor, id_n::Tuple{Index, Index}, use_random)
    id_t = inds(T)
    @assert length(id_t) == length(id_n)
    if !hasqns(T)
        dims_n = ITensors.dim.(id_n)
        dims_o = ITensors.dim.(id_t)
        Ttn = use_random ? randn(eltype(T), dims_n...) : zeros(eltype(T), dims_n...)
        copyto!(view(Ttn, UnitRange.(1, dims_o)...), array(T))
        return ITensor(Ttn, id_n...)
    end

    Tpt = ITensors.tensor(T)
    itpt = inds(Tpt)
    id_n_l, id_n_r = id_n

    Ttn = use_random ? random_itensor(eltype(T), flux(T), id_n_l, id_n_r) : ITensor(eltype(T), 0, flux(T), id_n_l, id_n_r)
    occursin("Cu", string(typeof(T.tensor))) && (Ttn = adapt(CuArray, Ttn))

    Tnt = ITensors.tensor(Ttn)
    itnt = inds(Tnt)

    target_l_pos = _qn_block_positions(space(itnt[1]))
    target_r_pos = _qn_block_positions(space(itnt[2]))

    foreach(ITensors.eachnzblock(Tpt)) do bl
        sp_l = ITensors.getblock(itpt[1], bl[1])
        sp_r = ITensors.getblock(itpt[2], bl[2])
        qn_l = first(sp_l)
        qn_r = first(sp_r)

        id_l = get(target_l_pos, qn_l, 0)
        id_r = get(target_r_pos, qn_r, 0)
        id_l == 0 && error("Could not find a matching left sector $(qn_l) in the padded tensor.")
        id_r == 0 && error("Could not find a matching right sector $(qn_r) in the padded tensor.")

        src_blv = ITensors.blockview(Tpt, bl)
        tgt_blv = ITensors.blockview(Tnt, Block(id_l, id_r))
        copyto!(view(tgt_blv, 1:last(sp_l), 1:last(sp_r)), src_blv)
    end

    return itensor(Tnt)
end


# Expands the last index of T from id_old to id_n without any combiner.
# For each non-zero block (bl_physical..., bl_old), the matching target block is
# (bl_physical..., bl_n) and the source data fills the first d_old slice of the
# last dimension. This avoids two full tensor reshapes from the old combiner path.
function _enlarge_tensor(T::ITensor, id_tu, id_old, id_n, use_random)
    @assert all(inds(T) .== [id_tu..., id_old])
    new_inds = [id_tu..., id_n]

    if !hasqns(T)
        dims_n = dim.(new_inds)
        dims_o = dim.([id_tu..., id_old])
        arr_n  = use_random ? randn(eltype(T), dims_n...) : zeros(eltype(T), dims_n...)
        copyto!(view(arr_n, ntuple(i -> 1:dims_o[i], length(dims_o))...), array(T))
        return ITensor(arr_n, new_inds...)
    end

    Ttn = use_random ? random_itensor(eltype(T), flux(T), new_inds...) :
                       ITensor(eltype(T), 0, flux(T), new_inds...)
    occursin("Cu", string(typeof(T.tensor))) && (Ttn = adapt(CuArray, Ttn))

    Tpt = ITensors.tensor(T)
    Tnt = ITensors.tensor(Ttn)
    N   = ndims(Tpt)

    src_inds     = inds(Tpt)
    target_n_pos = _qn_block_positions(space(inds(Tnt)[N]))

    foreach(ITensors.eachnzblock(Tpt)) do bl
        sp_last = ITensors.getblock(src_inds[N], bl[N])
        qn_r    = first(sp_last)
        d_old_r = last(sp_last)
        id_r    = get(target_n_pos, qn_r, 0)
        id_r == 0 && error("Could not find matching sector $(qn_r) in expanded tensor.")

        bl_n    = Block(ntuple(i -> bl[i], N - 1)..., id_r)
        src_blv = ITensors.blockview(Tpt, bl)
        tgt_blv = ITensors.blockview(Tnt, bl_n)
        copyto!(selectdim(tgt_blv, N, 1:d_old_r), src_blv)
    end

    return itensor(Tnt)
end

function _qn_capacities(sp::AbstractVector{<:Pair{QN, Int}})
    caps = Dict{QN, Int}()
    for (qn, dd) in sp
        caps[qn] = get(caps, qn, 0) + dd
    end
    return caps
end

function _qn_block_positions(sp::AbstractVector{<:Pair{QN, Int}})
    positions = Dict{QN, Int}()
    for (pos, (qn, _)) in enumerate(sp)
        haskey(positions, qn) || (positions[qn] = pos)
    end
    return positions
end

# Returns an index with each QN sector dimension >= the corresponding dimension
# in j_min. Also restores sectors present in j_min but dropped from j entirely
# (e.g. when intersect with id_max omits a sector). This prevents the expander
# from shrinking or losing existing bond sectors.
function _lower_bound_index(j::Index{Vector{Pair{QN, Int}}}, j_min::Index{Vector{Pair{QN, Int}}}; tags=tags(j))
    min_caps = _qn_capacities(space(j_min))
    j_caps   = _qn_capacities(space(j))
    new_space = map(first(ITensors.combineblocks(space(j)))) do (qn, d)
        qn => max(d, get(min_caps, qn, 0))
    end
    # re-add sectors present in j_min but completely absent in j
    for (qn, d) in first(ITensors.combineblocks(space(j_min)))
        haskey(j_caps, qn) || push!(new_space, qn => d)
    end
    return Index(new_space; tags=tags, dir=dir(j))
end

function _lower_bound_index(j::Index{Int64}, j_min::Index{Int64}; tags=tags(j))
    return Index(max(dim(j), dim(j_min)); tags=tags, dir=dir(j))
end

function _merge_index(idx::Index{Vector{Pair{QN, Int}}}; tags = tags(idx))
    merged = Pair{QN, Int}[]
    seen = Dict{QN, Int}()
    for (qn, dd) in space(idx)
        if haskey(seen, qn)
            seen[qn] += dd
            merged[seen[qn]] = qn => seen[qn]
        else
            push!(merged, qn => dd)
            seen[qn] = length(merged)
        end
    end
    return Index(merged; tags = tags, dir = dir(idx))
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
    return Index(min(ITensors.dim(j1),ITensors.dim(j2)); tags = tags, dir = dir(j1))
end

function _padding(j::Index{Vector{Pair{QN, Int}}}, jp::Index{Vector{Pair{QN, Int}}}, p::Float64, min::Int; tags = "Padded", kwargs...)
    @assert 0≤p≤1
    dir(j) ≠ dir(jp) && error(
    "To pad two indices, they must have the same direction. Trying to pad indices $j and $jp.",
    )
    isempty(space(jp)) && return j
    cmbblocks_pd = map(first(ITensors.combineblocks(space(jp)))) do (q, d)
            return q => max(round(Int, d*p), min)
    end
    jp_new = ITensors.Index(cmbblocks_pd; dir=dir(jp))

    return directsum(j, jp_new; tags = tags)

end

function _padding(j::Index{Vector{Pair{QN, Int}}}, jp::Index{Vector{Pair{QN, Int}}}, p::Int, min::Int; tags = "Padded", kwargs...)
    @assert 0≤p
    dir(j) ≠ dir(jp) && error(
    "To pad two indices, they must have the same direction. Trying to pad indices $j and $jp.",
    )
    isempty(space(jp)) && return j
    cmbblocks_pd = map(first(ITensors.combineblocks(space(jp)))) do (q, d)
            return q => max(Base.min(d, p), min)
    end
    jp_new = ITensors.Index(cmbblocks_pd; dir=dir(jp))

    return directsum(j, jp_new; tags = tags)
end

function _padding(j::Index{Int64}, jp::Index{Int64}, p::Float64, min::Int; tags = "Padded", kwargs...)
    @assert 0≤p≤1
    jp_new = Index(max(round(Int, ITensors.dim(jp)*p), min); tags = tags, dir = dir(jp))
    return directsum(j, jp_new; tags = tags)
end
function _padding(j::Index{Int64}, jp::Index{Int64}, p::Int, min_dim::Int; tags = "Padded", kwargs...)
    @assert 0≤p
    jp_new = Index(min(ITensors.dim(jp), p); tags = tags, dir = dir(jp))
    return directsum(j, jp_new; tags = tags)
end


shift_qn(j::Index{Int64}, ::Nothing) = j

function shift_qn(j::Index{Vector{Pair{QN, Int}}}, q_shift::QN)
    n_states = map(space(j)) do (qn, dd)
        qn_n = qn - dir(j)*q_shift
        qn_n => dd
    end
    return Index(n_states; tags = tags(j), dir = dir(j))
end
