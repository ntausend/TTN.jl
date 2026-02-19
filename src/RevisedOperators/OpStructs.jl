const NEXT_Op_GPU_ID = Ref{Int}(0)

"""
new_Op_GPU_id()
— return a fresh unique id; increments the counter.
"""
new_Op_GPU_id() = begin
  id = NEXT_Op_GPU_ID[]
  NEXT_Op_GPU_ID[] += 1
  return id
end

#=
Link struct unnecessary, but kept for reference
struct Link
  node::Tuple{Int,Int}  # (layer,node) 
  leg::Int8              # Which leg: 0=parent/top, 1=child1/left, 2=child2/right
end
=#

# ───────────────────────────────────────────────
# Low-level definition of a custom operator type
# ───────────────────────────────────────────────

"""
    Op

Represents a one- or multi-site operator with metadata and a precomputed LCAmap.

Fields:
- `id`        : Unique identifier to associate parts of a multi-site operator
- `site`     : Tuple of site indices the operator acts on (site1, site2, ...)
- `op`       : Tuple of ITensors, one per site
- `original_length` : Original length of the operator (e.g. number of sites)
- `length`   : Length of the operator after splitting (e.g. number of sites after splitting)
"""
struct Op_GPU
  id::Int
  site::Tuple{Int,Int}
  op::ITensor
  original_length::Int
  length::Int
end

length(op::Op_GPU) = op.length
# Op_GPU(id::Int, site::Tuple{Int,Int}, op::ITensor) = Op_GPU(id, (site,), (op,))

op_reduction(op::Op_GPU) = op.original_length - op.length

function ==(a::Op_GPU, b::Op_GPU)
    a.id == b.id && # isnt necessarily equal due to global unique id counter
    a.site == b.site &&
    a.op == b.op &&
    a.original_length == b.original_length &&
    a.length == b.length
    # all(==(true), isapprox.(a.op, b.op))  # or define more robust ITensor comparison
end

# function isapprox(a::Op_GPU, b::Op_GPU; atol=1e-12, rtol=1e-12)
#     a.id == b.id &&
#     a.site == b.site &&
#     isapprox(a.op, b.op)
#     # all(isapprox.(a.op, b.op; atol=atol, rtol=rtol))
# end


# ─────────────────────────────────────────────
# Mid-level: Tree tensor product operator (TPO)
# ─────────────────────────────────────────────

"""
    TPO

Stores a list of custom `Op` terms derived from the original `OpSum`.

Fields:
- `terms` :: Vector{Op}
"""
struct TPO_GPU
    terms::Vector{Op_GPU}
end


getindex(tpo::TPO_GPU, i::Int) = tpo.terms[i]
length(tpo::TPO_GPU) = length(tpo.terms)
iterate(tpo::TPO_GPU, state=1) = state > length(tpo) ? nothing : (tpo[state], state+1)
firstindex(tpo::TPO_GPU) = 1
lastindex(tpo::TPO_GPU) = length(tpo)

"""
init_Op_GPU_id_counter!(tpo::Vector{Op_GPU})
— after you build your TPO (and have assigned its original ids),
  call this to seed the counter one above the max you’ve used.
"""

function init_Op_GPU_id_counter!(tpo::Vector{Op_GPU})
  if isempty(tpo)
    NEXT_Op_GPU_ID[] = 1
  else
    NEXT_Op_GPU_ID[] = maximum(g.id for g in tpo) + 1
  end
  return NEXT_Op_GPU_ID[]
end


# ─────────────────────────────────────────────
# Top-level: Projected TPO structure (ProjTPO)
# ─────────────────────────────────────────────

"""
    ProjTPO

Encapsulates the TTN network, current ortho_center, and link operators needed for sweeping.

Fields:
- `net`          : tree network structure, e.g. `BinaryNetwork`
- `tpo`          : Tree product operator
- `ortho_center` : Tuple{layer, node} denoting the orthogonality center
- `link_ops`     : Dict{((layer, node), leg) => Vector{ITensor}} with link-operators
- `lca_map`      : Dict{(site1, site2) => Dict{(layer, node) => (lca_layer, lca_node, lca_links)}} mapping
                each site to its rerooted lowest common ancestor (LCA) under the current OC.
"""
mutable struct ProjTPO_GPU{N<:AbstractNetwork, T} <: AbstractProjTPO{N, T}
    net::N
    tpo::TPO_GPU
    ortho_center::Tuple{Int,Int}
    link_ops::Dict{Tuple{Tuple{Int,Int},Int}, Vector{Op_GPU}}
end

# function ProjTPO_GPU(net::N, tpo::TPO_GPU, oc::Tuple{Int,Int}, link_ops::Dict{Tuple{Tuple{Int,Int}, Int}, Vector{Op_GPU}}) where {N<:AbstractNetwork}
#     return ProjTPO_GPU{N, ITensor}(net, tpo, oc, link_ops)
# end

# network(ptpo::ProjTPO_GPU)      = ptpo.net
# tpo(ptpo::ProjTPO_GPU)     = ptpo.tpo
# oc(ptpo::ProjTPO_GPU)         = ptpo.oc
link_ops(ptpo::ProjTPO_GPU) = ptpo.link_ops

function ProjTPO_GPU(tpo::TPO_GPU, ttn::TreeTensorNetwork{N, T};
                       oc = Tuple(ttn.ortho_center), use_gpu::Bool = false, node_cache = Dict()) where {N, T}

    link_ops = upflow_to_root(ttn.net, ttn, tpo, oc; use_gpu = use_gpu, node_cache = node_cache)
    return ProjTPO_GPU{N, T}(ttn.net, tpo, oc, link_ops)
end


# ─────────────────────────────────────────────
# Helper function to build a TPO from an OpSum 
# ─────────────────────────────────────────────

"""
    build_tpo_from_opsum(ampo::OpSum, lat)
Builds a `TPO_GPU` from an `OpSum` by extracting terms and converting them to `Op_GPU`.

`ampo` is an `OpSum` containing the operator terms, and `lat` is the lattice structure.

This function constructs `Op_GPU` instances for each term,
and returns a `TPO_GPU` containing all the operator groups.
N-site operators are splitted into individual `Op_GPU` instances matched by their unique ID.
"""

function build_tpo_from_opsum(ampo::OpSum, lat::AbstractLattice)
    physidx = siteinds(lat)
    op_s    = Op_GPU[]
    terms   = filter(t -> !isapprox(coefficient(t),0),
                     ITensorMPS.terms(ITensorMPS.sortmergeterms(ampo)))

    op_id = 1
    for term in terms
        coeff = coefficient(term)
        len   = length(term)

        for (k, op) in enumerate(ITensors.terms(term))
            opname       = ITensors.which_op(op)
            siteidx      = ITensors.site(op)
            siteidx isa Int && (siteidx = (siteidx,))        # 1D→tuple
            siteidx_lin  = linear_ind(lat, siteidx)

            # multiply the coefficient **once**, e.g. on the first factor
            factor = (k == 1) ? coeff : 1.0
            op_t = factor * ITensors.op(physidx[siteidx_lin], opname;
                                     ITensors.params(op)...)

            push!(op_s, Op_GPU(op_id, (0, siteidx_lin), op_t, len, len))
        end
        op_id += 1
    end

    init_Op_GPU_id_counter!(op_s)
    return TPO_GPU(op_s)
end

TPO_GPU(ampo::OpSum, lat::AbstractLattice) = build_tpo_from_opsum(ampo, lat)

"""
    ops_on_node(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}) -> Vector{Op_GPU}

Return all operator terms (`Op_GPU`s) that are associated with any of the
three links connected to `pos` = (layer, node) in the TTN:
- parent link
- first child link
- second child link
"""
function ops_on_node(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int})
    net = ptpo.net
    link_ops = ptpo.link_ops

    links = Tuple{Tuple{Int,Int}, Int}[]
    parent = parent_node(net, pos)
    if parent ≠ nothing
        push!(links, (parent, which_child(net, pos)))
    end
    for (i, child) in enumerate(child_nodes(net, pos))
        push!(links, (pos, i))
    end

    ops = Op_GPU[]
    for link in links
        append!(ops, get(link_ops, link, Op_GPU[]))
    end
    return ops
end


# Find all operators by their current length
function get_length_terms(tpo::TPO_GPU, len::Int)
    return filter(op -> op.length == len, tpo.terms)
end

function get_length_terms(ops::Vector{Op_GPU}, len::Int)
    filter(g -> g.length == len, ops)
end

function get_length_terms(net::BinaryNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{Op_GPU}}, (layer, node)::Tuple{Int,Int}, len::Int)
    # gather all Op_GPU objects by length len
    # of (layer,node) and on the link to its parentS
    bucket = Dict{Int, Vector{Op_GPU}}()

    # the three links we care about
    links = (
        ((layer, node), 1),                          # first-child link
        ((layer, node), 2),                          # second-child link
        (parent_node(net, (layer, node)),
         which_child(net, (layer, node)))            # parent link
    )

    for link in links
        # get(link_ops, link, Vector{Op_GPU}()) → empty vector if link not present
        for op in get(link_ops, link, Vector{Op_GPU}())
            if op.length == len
                push!(get!(bucket, op.length, Vector{Op_GPU}()), op)
            end
        end
    end

    return bucket
end

# Collect all terms acting on a specific site
function get_site_terms(tpo::TPO_GPU, target_site::Tuple{Int,Int})
    # Filter groups where the target site is in the sites of the group
    return filter(op -> op.site == target_site, tpo.terms)
end

function get_site_terms(
        net::BinaryNetwork,
        link_ops::Dict{Tuple{Tuple{Int,Int},Int}, Vector{Op_GPU}},
        site::Tuple{Int,Int},
    )::Vector{Op_GPU}

    terms = Vector{Op_GPU}()

    # 1. parent link (absent if `site` happens to be the global root)
    if site != (number_of_layers(net), 1)
        parent = parent_node(net, site)
        link   = (parent, which_child(net, site))
        append!(terms, get(link_ops, link, Vector{Op_GPU}()))
    end

    # 2. on‑site link (child index == 0)
    onsite_link = (site, 0)
    append!(terms, get(link_ops, onsite_link, Vector{Op_GPU}()))

    return terms
end

# Find all operator groups by their unique ID
function get_id_terms(tpo::TPO_GPU, target_id::Int)
    return filter(op -> op.id == target_id, tpo.terms)
end

"""
    filter_id_term(link_ops, (layer, node), id) -> Vector{Op_GPU}

Return all `Op_GPU`s whose `op.id == id` that act on the two child links
of `(layer, node)` or the link to its parent.  If no such operators are
present, an empty vector is returned.
"""
function get_id_terms(net::BinaryNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{Op_GPU}}, (layer, node)::Tuple{Int,Int}, id::Int)
    # collect every Op_GPU on the three relevant links,
    # then pull out the bucket for the requested id
    get(get_id_terms(net::BinaryNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{Op_GPU}}, (layer, node)), id, Vector{Op_GPU}())
end

function get_id_terms(net::BinaryNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{Op_GPU}}, (layer, node)::Tuple{Int,Int})
    # gather all Op_GPU objects whose id appears on the two child links
    # of (layer,node) and on the link to its parent
    bucket = Dict{Int, Vector{Op_GPU}}()
    if layer == number_of_layers(net)
        links = (
        ((layer, node), 1),                 # first-child link
        ((layer, node), 2))                 # second-child link
    else
        # the three links we care about
        links = (
        ((layer, node), 1),                 # first-child link
        ((layer, node), 2),                 # second-child link
        (parent_node(net, (layer, node)),
         which_child(net, (layer, node))))  # parent link
    end
    for link in links
        # get(link_ops, link, Vector{Op_GPU}()) → empty vector if link not present
        for op in get(link_ops, link, Vector{Op_GPU}())
            push!(get!(bucket, op.id, Vector{Op_GPU}()), op)
        end
    end

    return bucket
end

## Find the index of a child node in the parent's child list
# which_child(net, parent, child) = findfirst(==(child), child_nodes(net, parent))
which_child(net::BinaryNetwork, child::Tuple{Int,Int}) = findfirst(==(child), child_nodes(net, parent_node(net, child)))







################### Temporary Location for VecProj_GPU definition and related functions ###################
mutable struct VecProj_GPU{N<:AbstractNetwork, T, P<:Tuple{Vararg{AbstractProjTPO{N, T}}}} <: AbstractProjTPO{N,T}
    net::N
    data::P
    ortho_center::Tuple{Int64,Int64}
end

function VecProj_GPU(all_projs::Tuple)
    ortho_center = all_projs[end].ortho_center

    for proj in all_projs
        @assert proj.ortho_center == ortho_center "All ProjTPOs in VecProj_GPU must have the same orthogonality center: found $(proj.ortho_center) and $ortho_center for $(typeof(proj))"
    end

    return VecProj_GPU(network(all_projs[1]), all_projs, ortho_center)
end

function VecProj_GPU(all_projs::Tuple, ttn::TreeTensorNetwork, target_oc::Tuple{Int,Int}; use_gpu::Bool = false, node_cache = Dict())
    # Move all projectors to the target orthogonality center
    updated_projs = map(all_projs) do proj
        if proj isa ProjTPO_GPU
            # For ProjTPO_GPU, use recalc_path_flows!
            current_oc = proj.ortho_center
            if current_oc != target_oc
                recalc_path_flows!(proj, ttn, current_oc, target_oc; use_gpu = use_gpu, node_cache = node_cache)
            end
            return proj
        elseif proj isa ProjTTN
            # For ProjTTN, use set_position! from AbstractProjectedTensorProductOperator
            current_oc = Tuple(proj.ortho_center)
            if current_oc != target_oc
                pth = connecting_path(network(ttn), current_oc, target_oc)
                if !isnothing(pth)
                    pth = vcat(current_oc, pth)
                    for (jj, pk) in enumerate(pth[1:end-1])
                        ism = ttn[pk]
                        update_environments!(proj, ism, pk, pth[jj+1])
                    end
                    proj.ortho_center .= target_oc
                end
            end
            return proj
        else
            error("Unsupported projector type: $(typeof(proj))")
        end
    end

    return VecProj_GPU(network(all_projs[1]), updated_projs, target_oc)
end

function set_position!(vecproj::VecProj_GPU{N,T}, ttn::TreeTensorNetwork{N,T}; use_gpu::Bool = false, node_cache = Dict()) where {N,T}
    oc_projtpo = ortho_center(vecproj)
    oc_ttn     = ortho_center(ttn)
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    all(oc_projtpo .== oc_ttn) && return vecproj

    # set position for ProjTPO_GPU
    recalc_path_flows!(vecproj.data[1], ttn, oc_projtpo, oc_ttn; use_gpu = use_gpu, node_cache = node_cache)

    # set position for ProjTTNs
    pth = connecting_path(network(ttn), oc_projtpo, oc_ttn)
    if !isnothing(pth)
        pth = vcat(oc_projtpo, pth)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end

    vecproj.ortho_center = oc_ttn
    return vecproj
end

function recalc_path_flows!(vecproj::VecProj_GPU, ttn::TreeTensorNetwork, oldroot::Tuple{Int,Int}, newroot::Tuple{Int,Int}; use_gpu::Bool = false, node_cache = Dict())
    
    oc_projtpo = ortho_center(vecproj)
    oc_ttn     = ortho_center(ttn)
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    # first, recalc path flows for the ProjTPO_GPU
    recalc_path_flows!(vecproj.data[1], ttn, oldroot, newroot; use_gpu = use_gpu, node_cache = node_cache)

    @assert oc_projtpo == oldroot "Expected ProjTPO ortho_center $(oc_projtpo) to match oldroot $(oldroot)"

    # then, recalc path flows for the ProjTTNs moving oldroot to newroot
    pth_forward = connecting_path(network(ttn), oldroot, newroot)
    if !isnothing(pth_forward)
        pth = vcat(oldroot, pth_forward)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end

    vecproj.ortho_center = newroot

    return vecproj
end

# highest-level dispatch for partial A on VecProj_GPU, dispatches to CPU or GPU implementation
function ∂A_GPU(ptpo::VecProj_GPU, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A_impl(ptpo, pos, Val(:gpu)) : _∂A_impl(ptpo, pos, Val(:cpu))
end

# lowest-level CPU implementation of partial A for VecProj_GPU
function _∂A_GPU(proj_operator::VecProj_GPU, pos::Tuple{Int,Int}; use_gpu::Bool = false)
   
    action_vec = map(ptpo -> ∂A_GPU(ptpo, pos; use_gpu = false), proj_operator.data)

    function action(T::ITensor)
        return mapreduce(+, action_vec) do act
            return act(T)
        end
    end
end

# maintains Noah's shape of partial A and dispatches to my CPU partial A implementation
function _∂A_impl(ptpo::VecProj_GPU, pos::Tuple{Int,Int}, ::Val{:cpu})
    return _∂A_GPU(ptpo, pos; use_gpu = false)
end

# maintains Noah's shape of partial A and executes GPU implementation
function _∂A_impl(ptpo::VecProj_GPU, pos::Tuple{Int,Int}, ::Val{:gpu})

    action_vec = map(ptpo -> ∂A_GPU(ptpo, pos; use_gpu = true), ptpo.data)

    function action(T::ITensor)
        
        T_gpu = gpu(T)

        return mapreduce(+, action_vec) do act
            return act(T_gpu)
        end
    end
end

# highest level catch for partial A, dispatches to CPU or GPU
function ∂A_GPU(proj_ttn::ProjTTN, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A_impl(proj_ttn, pos, Val(:gpu)) : _∂A_impl(proj_ttn, pos, Val(:cpu))
end

# lowest-level CPU implementation of ∂A for ProjTTN
function _∂A_GPU(proj_ttn::ProjTTN, pos::Tuple{Int,Int}; use_gpu::Bool = false)

    function action(T::ITensor)
        projector = contract(proj_ttn.local_env, dag(prime(proj_ttn.local_env)))
        return proj_ttn.weight * noprime(contract(T,projector))
    end
end

# maintains Noah's shape of partial A and dispatches to my CPU partial A implementation
function _∂A_impl(proj_ttn::ProjTTN, pos::Tuple{Int,Int}, ::Val{:cpu})
    return _∂A_GPU(proj_ttn, pos; use_gpu = false)
end

# maintains Noah's shape of partial A and executes GPU implementation
function _∂A_impl(proj_ttn::ProjTTN, pos::Tuple{Int,Int}, ::Val{:gpu})
    
    # o1 here has the same link in and out but with
    # different ids, must be getting rewritten or not updated

    o1 = gpu(proj_ttn.local_env)
    projector = contract(o1, dag(prime(o1)))

    function action(T::ITensor)
        T_gpu = gpu(T)
        return proj_ttn.weight * noprime(contract(T_gpu, projector))        
    end
end

# still not working
function recalc_expander_path_flows!(vecproj::VecProj_GPU, ttn::TreeTensorNetwork, oldroot::Tuple{Int,Int}, newroot::Tuple{Int,Int}; use_gpu::Bool = false, node_cache = Dict())
    #error("recalc_expander_path_flows! is not yet implemented for VecProj_GPU")
    
    oc_projtpo = ortho_center(vecproj)
    oc_ttn     = ortho_center(ttn)
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    # first, recalc path flows for the ProjTPO_GPU
    recalc_expander_path_flows!(vecproj.data[1], ttn, oldroot, newroot; use_gpu = use_gpu, node_cache = node_cache)

    @assert oc_projtpo == oldroot "Expected ProjTPO ortho_center $(oc_projtpo) to match oldroot $(oldroot)"

    # then, recalc path flows for the ProjTTNs moving oldroot to newroot
    pth_forward = connecting_path(network(ttn), oldroot, newroot)
    if !isnothing(pth_forward)
        pth = vcat(oldroot, pth_forward)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end
    # then, recalc path flows for the ProjTTNs moving newroot back to oldroot
    pth_back = connecting_path(network(ttn), newroot, oldroot)
    if !isnothing(pth_back)
        pth = vcat(newroot, pth_back)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end

    return vecproj
end
