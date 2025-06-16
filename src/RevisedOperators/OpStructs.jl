const NEXT_OPGROUP_ID = Ref{Int}(0)

"""
new_opgroup_id()
— return a fresh unique id; increments the counter.
"""
new_opgroup_id() = begin
  id = NEXT_OPGROUP_ID[]
  NEXT_OPGROUP_ID[] += 1
  return id
end

"""
    LCA

Stores the rerooted lowest common ancestor for a pair of sites under a given OC,
along with the legs of the LCA node that each site connects through.

Fields:
- `lca_node`  : Tuple{Int,Int} — (layer, node) of the LCA
- `legs`      : Tuple{Int,Int} — each in {0, 1, 2} = (parent, child1, child2)
"""
struct LCA
  lca_node::Tuple{Int,Int}
  legs::Tuple{Int,Int}
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
- `sites`     : Tuple of site indices the operator acts on (site1, site2, ...)
- `ops`       : Tuple of ITensors, one per site
"""
struct OpGroup
  id::Int
  site::Tuple{Int,Int}
  op::ITensor
  length::Int 
end

length(op::OpGroup) = op.length
# OpGroup(id::Int, site::Tuple{Int,Int}, op::ITensor) = OpGroup(id, (site,), (op,))


# ─────────────────────────────────────────────
# Mid-level: Tree tensor product operator (TPO)
# ─────────────────────────────────────────────

"""
    TPO

Stores a list of custom `Op` terms derived from the original `OpSum`.

Fields:
- `terms` :: Vector{Op}
"""
struct TPO_group
    terms::Vector{OpGroup}
end


getindex(tpo::TPO_group, i::Int) = tpo.terms[i]
length(tpo::TPO_group) = length(tpo.terms)
iterate(tpo::TPO_group, state=1) = state > length(tpo) ? nothing : (tpo[state], state+1)
firstindex(tpo::TPO_group) = 1
lastindex(tpo::TPO_group) = length(tpo)

"""
init_opgroup_id_counter!(tpo::Vector{OpGroup})
— after you build your TPO (and have assigned its original ids),
  call this to seed the counter one above the max you’ve used.
"""

function init_opgroup_id_counter!(tpo::Vector{OpGroup})
  if isempty(tpo)
    NEXT_OPGROUP_ID[] = 1
  else
    NEXT_OPGROUP_ID[] = maximum(g.id for g in tpo) + 1
  end
  return NEXT_OPGROUP_ID[]
end


# ─────────────────────────────────────────────
# Top-level: Projected TPO structure (ProjTPO)
# ─────────────────────────────────────────────

"""
    ProjTPO

Encapsulates the TTN network, current OC, and link operators needed for sweeping.

Fields:
- `net`        : tree network structure, e.g. `BinaryNetwork`
- `tpo`        : Tree product operator
- `oc`         : Tuple{layer, node} denoting the orthogonality center
- `link_ops`   : Dict{((layer, node), leg) => Vector{ITensor}} with link-operators
- `lca_map`    : Dict{(site1, site2) => Dict{(layer, node) => (lca_layer, lca_node, lca_links)}} mapping
                each site to its rerooted lowest common ancestor (LCA) under the current OC.
"""
struct ProjTPO_group
    net::AbstractNetwork
    tpo::TPO_group
    oc::Tuple{Int,Int}
    link_ops::Dict{Tuple{Tuple{Int,Int},Int}, Vector{OpGroup}} # ((layer, node), leg) => Vector of OpGroups
    lca_map::Dict{Int, Dict{Tuple{Int,Int}, LCA}}
end

# ─────────────────────────────────────────────
# Helper function to build a TPO from an OpSum 
# ─────────────────────────────────────────────

"""
    build_tpo_from_opsum(ampo::OpSum, lat)
Builds a `TPO_group` from an `OpSum` by extracting terms and converting them to `OpGroup`.

`ampo` is an `OpSum` containing the operator terms, and `lat` is the lattice structure.

This function constructs `OpGroup` instances for each term,
and returns a `TPO_group` containing all the operator groups.
N-site operators are splitted into individual `OpGroup` instances matched by their unique ID.
"""

function build_tpo_from_opsum(ampo::OpSum, lat::AbstractLattice)
    physidx = siteinds(lat)
    op_s    = OpGroup[]
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

            push!(op_s, OpGroup(op_id, (0, siteidx_lin), op_t, len))
        end
        op_id += 1
    end

    init_opgroup_id_counter!(op_s)
    return TPO_group(op_s)
end


# Find all operator groups by their length
function get_length_terms(tpo::TPO_group, len::Int)
    return filter(op -> op.length == len, tpo.terms)
end

function get_length_terms(group::Vector{OpGroup}, len::Int)
    filter(g -> g.length == len, group)
end

function get_length_terms(net::AbstractNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, (layer, node)::Tuple{Int,Int}, len::Int)
    # gather all OpGroup objects by length len
    # of (layer,node) and on the link to its parentS
    bucket = Dict{Int, Vector{OpGroup}}()

    # the three links we care about
    links = (
        ((layer, node), 1),                          # first-child link
        ((layer, node), 2),                          # second-child link
        (parent_node(net, (layer, node)),
         which_child(net, (layer, node)))            # parent link
    )

    for link in links
        # get(link_ops, link, Vector{OpGroup}()) → empty vector if link not present
        for op in get(link_ops, link, Vector{OpGroup}())
            if op.length == len
                push!(get!(bucket, op.length, Vector{OpGroup}()), op)
            end
        end
    end

    return bucket
end

# Collect all terms acting on a specific site
function get_site_terms(tpo::TPO_group, target_site::Tuple{Int,Int})
    # Filter groups where the target site is in the sites of the group
    return filter(op -> op.site == target_site, tpo.terms)
end

# Find all operator groups by their unique ID
function get_id_terms(tpo::TPO_group, target_id::Int)
    return filter(op -> op.id == target_id, tpo.terms)
end

"""
    filter_id_term(link_ops, (layer, node), id) -> Vector{OpGroup}

Return all `OpGroup`s whose `op.id == id` that act on the two child links
of `(layer, node)` or the link to its parent.  If no such operators are
present, an empty vector is returned.
"""
function get_id_terms(net::AbstractNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, (layer, node)::Tuple{Int,Int}, id)
    # collect every OpGroup on the three relevant links,
    # then pull out the bucket for the requested id
    get(get_id_terms(net::AbstractNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, (layer, node)), id, Vector{OpGroup}())
end

function get_id_terms(net::AbstractNetwork, link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{OpGroup}}, (layer, node)::Tuple{Int,Int})
    # gather all OpGroup objects whose id appears on the two child links
    # of (layer,node) and on the link to its parent
    bucket = Dict{Int, Vector{OpGroup}}()
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
        # get(link_ops, link, Vector{OpGroup}()) → empty vector if link not present
        for op in get(link_ops, link, Vector{OpGroup}())
            push!(get!(bucket, op.id, Vector{OpGroup}()), op)
        end
    end

    return bucket
end

# which_child(net, parent, child) = findfirst(==(child), child_nodes(net, parent))
which_child(net::AbstractNetwork, child::Tuple{Int,Int}) = findfirst(==(child), child_nodes(net, parent_node(net, child)))

## Find the index of a child node in the parent's child list
"""
    which_child(net, parent, child)
"""
#=
function which_child(net, parent, child)
    children = child_nodes(net, parent)
    for (i, c) in enumerate(children)
        if c == child
            return i
        end
    end
    error("Node $child is not a child of $parent")
end
=#
