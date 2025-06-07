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

struct Link
  layer::Int      # Layer where the link resides
  node::Int       # Index of the node at that layer
  leg::Int8       # Which leg: 0=parent/top, 1=child1/left, 2=child2/right
end


# ───────────────────────────────────────────────
# Low-level definition of a custom operator type
# ───────────────────────────────────────────────

"""
    Op

Represents a one- or multi-site operator with metadata and a precomputed LCAmap.

Fields:
- `id`        : Unique identifier to associate parts of a multi-site operator
- `sites`     : Tuple of site indices the operator acts on (site1, site2, ...)
- `ops`       : Vector of ITensors, one per site
- `lca_map`   : Dict{OC => (lca_node, lca_link)} where
               OC :: Tuple{Int,Int}
               lca_node :: Tuple{Int,Int}
               lca_link :: Tuple{Int,Int} # 0 = parent, 1 = child1, 2 = child2
"""
struct OpGroup
  id::Int
  sites::NTuple{N,Tuple{Int,Int}} where N
  ops::NTuple{N,ITensor} where N
end

length(op::OpGroup) = length(op.sites)

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
lastindex(tpo::TPO_group) = length(tpo.terms)

# IndexStyle(::Type{TPO_group}) = IndexLinear()

# ─────────────────────────────────────────────
# Top-level: Projected TPO structure (ProjTPO)
# ─────────────────────────────────────────────

"""
    ProjTPO

Encapsulates the TTN network, current OC, and link operators needed for sweeping.

Fields:
- `net`        : Tensor network (TTN)
- `tpo`        : Tree product operator
- `oc`         : Tuple{layer, node} denoting the orthogonality center
- `link_ops`   : Dict{(layer, node, leg) => Vector{ITensor}} with link-operators
- `lca_map`    : Dict{(site1, site2) => Dict{(layer, node) => (lca_layer, lca_node, lca_links)}} mapping
                each site to its rerooted lowest common ancestor (LCA) under the current OC.
"""
struct ProjTPO_group
    net::AbstractNetwork
    tpo::TPO
    oc::Tuple{Int,Int}
    link_ops::Dict{Link, Vector{ITensor}}
    lca_map::Dict{Int, Dict{Tuple{Int,Int}, LCA}}
end


# Collect all terms acting on a specific site
## Extend for ProjTPO?
function filter_site_terms(tpo::TPO_group, target_site::Tuple{Int,Int})
    # Filter groups where the target site is in the sites of the group
    return filter(g -> target_site in g.sites, tpo.terms)
end

function build_tpo_from_opsum(ampo::OpSum, lat)
        physidx = siteinds(lat)	
    op_s = OpGroup[]
    terms = filter(t -> !isapprox(coefficient(t),0), ITensorMPS.terms(ITensorMPS.sortmergeterms(ampo)))
    tpo = Vector{OpGroup}(undef, length(terms))

    op_id = 1
    for (i, term) in enumerate(terms)
        coeff = coefficient(term)

        # Build ITensors and extract site indices
        site_indices = Int[]
        operator_tensors = ITensor[]
        # test_tensor = Vector{ITensor}(undef,2)

        for (j, op) in enumerate(ITensors.terms(term))
            opname = ITensors.which_op(op)
            siteidx = ITensors.site(op)
            # Convert site index to linear index if necessary
            siteidx isa Int64 && (siteidx = Tuple(siteidx))
            siteidx_lin = linear_ind(lat, siteidx)

            # Create the ITensor for this operator
            op_t = coeff*ITensors.op(physidx[siteidx_lin], opname; ITensors.params(op)...)

            # Store the operator tensor and site index
            push!(operator_tensors, op_t)
            push!(site_indices, siteidx_lin)
        end
        # Create the OpGroup for this term
        site_indices = tuple(((0, x) for x in site_indices)...)
        operator_tensors = tuple(operator_tensors...)
        push!(op_s, OpGroup(op_id, site_indices, operator_tensors))
        op_id += 1
    end

    return TPO_group(op_s)
end