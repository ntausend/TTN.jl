# --- Helpers --------------------------------------------------------------
# import TTN: SimpleLattice, coordinate, linear_ind, inverse_mapping, nearest_neighbours
# Normalize an input "site spec" into a Vector of coordinate tuples
# site_spec can be:
#   - Int                  (linear index)
#   - NTuple{D,Int}        (coordinate)
#   - AbstractVector{Int}  (many linear indices)
#   - AbstractVector{<:NTuple} (many coordinates)
# Returns ::Vector{NTuple{D,Int}}
function _normalize_sites(lat::SimpleLattice, site_spec)
    if site_spec isa Int
        return [coordinate(lat, site_spec)]
    elseif site_spec isa NTuple
        return [site_spec]
    elseif site_spec isa AbstractVector{Int}
        return map(i -> coordinate(lat, i), site_spec)
    elseif site_spec isa AbstractVector{<:NTuple}
        return collect(site_spec)
    else
        throw(ArgumentError("Unsupported site specification: $(typeof(site_spec))"))
    end
end

# Enumerate NN coordinates (both directions) for a single coordinate `pos`
function _nn_coords(lat::SimpleLattice, pos::NTuple{N,Int}, periodic::Bool) where {N}
    nbrs = NTuple{N,Int}[]
    for d in 1:N
        for s in (-1, +1)
            trial = ntuple(i -> i == d ? pos[i] + s : pos[i], N)
            # wrap or skip at boundaries
            if periodic
                wrapped = ntuple(i -> mod(trial[i]-1, lat.dims[i]) + 1, N)
                push!(nbrs, wrapped)
            else
                all(1 .<= trial .<= lat.dims) && push!(nbrs, trial)
            end
        end
    end
    return nbrs
end

# Map a coordinate to external 1D index via inverse mapping
@inline function _to_mapped(lat::SimpleLattice, mapping_lin_to_mapped::Vector{Int}, coord::NTuple)
    return mapping_lin_to_mapped[linear_ind(lat, coord)]
end

# --- Per-site API ---------------------------------------------------------

"""
    nearest_neighbours(lat::SimpleLattice, mapping::Vector{Int}, site;
                       periodic=false) -> Vector{Int}

Return the mapped 1D nearest neighbours of a **single** `site`.
`site` can be a linear index or a coordinate tuple.
"""
function nearest_neighbours(lat::SimpleLattice, mapping::Vector{Int}, site;
                            periodic::Bool=false)
    mapping_lin_to_mapped = inverse_mapping(mapping)
    coord = only(_normalize_sites(lat, site))
    nn = _nn_coords(lat, coord, periodic)
    println("test")
    return map(c -> _to_mapped(lat, mapping_lin_to_mapped, c), nn)
end

"""
    nearest_neighbours(lat::SimpleLattice, mapping::Vector{Int}, sites::Union{Vector{Int},Vector{<:NTuple}};
                       periodic=false) -> Dict{Int,Vector{Int}}

Return a dictionary `mapped_site => Vector{mapped_neighbours}` for an arbitrary
subset `sites`. Elements of `sites` can be linear indices or coordinate tuples.
"""
function nearest_neighbours(lat::SimpleLattice, mapping::Vector{Int}, sites::AbstractVector;
                            periodic::Bool=false)
    mapping_lin_to_mapped = inverse_mapping(mapping)
    coords = _normalize_sites(lat, sites)

    out = Dict{Int, Vector{Int}}()
    for c in coords
        here = _to_mapped(lat, mapping_lin_to_mapped, c)
        nbrs = _nn_coords(lat, c, periodic)
        out[here] = map(nc -> _to_mapped(lat, mapping_lin_to_mapped, nc), nbrs)
    end
    return out
end
