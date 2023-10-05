"""
    nearest_neighbours(lat::SimpleLattice)
constructs an iterator over all pairs of nearest neighbours, given by a tuple of their respective 1D-positions according to mapping
"""

function nearest_neighbours(lat::SimpleLattice, mapping::Vector{Int}; periodic::Bool = false)
    prod_it = Iterators.product(UnitRange.(1, lat.dims)...)
    mapping = inverse_mapping(mapping)

    iter = map(prod_it) do pos
      map(enumerate(lat.dims)) do (dir,dim)
        (!periodic && pos[dir] >= dim) && return

        unit_vec = zeros(Int64, dimensionality(lat))
        unit_vec[dir] = 1
        nextpos = map(zip(pos .+ unit_vec, lat.dims)) do (pp,d)
          return mod(pp-1, d)+1 
        end

        return (mapping[linear_ind(lat, pos)], mapping[linear_ind(lat, Tuple(nextpos))]) 
      end
    end

    return Vector{Tuple{Vararg{Int}}}(filter(!isnothing, vcat(vec(iter)...))) 
end

function inverse_mapping(mapping::Vector{Int})
    inverse = sort(collect(zip(mapping, 1:length(mapping))), by = x -> x[1])
    return Int[x[2] for x in inverse]
end

function lattice_sites(lat::SimpleLattice)
    prod_it = Iterators.product(UnitRange.(1, lat.dims)...)
    return vec(collect(prod_it))
end

function snake_curve(lat::SimpleLattice)
    return map(lattice_sites(lat)) do pos
        mapreduce(+,enumerate(pos[2:end]), init = pos[1]) do (jj, pp)
            isodd(pp) ? (pp-1)*prod(lat.dims[1:jj]) : lat.dims[jj]-2*pos[jj]+1 + (pp-1)*prod(lat.dims[1:jj])
        end
    end
end

"""
      hilbert_curve(lat::SimpleLattice)
constructs a hilbert_curve mapping of the lattice, given as a vector of the linear indices of the lattice,
works for any lattice shapes in 1D and 2D
"""

function hilbert_curve(lat::SimpleLattice)
    curve = Vector{Tuple{Int, Int}}([])

    if dimensionality(lat) == 1 
        (w,h) = (lat.dims[1],1)
    elseif dimensionality(lat) == 2 
        (w,h) = lat.dims
    else
        error("hilbert curve not implemented for these lattice dimensions")
    end
    generate2d(0, 0, w, 0, 0, h, curve)
    curve_lin = map(p -> TTNKit._linear_ind_simple_lattice(p, lat.dims), curve)

    # need to invert the resulting curve
    # curve_inv = sort(collect(zip(curve_lin, eachindex(lat))), by = x -> x[1])
    #
    # return Vector{Int}([p[2] for p in curve_inv])
    return curve_lin
end

function generate2d(x, y, ax, ay, bx, by, curve::Vector{Tuple{Int, Int}})
    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sign(ax), sign(ay)) # unit major direction
    (dbx, dby) = (sign(bx), sign(by)) # unit orthogonal direction

    if h == 1
        # trivial row fill
        for _ in 1:w
            append!(curve, [(Int(x+1), Int(y+1))])
            (x,y) = (x + dax, y + day)
        end
        return
    end

    if w == 1
        # trivial column fill
        for _ in 1:w
            append!(curve, [(Int(x+1), Int(y+1))])
            (x,y) = (x + dbx, y + dby)
        end
        return
    end

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h
        if isodd(w2) && (w > 2)
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)
        end

        # long case: split in two parts only
        generate2d(x, y, ax2, ay2, bx, by, curve)
        generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by, curve)

    else
        if isodd(h2) && (h > 2)
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)
        end

        # standard case: one step up, one long horizontal, one step down
        generate2d(x, y, bx2, by2, ax2, ay2, curve)
        generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2, curve)
        generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby), -bx2, -by2, -(ax-ax2), -(ay-ay2), curve)
    end
end
