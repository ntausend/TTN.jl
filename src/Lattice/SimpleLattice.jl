function _linear_ind_simple_lattice(p::NTuple{D,Int}, dims::NTuple{D,Int}) where{D}
    res = mapreduce(+,enumerate(p[2:end]), init = p[1]) do (jj, pp)
        (pp-1)*prod(dims[1:jj])
    end
    return res
end

function _coordinate_simple_lattice(p::Int, dims::NTuple{D,Int}) where {D}
    p_vec = Vector{Int64}(undef, D)
    p_r = p
    for jj in D:-1:2
        modifier = prod(dims[1:jj-1])
        p_vec[jj] = div(p_r - 1, modifier) + 1
        p_r = mod1(p_r, modifier)
    end
    p_vec[1] = p_r
    return Tuple(p_vec)
end

struct SimpleLattice{D, S, I} <: AbstractLattice{D,S,I}
    lat::Vector{AbstractNode{S, I}}
    dims::NTuple{D,Int}
end
"""
```julia
    SimpleLattice(dims::NTuple{D, Int}, nodetype::Function; kwargs...) where D
```

Construction of a general Regular Lattice of dimension `D`. This Lattice serves as a basis for
the Binary networks, where every layer is represented as a simple lattice of the same dimensionality.
In 1d a simple lattice is just a chain, in 2d a rectangle with simple unit cell etc.
Every dimension should be multiple of 2 in order to be compatible with a Binary Network
(and thus the name `BinaryLattice`).

## Arguments:
- `dims::NTuple{D,Int}`: A `D` dimensional Tuple defining the dimensions of the lattice.
   every entry should be a multiple of 2, i.e. `dims[j] = 2^(n_j)` should be fulfilled for
   every 0≤ j≤ D.
- `nd`: Valid Node type, will be used for factory as `nd(l, n)` where `l` is the linear position
   and `n` is a description string
- In case, nd is of type ITensorNode, a type string defining the local hilbertspace is expcect following
  `nd`
```julia
lat = SimpleLattice((8,), ITensorNode, "SpinHalf"; kwargs...)
```
   would generate a chain filled with 'SpinHalf' degrees of freedom.
"""
function SimpleLattice(dims::NTuple{D, Int}, nd::Type{<:AbstractNode}; kwargs...) where {D}
    if nd isa ITensorNode
        error("Using node type ITensorNode needs specifying the type string for constructing the hilberspaces.")
    end
    # checking if dims are in the correct layout
    # why including this here? any reason not to define the lattice on abitrary dimensions?
    #_check_dimensions(dims)

    prod_it = Iterators.product(UnitRange.(1, dims)...)
    
    nd_names = map(prod_it) do p_ind
        mapreduce(p -> " $p", *, p_ind[2:end], init=string(p_ind[1]))
    end
    lin_inds = map(p -> _linear_ind_simple_lattice(p, dims), prod_it)
    lat_vec = map(zip(lin_inds, nd_names)) do (l, n)
        nd(l, n; kwargs...)
    end
    lat_vec = vec(lat_vec)
    #backend = hilbertspace(lat_vec[1]) isa Index ? ITensorsBackend : TensorKitBackend
    #backend == ITensorsBackend  && (lat_vec = ITensorNode.(lat_vec))
    #_backend = backend(lat_vec[1])
    # now convert the lattice in the case of being physical and ITensors being the backend to ITensorNodes
    #_backend == ITensorsBackend && lat_vec[1] isa PhysicalNode && (lat_vec = ITensorNode.(lat_vec))
    return SimpleLattice{length(dims), spacetype(lat_vec[1]), sectortype(lat_vec[1])}(lat_vec, dims)
end

# Fast factory of a lattice with trivial nodes with hilbertspace dimension local_dim
function SimpleLattice(dims::NTuple{D, Int}, local_dim::Int) where D
    return SimpleLattice(dims, TrivialNode; local_dim = local_dim)
end

# factory from indices, and from ITensor node with type specifier
function SimpleLattice(dims::NTuple{D, Int}, indices::Vector{<:Index}) where{D}
    #_check_dimensions(dims)
    prod_it = Iterators.product(UnitRange.(1, dims)...)
    lin_inds = map(p -> _linear_ind_simple_lattice(p, dims), prod_it)
    @assert length(lin_inds) == length(indices)

    lat_vec = map(zip(lin_inds, indices)) do (p,idx) ITensorNode(p,idx) end
    lat_vec = vec(lat_vec)
    return SimpleLattice{length(dims), spacetype(lat_vec[1]), sectortype(lat_vec[1])}(lat_vec, dims)
end
function SimpleLattice(dims::NTuple{D, Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...) where{D}
    # checking if dims are in the correct layout
    indices = siteinds(type, prod(dims); kwargs...)
    return SimpleLattice(dims, indices)
    #=
    _check_dimensions(dims)
    prod_it = Iterators.product(UnitRange.(1, dims)...)
    lin_inds = map(p -> _linear_ind_simple_lattice(p, dims), prod_it)

    lat_vec = map(l -> nd(l, type; kwargs...), lin_inds)
    lat_vec = vec(lat_vec) # necessary?
    return SimpleLattice{length(dims), spacetype(lat_vec[1]), sectortype(lat_vec[1]), ITensorsBackend}(lat_vec, dims)
    =#
end
SimpleLattice(dims::NTuple{D,Int}, type::AbstractString; kwargs...) where{D} = SimpleLattice(dims, ITensorNode, type; kwargs...)


#const BinaryChain = BinaryLattice{1}
"""
```julia
    Chain(n_sites::Int, type::AbstractString; kwargs...)
```

Creates a chain of length `n_sites`. The Hilbert-space of every node will be given by `type`.

# Arguments

---

- `n_sites`: Length of the chain
- `type`: Definition of the local Hilbert-space

# Keywords

---

The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space

"""
Chain(n_sites::Int, local_dim::Int; kwargs...) = SimpleLattice((n_sites,), local_dim; kwargs...)
Chain(n_sites::Int, nd::Type{<:AbstractNode}, args...; kwargs...) = SimpleLattice((n_sites,), nd, args...; kwargs...)
Chain(n_sites::Int, args...; kwargs...) = SimpleLattice((n_sites,), args...; kwargs...)


"""
```julia
    Rectangle(n_x::Int, n_y::Int, type::AbstractString; kwargs...)
```

Creates a rectangle of dimension `(n_x,n_y)`. The Hilbert-space of every node will be given by `type`.

# Arguments

---

- `n_sites`: Length of the chain
- `type`: Definition of the local Hilbert-space

# Keywords

---

The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space

"""
Rectangle(n_x::Int, n_y::Int, args...; kwargs...) = SimpleLattice((n_x, n_y), args...; kwargs...)
Rectangle(dims::Tuple{Int, Int}, args...; kwargs...) = SimpleLattice(dims, args...; kwargs...)
Square(n_lin::Int, args...; kwargs...) = Rectangle(n_lin, n_lin, args...; kwargs...)


Chain(indices::Vector{<:Index}) = SimpleLattice((length(indices),), indices)
Rectangle(dims::Tuple{Int,Int}, indices::Vector{<:Index}) = SimpleLattice(dims, indices)
Rectangle(n_x::Int, n_y::Int, indices::Vector{<:Index}) = SimpleLattice((n_x,n_y), indices)
Rectangle(indices::Matrix{<:Index}) = Rectangle(size(indices), flatten(indices))

Square(n_lin::Int, indices::Vector{<:Index}) = Rectangle((n_lin, n_lin), indices)

Base.size(lat::SimpleLattice) = lat.dims
Base.size(lat::SimpleLattice, d::Integer) = size(lat)[d]


function linear_ind(lat::SimpleLattice{D}, p::NTuple{D,Int}) where D
    return _linear_ind_simple_lattice(p, size(lat))
end


function coordinate(lat::SimpleLattice, p::Int)
    return _coordinate_simple_lattice(p, size(lat))
end


function ==(lat1::SimpleLattice{D1}, lat2::SimpleLattice{D2}) where{D1, D2}
    D1 == D2 || return false    

    all(size(lat1) .== size(lat2)) || return false
    is_physical(lat1) == is_physical(lat2) || return false

    isph = is_physical(lat1)

    are_equal = mapreduce(*, zip(lat1, lat2), init = true) do (nd1, nd2)
        if(isph)
            return hilbertspace(nd1) == hilbertspace(nd2)
        else
            return sectortype(nd1) == sectortype(nd2) && spacetype(nd1) == spacetype(nd2)
        end
    end
    return are_equal
end


function Base.show(io::IO, lat::SimpleLattice{D}) where{D}
    println(io, "Simple Lattice of dimension $D, with dimensions: $(size(lat)):\n")
    if( D == 1)
        s = "\t"
        lengths = Vector{Int64}(undef, length(lat))
        for (jj,nd) in enumerate(lat)
            #desc = description(nd)
            if nd isa PhysicalNode
                desc = string(hilbertspace(nd))
            else
                desc = string(sectortype(nd))
            end
            if (jj == 1)
                s_coord = "|$(desc)|"
            else
                s_coord = "-|$(desc)|"
            end
            s *= s_coord
            lengths[jj] = length(s_coord)
        end
        print(io, s)
        println(io, "")
        
        s = "\t"
        for jj in 1:length(lat)
            l_half = lengths[jj]÷2
            l_res  = lengths[jj] - l_half
            s_lower = repeat(" ", l_half) * "|" * repeat(" ", l_res-1)
            s *= s_lower
        end
        print(io,s)
        println(io,"")
    elseif (D == 2 )
        lengths = Matrix{Int64}(undef, size(lat))
        for yy in 1:size(lat,2)
            s = "\t"
            for xx in 1:size(lat,1)
                #desc = description(node(lat, to_linear_ind(lat, (xx,yy))))
                nd = node(lat, linear_ind(lat, (xx,yy)))
                if nd isa PhysicalNode
                    desc = string(hilbertspace(nd))
                else
                    desc = string(sectortype(nd))
                end
                if xx == 1
                    s_coord = "|$(desc)|"
                else
                    s_coord = "-|$(desc)|"
                end
                s *= s_coord
                lengths[xx,yy] = length(s_coord)
            end
            print(io, s)
            println(io, "")
            yy == size(lat,2) && break
            s = "\t"
            for jj in 1:size(lat,1)
                l_half = lengths[jj,yy]÷2
                l_res  = lengths[jj,yy] - l_half
                s_lower = repeat(" ", l_half) * "|" * repeat(" ", l_res-1)
                s *= s_lower
            end
            print(io,s)
            println(io,"")
        end
    else
        for nd in lat
            print(io, "\t")
            println(io, nd)
        end
    end
    # just implement a show for one and two dimensional cases..
    #D>2 && println(io, lat)
end
