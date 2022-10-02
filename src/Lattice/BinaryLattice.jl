function _to_linear_ind(p::Tuple{D,Int}, dims::Tuple{D,Int}) where{D}
    res = mapreduce(+,enumerate(p[2:end]), init = p[1]) do (jj, pp)
        (pp-1)*prod(dims[1:jj])
    end
    return res
end

struct BinaryLattice{Tuple{D,Int}} <: AbstractLattice{D}
    lat::Vector{Node}
    dims::NTuple{D,Int}
    function BinaryLattice(dims::NTuple{D, Int}, local_dim; field = ComplexSpace) where D
        n_layer = 0
        try
            n_layer = sum(Int64.(log2.(dims)))
        catch
            error("Number of Sites $dims is not compatible with a binary network of n 
                layers requireing pord(number_of_sites) = 2^n")

        end

        #lat_vec = map()
        [Node(_to_linear_ind(dims, ))]
    end
end