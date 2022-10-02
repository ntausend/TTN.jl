struct BinaryChain <: AbstractLattice{1}
    lat::Vector{Node}
    dims::Tuple{Int}

    function BinaryChain(number_of_sites::Int, local_dim::Int; field = ComplexSpace)
        n_layer = 0
        try
            n_layer = Int64(log2(number_of_sites))
        catch
            error("Number of Sites $number_of_sites is not compatible with a binary network of n 
                layers requireing number_of_sites = 2^n")
        end
        lat_vec = [Node(n, field(local_dim), "$n") for n in 1:number_of_sites]
        return new(lat_vec, (number_of_sites,))
    end
end

function parentNode(lat::BinaryChain, p::Int)
    @assert 0 < p ≤ number_of_sites(lat)
    return (1, (p + 1) ÷ 2)
end

function adjacencyMatrix(la::BinaryChain)
    n_sites = number_of_sites(la)
    n_layer = 0
    try
        n_layer = Int64(log2(n_sites))
    catch
        error("Number of Sites $number_of_sites is not compatible with a binary network of n 
              layers requireing number_of_sites = 2^n")
    end
    n_first_layer = 2^(n_layer-1)
    pos_phys = collect(1:n_sites)
    I = repeat(collect(1:n_first_layer), 2)
    J = vcat(pos_phys[1:2:end], pos_phys[2:2:end])
 
	return sparse(I,J,repeat([1], n_sites), n_first_layer, n_sites)
end

# depreicated as soon as general function is implemented
to_coordinate(::BinaryChain, p::Int) = (p,)

function Base.show(io::IO, la::BinaryChain)
    for nd in la
        print(io,"|")
        print(io,nd)
        print(io,"|")
        nd.s<length(la) &&  print(io," - ")
    end
    println(io,"")
    for nd in la
        print(io," |    ")
    end
    println(io,"")
end



import Base: ==
function ==(la1::BinaryChain, la2::BinaryChain)
    are_equal = local_dim(la1) == local_dim(la2)
    are_equal = are_equal && (number_of_sites(la1) == number_of_sites(la2))
    return are_equal
end