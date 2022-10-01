struct BinaryChain <: AbstractLattice{1}
    lat::Vector{Node}
    local_dim::Int64
    local_hilbertspace
    function BinaryChain(number_of_sites::Int64, local_dim::Int64; field = ComplexSpace)
        n_layer = 0
        try
            n_layer = Int64(log2(number_of_sites))
        catch
            error("Number of Sites $number_of_sites is not compatible with a binary network of n 
                layers requireing number_of_sites = 2^n")
        end
        lat_vec = [Node(n, "$n") for n in 1:number_of_sites]
        return new(lat_vec, local_dim, field(local_dim))
    end
end

function parentNode(lat::BinaryChain, p::Int64)
    @assert 0 < p ≤ number_of_sites(lat)
    return (1, (p + 1) ÷ 2)
end

function adjacencyMatrix(la::BinaryChain)
    n_sites = number_of_sites(la)
    n_first_layer = 2^(n_layer-1)
    pos_phys = collect(1:n_sites)
    I = repeat(collect(1:n_first_layer), 2)
    J = vcat(pos_phys[1:2:end], pos_this[2:2:end])
 
	return sparse(I,J,repeat([1], n_sites), n_first_layer, n_sites)
end

to_linear_ind(::BinaryChain) = x -> x[1]
to_coordinate(::BinaryChain) = x -> (x,)

function Base.show(io::IO, la::BinaryChain)
    for nd in la
        print(io,"|_|")
        nd.s<length(la) &&  print(io," - ")
    end
    println(io,"")
    for nd in la
        print(io," |    ")
    end
    println(io,"")
    for nd in la
        print(io," ")
        print(io,nd)
        print(io,"    ")
        #nd.s<length(la.lat) &&  print(io," - ")
    end
    println(io, "")
end



import Base: ==
function ==(la1::BinaryChain, la2::BinaryChain)
    are_equal = local_dim(la1) == local_dim(la2)
    are_equal = are_equal && (number_of_sites(la1) == number_of_sites(la2))
    return are_equal
end