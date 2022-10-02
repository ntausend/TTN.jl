struct BinarySquare <: AbstractLattice{2}
    lat::Vector{Node}
    dims::Tuple{Int, Int}

    function BinarySquare(lin_number_of_sites::Int64, local_dim::Int64; field = ComplexSpace)
        n_layer = 0
        try
            n_layer = Int64(log2(lin_number_of_sites))*2
        catch
            error("Linear Number of Sites $lin_number_of_sites is not compatible with a binary network of n 
                layers requireing lin_number_of_sites = 2^(2n)")
        end
        
        lat_vec = Vector{Node}(undef, lin_number_of_sites^2)
        for xx in 1:lin_number_of_sites, yy in 1:lin_number_of_sites
            linind = xx + (yy - 1)*lin_number_of_sites
            lat_vec[linind] = Node(linind,  field(local_dim),"$xx $yy")
        end
        #lat_vec = [Node(n, "$n") for n in 1:number_of_sites]
        return new(lat_vec, (lin_number_of_sites, lin_number_of_sites))
    end
end

# depreicated as soon as general function is implemented
function to_coordinate(lat::BinarySquare, p::Int)
    n_lin = size(lat,1)
    return (mod1(p, n_lin), div(p - 1,n_lin) + 1)
end

# should be the same formular as the Binary chain? Since we
# are parrying along the x direction
# Therefore (2x - 1, y) x (2x, y) -> (l=1, x,y)
# and since (2x-1, y) -> 2x-1 + y*n_lin =  
#           (2x, y)   -> 2x + (2y)
# are adjacent in the linear linear index
function parentNode(lat::BinarySquare, p::Int64)
    @assert 0 < p ≤ number_of_sites(lat)
    
    #xx, yy = to_coordinate(lat)(p)
    #px = (xx + 1) ÷ 2
    #pos_new = to_linear_ind(lat)((px + (yy-1) * linear_length(length)÷2))
   
    return (1, div((p+1),2))
end

# is the same as for Binary Lattice due to our choice of first pairing
function adjacencyMatrix(la::BinarySquare)
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