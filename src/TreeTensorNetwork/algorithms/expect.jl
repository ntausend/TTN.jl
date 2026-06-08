# expectation value of a onsite operator i.e.: <n_j>

"""
```julia
   expect(ttn::TreeTensorNetwork, op)
```
Returns the expectation value of the local observable `op` evaluated at every position of the lattice.
"""
function ITensorMPS.expect(ttn::TreeTensorNetwork, op)
    physlat = physical_lattice(network(ttn))
    res = map(eachindex(physlat)) do (pos)
        expect(ttn, op, pos)
    end
    dims = size(physlat)
    return reshape(res, dims)
end
"""
```julia
   expect(ttn::TreeTensorNetwork, op, pos:Ntuple)
```
Returns the expectation value of the local observable `op` evaluated at the position `pos` given as the d-dimensional coordinate.
"""
function ITensorMPS.expect(ttn::TreeTensorNetwork, op, pos::NTuple)
    return expect(ttn, op, linear_ind(physical_lattice(network(ttn)),pos))
end

"""
```julia
   expect(ttn::TreeTensorNetwork, op, pos:Int)
```
Returns the expectation value of the local observable `op` evaluated at the position `pos` given as the linearized coordinate.
"""
function ITensorMPS.expect(ttn::TreeTensorNetwork, _op, pos::Int)

    net = network(ttn)
    ttnc = copy(ttn)

    idx = siteinds(net)[pos]
    O   = convert_cu(op(_op, idx), ttn[(1,1)])

    # linear position in the D-dimensional lattice
    ch_pos = (0,pos)
    # finding parent node position

    parent_pos = parent_node(net, ch_pos)

    # move ortho_center to parent pos
    move_ortho!(ttnc, parent_pos)

    # find the index of the child
    T = ttnc[parent_pos]
    # perform the contraction
    res = dot(T, noprime(O*T))

    return res
end


function ITensorMPS.expect(all_ttns::Vector, op)
    physlat = physical_lattice(network(all_ttns[1]))
    res = map(eachindex(physlat)) do (pos)
        expect(all_ttns, op, pos)
    end
    dims = size(physlat)
    return reshape(res, dims)
end

function ITensorMPS.expect(all_ttns::Vector, op, pos::NTuple)
    return expect(all_ttns, op, linear_ind(physical_lattice(network(all_ttns[1])),pos))
end

#=function ITensorMPS.expect(all_ttns::Vector{TreeTensorNetwork{N, T}}, _op, pos::Int) where{N<:BinaryNetwork, T}

    net = network(all_ttns[1])

    idx = siteinds(net)[pos]
    O   = convert_cu(op(_op, idx), all_ttns[1][(1,1)])
    ch_pos = (0,pos)
    parent_pos = parent_node(net, ch_pos)

    res_mat = zeros(ComplexF64, length(all_ttns), length(all_ttns))

    for i in 1:length(all_ttns)
        for j in 1:length(all_ttns)
            println("Calculating expectation value for TTN pair ($i, $j)")
            elT = promote_type(eltype(all_ttns[i]), eltype(all_ttns[j]))
            
            if !(sectortype(net) == Int64)
                fl1 = flux(all_ttns[i][number_of_layers(net), 1])
                fl2 = flux(all_ttns[j][number_of_layers(net), 1])
                fl1 == fl2 || return zero(elT)
            end

            
            phys_lat = physical_lattice(net)
            res = map(phys_lat) do nd
                delta(dag(hilbertspace(nd)), prime(hilbertspace(nd)))
            end


            for ll in eachlayer(net)
                nt = number_of_tensors(net,ll)
                res_new = Vector{T}(undef, nt)
                for pp in eachindex(net, ll)
                    if (ll,pp) == parent_pos
                        childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
                        tn1 = all_ttns[i][ll,pp]
                        tn2 = all_ttns[j][ll,pp]
                        rpre1 = res[childs_idx[1]]
                        rpre2 = res[childs_idx[2]]
                        res_new[pp] = dag(prime(tn1)) * (((dag(O) * rpre1) * tn2) * rpre2)
                    else
                        childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
                        tn1 = all_ttns[i][ll,pp]
                        tn2 = all_ttns[j][ll,pp]
                        rpre1 = res[childs_idx[1]]
                        rpre2 = res[childs_idx[2]]
                        res_new[pp] = dag(prime(tn1))*((tn2 * rpre1) * rpre2)
                    end
                end
                res = res_new
            end

            length(res) == 1 || error("Tree Tensor Contraction don't lead to a single resulting tensor.")
            res = res[1]

            display(res)

            #sres = tr(res)

            sres = ITensors.scalar(res)

            res_mat[i,j] = sres
        end
    end


    return eigen(res_mat)
end=#

function ITensorMPS.expect(all_ttns::Vector{TreeTensorNetwork{N, T}}, _op, pos::Int) where {N<:BinaryNetwork, T}
    net = network(all_ttns[1])
    idx = siteinds(net)[pos]
    O   = convert_cu(op(_op, idx), all_ttns[1][(1,1)])
    res_mat = zeros(ComplexF64, length(all_ttns), length(all_ttns))

    for i in 1:length(all_ttns)
        for j in 1:length(all_ttns)
            elT = promote_type(eltype(all_ttns[i]), eltype(all_ttns[j]))

            if !(sectortype(net) == Int64)
                fl1 = flux(all_ttns[i][number_of_layers(net), 1])
                fl2 = flux(all_ttns[j][number_of_layers(net), 1])
                if fl1 != fl2
                    res_mat[i,j] = zero(elT)
                    continue
                end
            end

            phys_lat = physical_lattice(net)
            res = map(enumerate(phys_lat)) do (k, nd)
                s = hilbertspace(nd)
                k == pos ? O : delta(dag(s), prime(s))
            end

            for ll in eachlayer(net)
                nt = number_of_tensors(net, ll)
                res_new = Vector{T}(undef, nt)
                for pp in eachindex(net, ll)
                    childs_idx = getindex.(child_nodes(net, (ll,pp)), 2)
                    tn1 = all_ttns[i][ll,pp]   # bra (network i)
                    tn2 = all_ttns[j][ll,pp]   # ket (network j)
                    rpre1 = res[childs_idx[1]]
                    rpre2 = res[childs_idx[2]]
                    braT  = dag(prime(tn1))
                    res_new[pp] = braT * ((tn2 * rpre1) * rpre2)
                end
                res = res_new
            end

            length(res) == 1 || error("Tree Tensor Contraction doesn't lead to a single resulting tensor.")
            res = res[1]
            #@show i, j, inds(res)
            res_mat[i,j] = order(res) == 0 ? scalar(res) : tr(res)
        end
    end

    return eigen(res_mat)
end