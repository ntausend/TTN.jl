
function correlation(ttn::TreeTensorNetwork{D}, op1::TensorMap, op2::TensorMap, 
                     pos1::Union{NTuple{D,Int}, Int}, pos2::Union{NTuple{D,Int}, Int}) where{D}
    net = network(ttn)
    return _correlation(ttn, net, op1, op2, pos1, pos2)
end

function _correlation(ttn::TreeTensorNetwork{D}, net::AbstractNetwork{D}, op1::TensorMap, op2::TensorMap, 
                     pos1::NTuple{D,Int}, pos2::NTuple{D,Int}) where{D}
    pos1_lin = linear_ind(physical_lattice(net), pos1)
    pos2_lin = linear_ind(physical_lattice(net), pos2)
    return _correlation(ttn, net, op1, op2, pos1_lin, pos2_lin)
                        
end

function _correlation(ttn::TreeTensorNetwork{D}, net::AbstractNetwork{D}, op1::TensorMap, op2::TensorMap, 
                     pos1::Int, pos2::Int) where{D}

    ttnc = copy(ttn)

    physlat = physical_lattice(net)
    hilbttn = hilbertspace(node(physlat,1))

    for op in (op1, op2)
        doo1  = domain(op)
        codo1 = codomain(op)
        if !(ProductSpace(hilbttn) == doo1 == codo1)
            error("Codomain and domain of operator $op not matching with local hilbertspace.")
        end
    end

end