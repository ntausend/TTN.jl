struct TensorProductOperatorSum{L<:AbstractLattice} <: AbstractTensorProductOperator{L}
    interactions::Vector{AbstractInteraction}
    lat::L
end
function TensorProductOperatorSum(lat::L) where {L<:AbstractLattice}
    ints = Vector{AbstractInteraction}()
    return TensorProductOperatorSum{L}(ints, lat)
end
interactions(tpo::TensorProductOperatorSum) = tpo.interactions


import Base: +
function add(tpo::TensorProductOperatorSum, int::AbstractInteraction)
    new_int = push!(interactions(tpo), int)
    return TensorProductOperatorSum(new_int, lattice(tpo))
end

+(tpo::TensorProductOperatorSum, int::AbstractInteraction) = add(tpo, int)


import Base: sort
function Base.sort(tpo::TensorProductOperatorSum)
    ints = interactions(tpo)
    new_interactions = sort(ints, lt = (x,y) -> isless(interaction_length(x), interaction_length(y)))
    return TensorProductOperatorSum(new_interactions, lattice(tpo))
end

function Base.show(io::IO, tpo::TensorProductOperatorSum)
    for (jj, int) in enumerate(interactions(tpo))
        msg  = "Interaction $(name(int)) with number $jj "
        msg *= "acting between sites $(interaction_positions(int)) and coupling: $(coupling(int))"
        println(io,msg)
    end
    #println(collect(interactions(tpo)))
end