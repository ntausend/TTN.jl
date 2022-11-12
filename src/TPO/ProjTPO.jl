struct ProjTensorProductOperator{D, S<:IndexSpace, I<:Sector}
    net::AbstractNetwork{D, S, I}
    tpo::AbstractTensorProductOperator

    bottom_envs::Vector{Vector{Vector{AbstractTensorMap}}}
    top_envs::Vector{Vector{AbstractTensorMap}}
end

network(projTPO::ProjTensorProductOperator) = projTPO.net
tensor_product_operator(projTPO::ProjTensorProductOperator) = projTPO.tpo
bottom_environment(projTPO::ProjTensorProductOperator, pos::Tuple{Int,Int}) = projTPO.bottom_envs[pos[1]][pos[2]]
bottom_environment(projTPO::ProjTensorProductOperator, pos::Tuple{Int,Int}, n_child::Int) = bottom_environment(projTPO, pos)[n_child]
top_environment(projTPO::ProjTensorProductOperator, pos::Tuple{Int,Int}) = projTPO.top_envs[pos[1]][pos[2]]


function _build_environments!(projTPO::ProjTensorProductOperator, ttn::TreeTensorNetwork)
    throw(NotImplemented(:_build_environments!, "AbstractTensorProductOperator"))
end


function rebuild_environments!(projTPO::ProjTensorProductOperator, ttn::TreeTensorNetwork)
    net = network(projTPO)
    @assert net == network(ttn)
    return _build_environments!(projTPO,ttn)
end

function update_environments!(projTPO::ProjTensorProductOperator, isom::AbstractTensorMap, pos::Tuple{Int,Int}, dir::Tuple{Int, Int})
    if dir[1] == 1
        @assert dir[2] == 0
        _update_top_environment!(projTPO, isom, pos)
    else
        @assert dir[1] == -1
        _update_bottom_environment!(projTPO, isom, pos, dir[2])
    end
end

# how to do this abstractly for arbitrary networks??
function _update_top_environment!(projTPO::ProjTensorProductOperator, isom::AbstractTensorMap, pos::Tuple{Int,Int})
    error("Not Implemented")
end
function _update_bottom_environment!(projTPO::ProjTensorProductOperator, isom::AbstractTensorMap, pos::Tuple{Int,Int}, n_child::Int)
    error("Not Implemented")
end

# action of the TPO on the removed A tensor at position p
# how to define this in the most abstract way for arbitrary networks???
function ∂A(projTPO::ProjTensorProductOperator, pos::Tuple{Int,Int})
    return ∂A(projTPO, network(projTPO), pos)
end

# special case of simple binary network... this is easy
function ∂A(projTPO::ProjTensorProductOperator, net::BinaryNetwork, pos::Tuple{Int,Int})
    top_env = top_environment(projTPO, pos)
    chld_envs = bottom_environment(projTPO, pos)
    function action(T::AbstractTensorMap)
        # do stuff here -> need paper for considering the contractions
        #@tensor 
    end
end