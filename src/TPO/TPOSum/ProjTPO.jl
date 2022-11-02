struct ProjTensorProductOperator{D, S<:IndexSpace, I<:Sector}
    net::AbstractNetwork{D, S, I}
    tpo::AbstractTensorProductOperator

    bottom_envs::Vector{Vector{AbstractTensorMap}}
    top_envs::Vector{AbstractTensorMap}
end