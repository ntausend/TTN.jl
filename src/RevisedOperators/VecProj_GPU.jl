mutable struct VecProj_GPU{N<:AbstractNetwork, T, P<:Tuple{Vararg{AbstractProjTPO{N, T}}}} <: AbstractProjTPO{N,T}
    net::N
    data::P
    ortho_center::Tuple{Int64,Int64}
end

function VecProj_GPU(all_projs::Tuple)
    ortho_center = all_projs[end].ortho_center

    for proj in all_projs
        @assert proj.ortho_center == ortho_center "All ProjTPOs in VecProj_GPU must have the same orthogonality center: found $(proj.ortho_center) and $ortho_center for $(typeof(proj))"
    end

    return VecProj_GPU(network(all_projs[1]), all_projs, ortho_center)
end

function VecProj_GPU(all_projs::Tuple, ttn::TreeTensorNetwork, target_oc::Tuple{Int,Int}; use_gpu::Bool = false, node_cache = Dict())
    # Move all projectors to the target orthogonality center
    updated_projs = map(all_projs) do proj
        if proj isa ProjTPO_GPU
            # For ProjTPO_GPU, use recalc_path_flows!
            current_oc = proj.ortho_center
            if current_oc != target_oc
                recalc_path_flows!(proj, ttn, current_oc, target_oc; use_gpu = use_gpu)
            end
            return proj
        elseif proj isa ProjTTN
            # For ProjTTN, use set_position! from AbstractProjectedTensorProductOperator
            current_oc = Tuple(proj.ortho_center)
            if current_oc != target_oc
                pth = connecting_path(network(ttn), current_oc, target_oc)
                if !isnothing(pth)
                    pth = vcat(current_oc, pth)
                    for (jj, pk) in enumerate(pth[1:end-1])
                        ism = ttn[pk]
                        update_environments!(proj, ism, pk, pth[jj+1])
                    end
                    proj.ortho_center .= target_oc
                end
            end
            return proj
        else
            error("Unsupported projector type: $(typeof(proj))")
        end
    end

    return VecProj_GPU(network(all_projs[1]), updated_projs, target_oc)
end

function set_position!(vecproj::VecProj_GPU{N,T}, ttn::TreeTensorNetwork{N,T}; use_gpu::Bool = false, node_cache = Dict()) where {N,T}
    oc_projtpo = ortho_center(vecproj)
    oc_ttn     = ortho_center(ttn)
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    all(oc_projtpo .== oc_ttn) && return vecproj

    # set position for ProjTPO_GPU
    recalc_path_flows!(vecproj.data[1], ttn, oc_projtpo, oc_ttn; use_gpu = use_gpu, node_cache = node_cache)

    # set position for ProjTTNs
    pth = connecting_path(network(ttn), oc_projtpo, oc_ttn)
    if !isnothing(pth)
        pth = vcat(oc_projtpo, pth)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end

    vecproj.ortho_center = oc_ttn
    return vecproj
end

function recalc_path_flows!(vecproj::VecProj_GPU, ttn::TreeTensorNetwork, oldroot::Tuple{Int,Int}, newroot::Tuple{Int,Int}; use_gpu::Bool = false, node_cache = Dict())
    
    oc_projtpo = ortho_center(vecproj)
    oc_ttn     = ortho_center(ttn)
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    # first, recalc path flows for the ProjTPO_GPU
    recalc_path_flows!(vecproj.data[1], ttn, oldroot, newroot; use_gpu = use_gpu, node_cache = node_cache)

    @assert oc_projtpo == oldroot "Expected ProjTPO ortho_center $(oc_projtpo) to match oldroot $(oldroot)"

    # then, recalc path flows for the ProjTTNs moving oldroot to newroot
    pth_forward = connecting_path(network(ttn), oldroot, newroot)
    if !isnothing(pth_forward)
        pth = vcat(oldroot, pth_forward)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end

    vecproj.ortho_center = newroot

    return vecproj
end

# highest-level dispatch for partial A on VecProj_GPU, dispatches to CPU or GPU implementation
function ∂A_GPU(ptpo::VecProj_GPU, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A_impl(ptpo, pos, Val(:gpu)) : _∂A_impl(ptpo, pos, Val(:cpu))
end

# lowest-level CPU implementation of partial A for VecProj_GPU
function _∂A_GPU(proj_operator::VecProj_GPU, pos::Tuple{Int,Int}; use_gpu::Bool = false)
   
    action_vec = map(ptpo -> ∂A_GPU(ptpo, pos; use_gpu = false), proj_operator.data)

    function action(T::ITensor)
        return mapreduce(+, action_vec) do act
            return act(T)
        end
    end
end

# maintains Noah's shape of partial A and dispatches to my CPU partial A implementation
function _∂A_impl(ptpo::VecProj_GPU, pos::Tuple{Int,Int}, ::Val{:cpu})
    return _∂A_GPU(ptpo, pos; use_gpu = false)
end

# maintains Noah's shape of partial A and executes GPU implementation
function _∂A_impl(ptpo::VecProj_GPU, pos::Tuple{Int,Int}, ::Val{:gpu})

    action_vec = map(ptpo -> ∂A_GPU(ptpo, pos; use_gpu = true), ptpo.data)

    function action(T::ITensor)
        
        T_gpu = gpu(T)

        return mapreduce(+, action_vec) do act
            return act(T_gpu)
        end
    end
end

# highest level catch for partial A, dispatches to CPU or GPU
function ∂A_GPU(proj_ttn::ProjTTN, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A_impl(proj_ttn, pos, Val(:gpu)) : _∂A_impl(proj_ttn, pos, Val(:cpu))
end

# lowest-level CPU implementation of ∂A for ProjTTN
function _∂A_GPU(proj_ttn::ProjTTN, pos::Tuple{Int,Int}; use_gpu::Bool = false)

    function action(T::ITensor)
        projector = contract(proj_ttn.local_env, dag(prime(proj_ttn.local_env)))
        return proj_ttn.weight * noprime(contract(T,projector))
    end
end

# maintains Noah's shape of partial A and dispatches to my CPU partial A implementation
function _∂A_impl(proj_ttn::ProjTTN, pos::Tuple{Int,Int}, ::Val{:cpu})
    return _∂A_GPU(proj_ttn, pos; use_gpu = false)
end

# maintains Noah's shape of partial A and executes GPU implementation
function _∂A_impl(proj_ttn::ProjTTN, pos::Tuple{Int,Int}, ::Val{:gpu})
    
    # o1 here has the same link in and out but with
    # different ids, must be getting rewritten or not updated


    function action(T::ITensor)

        #=println("In GPU implementation of ∂A for ProjTTN")
        all_inds = inds(proj_ttn.local_env)
        all_dims = dim.(all_inds)
        println("Length of all_inds: $(length(all_inds))")
        println("Dimensions of all_inds: $(all_dims)")
        println("Final Bond Dim: $(prod(all_dims))")=#
        o1 = gpu(proj_ttn.local_env)
        #projector = contract(o1, dag(prime(o1)))
        T_gpu = gpu(T)
        #return proj_ttn.weight * noprime(contract(T_gpu, projector))        
        return proj_ttn.weight * noprime(contract(contract(T_gpu, o1),dag(prime(o1))))
    end
end

function recalc_expander_path_flows!(vecproj::VecProj_GPU, ttn::TreeTensorNetwork, oldroot::Tuple{Int,Int}, newroot::Tuple{Int,Int}; use_gpu::Bool = false, node_cache = Dict())
    
    oc_projtpo = ortho_center(vecproj)
    oc_ttn     = ortho_center(ttn)
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    # first, recalc path flows for the ProjTPO_GPU
    recalc_expander_path_flows!(vecproj.data[1], ttn, oldroot, newroot; use_gpu = use_gpu, node_cache = node_cache)

    @assert oc_projtpo == oldroot "Expected ProjTPO ortho_center $(oc_projtpo) to match oldroot $(oldroot)"

    # then, recalc path flows for the ProjTTNs moving oldroot to newroot
    pth_forward = connecting_path(network(ttn), oldroot, newroot)
    if !isnothing(pth_forward)
        pth = vcat(oldroot, pth_forward)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end
    # then, recalc path flows for the ProjTTNs moving newroot back to oldroot
    pth_back = connecting_path(network(ttn), newroot, oldroot)
    if !isnothing(pth_back)
        pth = vcat(newroot, pth_back)
        for i in 2:length(vecproj.data)
            for (jj, pk) in enumerate(pth[1:end-1])
                ism = ttn[pk]
                update_environments!(vecproj.data[i], ism, pk, pth[jj+1])
            end
        end
    end

    return vecproj
end

