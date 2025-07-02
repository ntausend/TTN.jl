function full_contraction(ttn::TreeTensorNetwork, tpo::TPO_GPU)
    ptpo = ProjTPO_GPU(ttn, tpo)
    return full_contraction(ttn, ptpo)
end

function full_contraction(ttn::TreeTensorNetwork, ptpo::ProjTPO_GPU)
    # set the ptpo to the correct position of the ttn
    ptpo = set_position!(ptpo, ttn)
    oc = ortho_center(ttn)

    # get the action of the operator on the orthogonlity center
    action = ∂A(ptpo, oc)
    T = ttn[oc]
    # build the contraction
    return dot(T, action(T))
end

function set_position!(pTPO::ProjTPO_GPU{N,T}, ttn::TreeTensorNetwork{N,T}) where {N,T}
    oc_projtpo = ortho_center(pTPO)
    oc_ttn     = ortho_center(ttn)
    # both structures should be gauged.. otherwise no real thing todo
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    all(oc_projtpo .== oc_ttn) && return pTPO

    # move oc of link_operators from oc_projtpo to oc_ttn
    recalc_path_flows!(pTPO, ttn, oc_ttn)
    return pTPO
end

"""
    ∂A(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int})

Return a closure `action(T)` that applies the part of the *projected*
Hamiltonian living on node `pos` to an `ITensor T` on the **same** node.

For every operator–ID that still has support on one (or more) of the three
links attached to `pos` (parent, first-child, second-child) the routine

  1. collects the corresponding `ITensor`s that represent that operator’s
     pieces near `pos`;
  2. builds the minimal contraction network `[T, op₁, op₂, …]`;
  3. contracts it with an optimal sequence; and
  4. finally sums all such contributions.

This reproduces the behaviour of the original `∂A` but now talks to the new
`ProjTPO_GPU` data model.
"""
#=
function ∂A(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int})
    net      = ptpo.net
    link_ops = ptpo.link_ops

    # ─────────────── gather the environment ────────────────
    # Dict{Int, Vector{Op_GPU}} keyed by the global operator id
    id_bucket = get_id_terms(net, link_ops, pos)

    # Pre-extract the raw ITensors; cheap and avoids allocations
    envs = [map(g -> g.op, grp) for grp in values(id_bucket)]

    # ───────────────────── the action closure ─────────────────
    function action(T::ITensor)
        isempty(envs) && return zero(T)  # nothing acts on this node

        acc = nothing
        for trm in envs
            tensor_list = vcat(T, trm)                      # [T, op₁, …]
            seq         = ITensors.optimal_contraction_sequence(tensor_list)
            contrib     = noprime(contract(tensor_list; sequence = seq))
            acc === nothing ? (acc = contrib) : (acc += contrib)
        end
        return acc
    end

    return action
end
=#

function ∂A_GPU(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    net      = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)

    # Pre-extract the raw ITensors from all Op_GPU environments
    envs = [map(g -> g.op, grp) for grp in values(id_bucket)]

    if use_gpu
        action = function (T::ITensor)
            isempty(envs) && return zero(T)

            acc_gpu = nothing
            T_gpu = convert_cu(T, T)

            for trm in envs
                ops_gpu = convert_cu(trm, trm[1])
                tensor_list = vcat(T_gpu, ops_gpu)
                seq = ITensors.optimal_contraction_sequence(tensor_list)
                contrib_gpu = noprime(contract(tensor_list; sequence = seq))
                acc_gpu === nothing ? (acc_gpu = contrib_gpu) : (acc_gpu += contrib_gpu)
            end
            return cpu(acc_gpu)
        end
    else
        action = function (T::ITensor)
            isempty(envs) && return zero(T)

            acc = nothing
            for trm in envs
                tensor_list = vcat(T, trm)
                seq         = ITensors.optimal_contraction_sequence(tensor_list)
                contrib     = noprime(contract(tensor_list; sequence = seq))
                acc === nothing ? (acc = contrib) : (acc += contrib)
            end
            return acc
        end
    end

    return action
end


