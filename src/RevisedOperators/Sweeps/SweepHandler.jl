"""
    ∂A(ptpo::ProjTPO_group, pos::Tuple{Int,Int})

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
`ProjTPO_group` data model.
"""
function ∂A(ptpo::ProjTPO_group, pos::Tuple{Int,Int})
    net      = ptpo.net
    link_ops = ptpo.link_ops

    # ─────────────── gather the “environment” ────────────────
    # Dict{Int, Vector{OpGroup}} keyed by the global operator id
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
