function write_tree_jld2(file_name::AbstractString, ttn::TTNKit.TreeTensorNetwork; kwargs...)
    binary_file = jldopen(file_name,"w")

    alldata = JLD2.Group(binary_file, "all_data")
    alldata["ttn"] = ttn

    close(binary_file)
    return nothing
end


function reconstruct_jld2_network(wavefunc)
    lattices_jld2 = (wavefunc.net).lattices
    allnodes = [lattices_jld2[i].lat for i in 1:length(lattices_jld2)]
    alldims = [lattices_jld2[i].dims for i in 1:length(lattices_jld2)]
    all_simplelattices = [TTNKit.SimpleLattice(allnodes[i],alldims[i]) for i in 1:length(lattices_jld2)]
    return TernaryNetwork(all_simplelattices)
end
function reconstruct_jld2_ttn(wavefunc)
    remade_network = reconstruct_jld2_network(wavefunc)
    return TTNKit.TreeTensorNetwork(wavefunc.data,wavefunc.ortho_direction,wavefunc.ortho_center,remade_network)
end

function write_results(maxdim, entr, x, e, save_root, L)

    svn = joinpath(save_root, "results_L$(L).h5")
    chrd_pos = chrd_dist.(x, N)
    h5open(svn, "cw") do f
            if haskey(f, "bnd_$(maxdim)")
                delete_object(f,"bnd_$(maxdim)")
            end

            write(f, "bnd_$(maxdim)/entanglement/entropy", entr)
            write(f, "bnd_$(maxdim)/entanglement/position", x)
            write(f, "bnd_$(maxdim)/entanglement/chrd_pos", chrd_pos)
            write(f, "bnd_$(maxdim)/energ", e)
        end
    return nothing
end
