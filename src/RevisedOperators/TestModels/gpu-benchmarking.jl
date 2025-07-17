#####################################################
#=

This file contains GPU testing and benchmarking on TTNs

=#
######################################################

#using Pkg
#Pkg.activate(".")

using TTN
include("../other-funcs/include-other-files.jl")
include_other_files(["other-funcs/data-storage-funcs.jl","review-practice-codes/overwriting-ttn.jl"])

#using CUDA
using LinearAlgebra

mutable struct BenchmarkObserver <: TTN.AbstractObserver
    # sweep_count::Int
    sweep_times::Vector{Float64}

    BenchmarkObserver() = new([])
end

function TTN.measure!(o::BenchmarkObserver; kwargs...)
    append!(o.sweep_times,[kwargs[:dt]])
end

#=
# optional custom observer
function TTN.checkdone!(o::BenchmarkObserver;kwargs...)
	sh = kwargs[:sweep_handler]
    if sh.current_sweep == o.sweep_count
        return true
    else
        return false
    end
end
=#

function simple_HH_ham(latsize::Tuple; kwargs...)

    if_periodic::Bool = kwargs[:if_periodic]
    t_strength::Float64 = kwargs[:t_strength]
    phi::Float64 = kwargs[:phi]

    hopping = TTN.OpSum()
    for s_phys in 1:latsize[1]
        for s_synth in 1:latsize[2]
            starting_site = [s_phys,s_synth]
            for which_axis in [1,2]
                ending_site = starting_site .+ ((which_axis == 1,which_axis == 2))

                if ending_site[which_axis] > latsize[which_axis] 
                    if if_periodic
                        ending_site[which_axis] = 1
                    else
                        continue
                    end
                end

                if ending_site[which_axis] < 1
                    if if_periodic
                        ending_site[which_axis] = latsize[which_axis]
                    else
                        continue
                    end
                end

                if starting_site[1] == ending_site[1] # Synthetic Dimension Hopping
                    coeff = t_strength
                elseif starting_site[2] == ending_site[2] # Physical Dimension Hopping
                    coeff = t_strength * exp(im*2*pi*(phi*starting_site[2]))
                end

                hopping += (coeff,"Adag",Tuple(starting_site),"A",Tuple(ending_site))
                hopping += (conj(coeff),"Adag",Tuple(ending_site),"A",Tuple(starting_site))
            end
        end
    end

    return hopping
end

function make_HH_ham(net::TTN.AbstractNetwork; kwargs...)
    lx::Int,ly::Int = size(net.lattices[1])
    n::Int = Int(ceil(0.5*lx*ly))
    if_periodic::Bool = get(kwargs,:if_periodic,false)
    
    phys_shift,synth_shift = !if_periodic,!if_periodic
	alpha = n/(0.5*(lx - phys_shift)*(ly - synth_shift))
    
    ham = simple_HH_ham((lx,ly); t_strength=1.0, phi=alpha, if_periodic=if_periodic)

    tpo = TTN.build_tpo_from_opsum(ham,TTN.physical_lattice(net))
    
    return tpo
end

function next_nearest_neighbours(lat::TTN.SimpleLattice, mapping::Vector{Int}; periodic::Bool = false)

    prod_it = Iterators.product(UnitRange.(1, lat.dims)...)
    mapping = TTN.inverse_mapping(mapping)
        iter = map(prod_it) do pos
           map([-1,+1]) do dir
                    xpos = pos[2] + dir
                    ypos = pos[1] + 1
                    (!periodic && (xpos == 0 || xpos > lat.dims[2])) && return
                    (!periodic && (ypos > lat.dims[1])) && return

                        new_pos = (ypos, xpos)
                        nextpos = map(zip(new_pos, lat.dims)) do (pp,d)
                                return mod(pp-1, d)+1
                        end
                        return (mapping[TTN.linear_ind(lat, pos)], mapping[TTN.linear_ind(lat, Tuple(nextpos))])
                end
        end
    return Vector{Tuple{Vararg{Int}}}(filter(!isnothing, vcat(vec(iter)...)))
end

function simple_J1J2_ham(net::TTN.AbstractNetwork; kwargs...)
    J1::Float64 = kwargs[:j1_strength]
    J2::Float64 = kwargs[:j2_strength]
    periodic::Bool = kwargs[:if_periodic]

    lat = TTN.physical_lattice(net)

    ampo = TTN.OpSum()
    for bond in TTN.nearest_neighbours(lat, collect(eachindex(lat)), periodic = periodic)
        b1 = TTN.coordinate(lat, first(bond))
        b2 = TTN.coordinate(lat, last(bond))
        ampo += J1, "X", b1, "X", b2
        ampo += J1, "Y", b1, "Y", b2
        ampo += J1, "Z", b1, "Z", b2
    end
    for bond in next_nearest_neighbours(lat, collect(eachindex(lat)), periodic = periodic)
        b1 = TTN.coordinate(lat, first(bond))
        b2 = TTN.coordinate(lat, last(bond))
        ampo += J2, "X", b1, "X", b2
        ampo += J2, "Y", b1, "Y", b2
        ampo += J2, "Z", b1, "Z", b2
    end
    return ampo
end

function make_J1J2_ham(net::TTN.AbstractNetwork; kwargs...)
    lx::Int,ly::Int = size(net.lattices[1])
    n::Int = Int(ceil(0.5*lx*ly))
    if_periodic::Bool = kwargs[:if_periodic]
    j1_strength::Float64 = kwargs[:j1_strength]
    j2_strength::Float64 = kwargs[:j2_strength]
    
    ham = simple_J1J2_ham(net; j1_strength=j1_strength, j2_strength=j2_strength, if_periodic=if_periodic)    

    tpo = TTN.build_tpo_from_opsum(ham,TTN.physical_lattice(net))
    
    return tpo
end

function build_hamiltonian(model::String,net::TTN.AbstractNetwork; kwargs...)
    if model == "HH"
        tpo = make_HH_ham(net; kwargs...)
    elseif model == "J1J2"
        tpo = make_J1J2_ham(net; kwargs...)
    end
    return tpo
end

function build_network(model::String,lx::Int,ly::Int; kwargs...)
    max_occ::Int = get(kwargs,:max_occ,2)

    if model == "HH"
        # net = BinaryRectangularNetwork((lx,ly), "Boson"; conserve_qns=false, dim = max_occ+1)
        net = BinaryRectangularNetwork((lx,ly), "S=1/2"; conserve_qns=false, dim = max_occ+1)
    elseif model == "J1J2"
        # net = BinaryRectangularNetwork((lx,ly), "Qubit"; conserve_qns=false)
        net = BinaryRectangularNetwork((lx,ly), "S=1/2"; conserve_qns=false)

    end
    return net
end

function initialize_benchmark(pu_type::String,net::TTN.AbstractNetwork,ham_tpo::TTN.TPO; kwargs...)

    # build TTN
    psi = RandomTreeTensorNetwork(net; maxdim=25)
    if pu_type == "gpu"
        use_gpu = true
    else
        use_gpu = false
    end

    # do DMRG
    # sp = TTN.dmrg(psi,tpo; number_of_sweeps=5, maxdims=25, cutoff=0, eigsolve_krylovdim=5, eigsolve_verbosity=0, use_gpu=use_gpu)

    # do tdvp
    sp = TTN.tdvp(psi, tpo, initialtime = 0, finaltime = 1, timestep = 0.1, use_gpu = use_gpu);


    println("Finished $pu_type initialization")
end

function run_benchmark(pu_type::String,mdim::Int,net::TTN.AbstractNetwork,ham_tpo::TTN.TPO,filepath::Union{String,Nothing}; kwargs...)
    # get kwargs
    opl::Int = get(kwargs,:opl,1)

    timestep = get(kwargs, :timestep, 1e-2)
    initialtime = get(kwargs, :initialtime, 0.)
    finaltime = get(kwargs, :finaltime, 1.)
    num_sweeps = (finaltime - initialtime) / timestep

    # build TTN
    psi = RandomTreeTensorNetwork(net; maxdim=mdim)

    if pu_type == "gpu"
        use_gpu = true
    else
        use_gpu = false
    end
    opl > 1 && println("Starting linkdim = $(maxlinkdim(psi))")

    # make observer
    obs = BenchmarkObserver()

    #=
    # do DMRG
    time_start = time()
    sp = TTN.dmrg(psi,ham_tpo; observer=obs, number_of_sweeps=num_sweeps, maxdims=mdim, cutoff=0, eigsolve_krylovdim=5, eigsolve_verbosity=0, use_gpu = use_gpu)
    final_runtime = (time() - time_start) / num_sweeps
    
    append!(obs.sweep_times,[final_runtime])

    final_mdim = TTN.maxlinkdim(sp.ttn)
    =#

    # do tdvp
    time_start = time()
    sp = TTN.tdvp(psi, tpo, initialtime = initialtime, finaltime = finaltime, timestep = timestep, use_gpu = use_gpu)
    final_runtime = (time() - time_start) / num_sweeps

    append!(obs.sweep_times,[final_runtime])
    final_mdim = TTN.maxlinkdim(sp.ttn)

    # save data

    new_data = Dict([("$final_mdim",obs.sweep_times)])
    !isnothing(filepath) && modify_data_hdf5(new_data,filepath,"all_data")

    opl > 0 && println("GPU: mdim = $final_mdim, time = $final_runtime")

    return obs,final_mdim
end

function get_pu_info(pu_type::String)
    if pu_type == "gpu"
        pu_info = CUDA.versioninfo()
    elseif pu_type == "cpu"
        pu_info = read(`lscpu`, String)
    end
    return pu_info
end

function make_datafile(dataloc::String; model_paras...)
    # dataloc = get_folder_location("cluster-data/gpu-benchmarking")
    dataloc = get_folder_location("C:\\Users\\elbracht\\Documents\\Master\\cluster-data\\gpu-benchmarking")

    if_save_data::Bool = model_paras[:if_save_data]
    benchmark_type::String = model_paras[:benchmark_type]
    which_model::String = model_paras[:model]
    layers::Int = model_paras[:layers]
    min_mdim::Int = model_paras[:min_mdim]
    max_mdim::Int = model_paras[:max_mdim]
    count_mdim::Int = model_paras[:count_mdim]

    pu_info = get_pu_info(benchmark_type)
    metadata::Dict{String,Any} = named_tuple_to_dict(model_paras)
    metadata["pu_info"] = pu_info
    filename = "$(benchmark_type)-benchmarking-data-model-$which_model-layers-$layers-startmdim-$min_mdim-endmdim-$max_mdim-countmdim-$count_mdim.h5"

    filepath = if_save_data ? joinpath(dataloc,filename) : nothing
    model_paras[:if_save_data] && (filename = write_data_hdf5(filepath,Dict(),metadata))

    return filepath
end

function benchmark_model_params(args_dict::Dict{String,Any})
    # set benchmarking parameters
    benchmark_type::String = args_dict["benchmark_type"]
    min_mdim::Int = args_dict["min_mdim"]
    max_mdim::Int = args_dict["max_mdim"]
    count_mdim::Int = args_dict["count_mdim"]
    diff_mdim::Int = Int(floor((max_mdim - min_mdim) / count_mdim))
    thread_count::Int = get(args_dict,"thread_count",1)
    count_mdim = length(min_mdim:diff_mdim:max_mdim)
    if_save_data::Bool = get(args_dict,"if_save_data",true)

    # set system and Ham parameters
    which_model = args_dict["model"]
    lx = get(args_dict,"lx",4)
    ly = get(args_dict,"ly",lx)
    n = Int(ceil(0.5*lx*ly))
    layers = Int(log(2,lx*ly))
    if_periodic = get(args_dict,"if_periodic",false)
    max_occ = 2
    etol = 1e-5
    conserve_qns = false
    j1_strength::Float64 = get(args_dict,"j1",1.0)
    j2_strength::Float64 = get(args_dict,"j2",0.5)

    opl::Int = get(args_dict,"opl",1)

    opl > 0 && println("Benchmarking $which_model model on $benchmark_type with $(lx)x$ly lattice from $min_mdim to $max_mdim with $count_mdim points")

    model_paras = (benchmark_type=benchmark_type,
                    if_save_data=if_save_data,
                    min_mdim=min_mdim,
                    max_mdim=max_mdim,
                    count_mdim=count_mdim,
                    diff_mdim=diff_mdim,
                    thread_count=thread_count,
                    model=which_model,
                    j1_strength=j1_strength,
                    j2_strength=j2_strength,
                    lx=lx,
                    ly=ly,
                    n=n,
                    layers=layers,
                    if_periodic=if_periodic,
                    max_occ=max_occ,
                    etol=etol,
                    conserve_qns=conserve_qns)

    return model_paras
end

# args_dict = Dict([("benchmark_type","gpu"),("model","J1J2"),("min_mdim",40),("max_mdim",60),("count_mdim",5),("if_save_data",true)])
# args_dict = make_args_dict(ARGS)
args_dict = Dict(
    "benchmark_type" => "gpu",
    "model" => "J1J2",
    "min_mdim" => 40,
    "max_mdim" => 60,
    "count_mdim" => 5,
    "if_save_data" => true,
    
    # TDVP-specific keys:
    "timestep" => 0.1,
    "initialtime" => 0.0,
    "finaltime" => 1.0
)
args_dict = make_args_dict(ARGS)

# make model parameters
model_paras = benchmark_model_params(args_dict)
benchmark_type = model_paras[:benchmark_type]

# make datafile
# dataloc = get_folder_location("cluster-data/gpu-benchmarking")
dataloc = get_folder_location("C:\\Users\\elbracht\\Documents\\Master\\cluster-data\\gpu-benchmarking")
filepath = make_datafile(dataloc; model_paras...)

# create network
lx,ly = model_paras[:lx],model_paras[:ly]
net = build_network(model_paras[:model],lx,ly)

# create Hamiltonian
tpo = build_hamiltonian(model_paras[:model],net; model_paras...)

# initialization step
initialize_benchmark(benchmark_type,net,tpo)

# run benchmark

BLAS.set_num_threads(model_paras[:thread_count])

mdims = collect(model_paras[:min_mdim]:model_paras[:diff_mdim]:model_paras[:max_mdim])
alltimes = zeros(Float64,length(mdims))
allmdims = zeros(Float64,length(mdims))
for (idx,mdim) in enumerate(mdims)
    localobs,allmdims[idx] = run_benchmark(benchmark_type,mdim,net,tpo,filepath)
    alltimes[idx] = localobs.sweep_times[end]
end


# plot benchmarking results
if true
    using PyPlot,LaTeXStrings
    layers = 4
    lx,ly = Int(sqrt(2^layers)),Int(sqrt(2^layers))
    model = "HH"
    # dataloc = get_folder_location("cluster-data/gpu-benchmarking")
    dataloc = get_folder_location("C:\\Users\\elbracht\\Documents\\Master\\cluster-data\\gpu-benchmarking")
    all_files = readdir(dataloc)
    filter!(x -> occursin("h5",x),all_files)
    filter!(x -> occursin("layers-$layers",x),all_files)

    cpu_files = filter(x -> occursin("cpu",x),all_files)
    for (idx,f) in enumerate(cpu_files)
        d,m = read_data(joinpath(dataloc,f))
        alltimes = []
        allmdims = []
        for (k,v) in d
            push!(alltimes,v)
            push!(allmdims,parse(Float64,k))
        end
        scatter(allmdims,alltimes,label="CPU",color="r")
    end

    gpu_files = filter(x -> occursin("gpu",x),all_files)
    for (idx,f) in enumerate(gpu_files)
        d,m = read_data(joinpath(dataloc,f))
        alltimes = []
        allmdims = []
        which_gpu = " Desktop"
        col = "g"
        for (k,v) in d
            if typeof(v) == Float64
                push!(alltimes,v)
            else
                push!(alltimes,v[end])
                which_gpu = " A100 Multi-CPU"
                col = "b"

                if m["count_mdim"] == 51
                    which_gpu = " A100 1CPU"
                    col = "k"
                end
            end
            push!(allmdims,parse(Float64,k))
        end
        scatter(allmdims,alltimes,label="GPU"*which_gpu,color=col)
    end

    h100_times = [198.87454857826233, 265.86777238845826, 280.11588940620425, 306.8766342163086, 322.31804299354553, 286.09227719306944, 310.1168940067291, 373.2427127838135, 346.71673998832705, 318.8797718048096, 302.0222489833832, 352.215061378479, 386.3397141933441, 396.3376505851746, 391.7669888019562, 402.23571157455444, 404.59406418800353, 397.2337389945984, 434.29764280319216, 442.1614262104034, 427.78434977531435, 417.5367869853973, 428.7551634311676, 445.4978229999542, 402.9635287761688, 322.4677674293518, 463.43472800254824, 432.6612326145172, 452.6140230178833, 328.6008199691772, 473.66247539520265, 360.0862644195557, 449.306504201889, 361.0193250179291, 399.866454410553, 476.1155910015106, 485.3418007850647, 433.2053366184235, 470.3154815673828, 481.42112321853637, 528.8874010086059, 542.0552119731904, 508.9100049972534, 548.6776924133301, 474.28850717544555, 510.1942920207977, 493.6719464302063, 560.277167224884]
    h100_mdims = range(50,stop=1000,length=50)[1:48]
    scatter(h100_mdims,h100_times,label="GPU H100",color="m")

    xlabel("Max Bond Dimension")
    ylabel("Time (s)")
    title("Sweep Time vs "*L"$\chi$"*", $(lx)x$(ly), $model model")
    legend()
    xscale("log")
    yscale("log")
end

#=dataloc = get_folder_location("cluster-data/gpu-benchmarking")
for f in all_files
    d,m = read_data(joinpath(dataloc,f))

    layers = m["layers"]
    stren = m["stren"]
    if_periodic = m["if_periodic"]
    lx = m["lx"]
    ly = m["ly"]
    max_occ = m["max_occ"]
    conserve_qns = true
    net = BinaryRectangularNetwork((lx,ly), "Boson"; conserve_qns=conserve_qns, dim = max_occ+1)
    params_dict = Dict([("hopping_anisotropy",1.0),("if_check_fluxes",false),("particles",n),("layers",layers),("filling",0.5),("onsite_strength",stren),("lr","all"),("if_periodic_phys",if_periodic),("if_periodic_synth",if_periodic)])
    model_paras = get_normal_model_params(params_dict)
    ham = long_range_HH_ham(net,model_paras[:ts],model_paras[:alpha]; model_paras...)

    new_data = Dict([("model","HH"),("ham",ham)])
    modify_data_hdf5(new_data,filepath,"metadata")
end=#


































"fin"