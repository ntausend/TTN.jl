#NDTensors.similar(D::NDTensors.Dense) = NDTensors.Dense(Base.similar(NDTensors.data(D)))

is_cu(::Type{<:Array}) = false
is_cu(::Type{<:Number}) = false
is_cu(::Type{<:CuArray}) = true
is_cu(X::Type{<:NDTensors.TensorStorage}) = is_cu(NDTensors.datatype(X))
is_cu(X::Type{<:NDTensors.Tensor}) = is_cu(NDTensors.storagetype(X))

is_cu(x::NDTensors.TensorStorage) = is_cu(typeof(x))
is_cu(x::NDTensors.Tensor) = is_cu(typeof(x))
is_cu(x::ITensor) = is_cu(typeof(NDTensors.tensor(x)))

function convert_cu(T::ITensor, T_type::ITensor)
    #ITensorGPU.is_cu(T_type) || return T
    is_cu(T_type) || return T

	# explicit cast on dense arrays in the case of a delta since the gpu interface does not support Diag NDTensors... sofar
    if ITensors.tensor(T) isa ITensors.NDTensors.DiagTensor
        return adapt(CuArray, dense(T))
    end
    # return cu(T)
    return adapt(CuArray, T)
end


function convert_cu(T::ITensor)
    is_cu(T) && return T
    return adapt(CuArray, T)
end

convert_cu(T::ITensor, ttn::TreeTensorNetwork) = convert_cu(T, ttn[(1,1)])

function convert_cpu(T::ITensor)
    return adapt(Array, T)
end

function convert_cu(T::Vector{ITensor}, T_type::ITensor)
    return map(t -> convert_cu(t, T_type), T)
end

function convert_cu(T::Op, T_type::ITensor)
    is_cu(T_type) || return T
    return Op(adapt(CuArray, which_op(T)), T.sites...; T.params...)
end

convert_cu(T::Op, ttn::TreeTensorNetwork) = convert_cu(T, ttn[(1,1)])
convert_cu(T::Vector{Op}, ttn::TreeTensorNetwork) = map(t -> convert_cu(t, ttn), T)

function convert_cpu(T::Op)
    return Op(adapt(Array, which_op(T)), T.sites...; T.params...)
end

function convert_cu(T::Op)
    is_cu(which_op(T)) && return T
    return Op(adapt(CuArray, which_op(T)), T.sites...; T.params...)
end

function convert_cu(T::Prod{Op})
    new_args = tuple(convert_cu.(collect(T.args))...)
    return Prod{Op}(T.f, new_args, T.kwargs)
end

function convert_cu(env::Vector{Prod{Op}})
    return convert_cu.(env)
end

function convert_cu(env::Vector{Vector{Vector{Prod{Op}}}})
    return map(x->map(y->convert_cu.(y), x), env)
end

function convert_cu(T::Vector{Op})
    return convert_cu.(T)
end

function convert_cu(T::Vector{Vector{Op}})
    return convert_cu.(T)
end

function convert_cpu(T::Prod{Op})
    new_args = tuple(convert_cpu.(collect(T.args))...)
    return Prod{Op}(T.f, new_args, T.kwargs)
end

function convert_cpu(env::Vector{Prod{Op}})
    return convert_cpu.(env)
end

function convert_cpu(env::Vector{Vector{Vector{Prod{Op}}}})
    return map(x->map(y->convert_cpu.(y), x), env)
end

function convert_cpu(T::Vector{Op})
    return convert_cpu.(T)
end


"""
```julia
    cpu(ttn::TreeTensorNetwork; type::Type = ComplexF64)
```

Copy the data of the tree tensor network from the GPU to the CPU.
"""
function cpu(ttn::TreeTensorNetwork; type::Type = ComplexF64)
    for ll in eachindex(ttn.data)
        for pp in eachindex(ttn.data[ll])
            ttn.data[ll][pp] = adapt(Array, ttn.data[ll][pp])
        end
    end
    return ttn
end

# function cpu(ttn::TreeTensorNetwork; type::Type = ComplexF64)
#     datac = deepcopy(ttn.data)
#     datacpu = map(datac) do layerdata
#         return map(T -> adapt(Array, T), layerdata)
#     end
# 
#     ortho_centerc = deepcopy(ttn.ortho_center)
#     netc = deepcopy(ttn.net)
#     ortho_directionc = deepcopy(ttn.ortho_direction)
#     return TreeTensorNetwork(datacpu, ortho_directionc, ortho_centerc, netc)
# end
"""
```julia
    gpu(ttn::TreeTensorNetwork; type::Type = ComplexF64)
```

Copy the data of the tree tensor network onto the GPU.
"""
function gpu(ttn::TreeTensorNetwork; type::Type = ComplexF64)
    datac = deepcopy(ttn.data)
    datagpu = map(datac) do layerdata
        return map(T -> adapt(CuArray, T), layerdata)
    end
    ortho_centerc = deepcopy(ttn.ortho_center)
    netc = deepcopy(ttn.net)
    ortho_directionc = deepcopy(ttn.ortho_direction)
    return TreeTensorNetwork(datagpu, ortho_directionc, ortho_centerc, netc)
end

#=
function gpu(mpo::MPOWrapper{L, M}; type::Type = ComplexF64) where{L,M}
    datac = deepcopy(mpo.data)
    datagpu = map(T -> cu(type, T), datac) 
    mappingc = deepcopy(mpo.mapping)
    latc = deepcopy(mpo.lat)
    return MPOWrapper{L, M}(latc, datagpu, mappingc)
end

function gpu(tpo::TPO{L}, ttn::TreeTensorNetwork) where L
    latc = deepcopy(tpo.lat)
    datac = deepcopy(tpo.data)
    datagpu = map(datac) do T
            newargs = Tuple(map(T.args) do vecOp
                    map(op -> convert_cu(op, ttn), vecOp)
                end)
        return ITensors.Applied(T.f, newargs)
    end
    return TPO{L}(latc, datagpu)
end
=#

