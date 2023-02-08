function convert_cu(T::ITensor, T_type::ITensor)
    ITensorGPU.is_cu(T_type) || return T
    return cu(eltype(T_type), T)
end

convert_cu(T::ITensor, ttn::TreeTensorNetwork{<:AbstractNetwork,ITensor}) = convert_cu(T, ttn[(1,1)])

function convert_cu(T::Vector{ITensor}, T_type::ITensor)
    return map(t -> convert_cu(t, T_type), T)
end

function convert_cu(T::Op, T_type::ITensor)
    ITensorGPU.is_cu(T_type) || return T
    return Op(cu(eltype(T_type), which_op(T)), T.sites...; T.params...)
end

convert_cu(T::Op, ttn::TreeTensorNetwork{<:AbstractNetwork,ITensor, ITensorsBackend}) = convert_cu(T, ttn[(1,1)])
convert_cu(T::Vector{Op}, ttn::TreeTensorNetwork{<:AbstractNetwork,ITensor, ITensorsBackend}) = map(t -> convert_cu(t, ttn), T)

function gpu(ttn::TTNKit.TreeTensorNetwork; type::Type = ComplexF64)
    datac = deepcopy(ttn.data)
    datagpu = map(datac) do layerdata
        map(T -> cu(type, T), layerdata)
    end
    ortho_centerc = deepcopy(ttn.ortho_center)
    netc = deepcopy(ttn.net)
    ortho_directionc = deepcopy(ttn.ortho_direction)
    return TreeTensorNetwork(datagpu, ortho_directionc, ortho_centerc, netc)
end

function gpu(mpo::TTNKit.MPOWrapper{L, M, TTNKit.ITensorsBackend}; type::Type = ComplexF64) where{L,M}
    datac = deepcopy(mpo.data)
    datagpu = map(T -> cu(type, T), datac) 
    mappingc = deepcopy(mpo.mapping)
    latc = deepcopy(mpo.lat)
    return TTNKit.MPOWrapper{L, M, TTNKit.ITensorsBackend}(latc, datagpu, mappingc)
end

function gpu(tpo::TTNKit.TPO{L, TTNKit.ITensorsBackend}, ttn::TreeTensorNetwork) where L
    latc = deepcopy(tpo.lat)
    datac = deepcopy(tpo.data)
    datagpu = map(datac) do T
    	newargs = Tuple(map(T.args) do vecOp
	   map(op -> convert_cu(op, ttn), vecOp)
	end)
	return ITensors.Applied(T.f, newargs)
    end
    return TTNKit.TPO{L, TTNKit.ITensorsBackend}(latc, datagpu)
end

