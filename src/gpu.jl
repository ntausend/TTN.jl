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

function convert_cu(T::ITensor, T_type::ITensor)
    return ITensorGPU.is_cu(T_type) ? cu(T) : T
end

function convert_cu(T::Vector{ITensor}, T_type::ITensor)
    return ITensorGPU.is_cu(T_type) ? cu.(T) : T
end
