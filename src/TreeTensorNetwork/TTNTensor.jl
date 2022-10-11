# todo -> more elegant way for contracting tensors?
const TTNTensor{S, N, 1} = AbstractTensorMap{S, N, 1} where {S<:EuclideanSpace, N}