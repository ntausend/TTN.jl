# represents neutral onsite operators like the density operator etc.
const OnSiteOperator{S} = AbstractTensorMap{S, 1, 1} where {S <: EuclideanSpace}
const TreeLegTensor{S}  = AbstractTensorMap{S, 1, 2} where {S <: EuclideanSpace}