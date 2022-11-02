abstract type AbstractTensorProductOperator{L <: AbstractLattice} end

dimensionality(::AbstractTensorProductOperator{N}) where N = dimensionality(N)
lattice(tpo::AbstractTensorProductOperator) = tpo.lat


