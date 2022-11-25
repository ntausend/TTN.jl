abstract type AbstractTensorProductOperator{L <: AbstractLattice} end

dimensionality(::AbstractTensorProductOperator{L}) where L = dimensionality(L)
lattice(tpo::AbstractTensorProductOperator) = tpo.lat


