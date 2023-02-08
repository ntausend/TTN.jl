abstract type AbstractTensorProductOperator{L <: AbstractLattice, B<:AbstractBackend} end

dimensionality(::AbstractTensorProductOperator{L}) where L = dimensionality(L)
lattice(tpo::AbstractTensorProductOperator) = tpo.lat
backend(::Type{<:AbstractTensorProductOperator{L,B}}) where{L,B} = B
backend(tpo::AbstractTensorProductOperator) = backend(typeof(tpo)) 
ProjectedTensorProductOperator(ttn::TreeTensorNetwork, tpo::AbstractTensorProductOperator) = nothing