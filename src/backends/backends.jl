abstract type AbstractBackend end

struct ITensorsBackend <: AbstractBackend end
struct TensorKitBackend <: AbstractBackend
    field::Union{Field, Type{<:EuclideanSpace}}
    TensorKitBackend(;field = ComplexSpace) = new(field)
end