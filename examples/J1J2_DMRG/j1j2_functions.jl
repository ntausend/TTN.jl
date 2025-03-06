# J1J2 Hamiltonian, J1 is the nearst neighbor coupling, J2 the coupling to next nearest neighbors
function J1J2Ham(J1, J2, lat; periodic = false)
    ampo = OpSum()
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

# monitoring the energy
mutable struct EnergyObserver <: AbstractObserver
    en_vec::Vector{Float64}
    EnergyObserver() = new(Float64[])
end

import ITensorMPS: measure!
function measure!(ob::EnergyObserver; kwargs...)
    sh = kwargs[:sweep_handler]
    push!(ob.en_vec, sh.current_energy)
end

# function to get the next nearest neighbors on our lattice
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
