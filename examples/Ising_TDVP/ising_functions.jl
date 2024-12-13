# TFIM Hamiltonian, J is the nearst neighbor coupling, g the transverse field and h the longitudinal field
function TFIMHam(J, g, h, lat; periodic=false)
   ampo = ITensors.OpSum()
   for p in TTN.coordinates(lat)
      ampo += g, "Z", p
      ampo += h, "X", p
   end
   for bond in nearest_neighbours(lat, collect(eachindex(lat)); periodic = periodic)
      b1 = TTN.coordinate(lat, first(bond))
      b2 = TTN.coordinate(lat, last(bond))
      ampo += J, "X", b1, "X", b2
  end
  return ampo
end



# tdvp observer,returns basic observables like energy,x- and z-polarization and half-system entropy
mutable struct tdvpObserver <: AbstractObserver
    time::Vector{Float64}
    energy::Vector{Float64}
    x_pol::Vector{Matrix{Float64}}
    z_pol::Vector{Matrix{Float64}}
    entropy::Vector{Float64}
end

function tdvpObserver() 
  return tdvpObserver(Float64[], Float64[], Vector{Matrix{Float64}}([]), Vector{Matrix{Float64}}([]), Float64[])
end

import ITensorMPS: measure!
function measure!(ob::tdvpObserver; kwargs...)
    tdvp = kwargs[:sweep_handler]
    lat = TTN.physical_lattice(TTN.network(tdvp.ttn))

    topPos = (TTN.number_of_layers(TTN.network(tdvp.ttn)), 1)
    nextToTopPos = (TTN.number_of_layers(TTN.network(tdvp.ttn))-1, 1)

    action = TTN.∂A(tdvp.pTPO, topPos)
    T = tdvp.ttn[topPos]
    actionT = action(T)

    E = real(ITensors.scalar(dag(T)*actionT))
    push!(ob.time, round(tdvp.current_time, digits = 3))
    push!(ob.energy, E)
    push!(ob.x_pol, real.(TTN.expect(tdvp.ttn, "X")))
    push!(ob.z_pol, real.(TTN.expect(tdvp.ttn, "Z")))
    push!(ob.entropy, TTN.entanglement_entropy(tdvp.ttn, topPos, nextToTopPos))
end


# util function to save data in obs
function savedata(name::String, obs::tdvpObserver)
    name == "" && return
    h5open(name*".h5", "w") do file    
        # iterate through the fields of obs and append the data to the dataframe
        for n in fieldnames(typeof(obs))
            create_group(file, String(n))
            for (i,data) in enumerate(getfield(obs,n))
                file[String(n)][string(i)] = data
            end
        end 
    end
end
