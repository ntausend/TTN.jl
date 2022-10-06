struct SoftCoreBosonNode{S<:IndexSpace,I<:Sector,N} <: PhysicalNode{S,I}
    s::Int
    hilbertspace::S
    desc::AbstractString
    function SoftCoreBosonNode(pos::Int, desc::AbstractString="";
            conserve_qns::Bool = true, conserve_parity::Bool = conserve_qns,
            number_of_bosons::Int = 1, parity::Int = 2)
        # 1 is the minimum number of states. otherwise this makes no sense at all
        @assert number_of_bosons ≥ 1
        # parity conservation has to be larger than 2.. otherwise this makes no sense
        @assert parity≥2
        if conserve_qns
            sp = U1Space
            sectors = Tuple(map(0:number_of_bosons) do (nn)
                nn => 1
            end)
        elseif conserve_parity
            sp = ZNSpace{parity}
            n_compl = div(number_of_bosons+1, parity)
            n_res   = rem(number_of_bosons+1, parity)
            sec_av  = 0:parity-1

            sectors = map(sec_av) do (nn)
                n_sec = nn+1 ≤ n_res ? n_compl + 1 : n_compl
                nn => n_sec
            end
        else
            sp = ComplexSpace
            sectors = (number_of_bosons + 1)
        end
        return new{sp, sectortype(sp), number_of_bosons}(pos, sp(sectors), "SCB "*desc)
    end
end


function state_dict(::Type{SoftCoreBosonNode{S,I,N}}) where{S,I,N}
    states_vec = map(1:N+1) do nn
        vec = zeros(N+1)
        vec[nn] = 1
        return vec
    end
    d = Dict{String, Vector{Int}}()
    names = string.(collect(0:N))
    for (s, st) in zip(names, states_vec)
        d[s] = st
    end
    return d
end


function state_dict(::Type{SoftCoreBosonNode{S,ZNIrrep{P},N}}) where{S,P,N}
    
    states_vec = map(1:N+1) do nn
        vec = zeros(N+1)
        vec[nn] = 1
        return vec
    end
    d = Dict{String, Vector{Int}}()
    st = collect(0:N)
    names = string.(st)

    for (n,s) in zip(names,st)
        chrg = mod(s, P)
        idx_bl  = div(s, P) + 1
        idx_vec = idx_bl + chrg*P
        d[n] = states_vec[idx_vec]
    end
    return d
end

charge_dict(::Type{SoftCoreBosonNode{S,Trivial}}) where {S} = nothing
function charge_dict(::Type{SoftCoreBosonNode{S, U1Irrep, N}}) where{S,N}
    st = collect(0:N)
    names = string.(st)

    I = U1Irrep
    d = Dict{String, I}()
    for (n, s) in zip(names, st)
        d[n] = I(s)
    end
    return d
end

function charge_dict(::Type{SoftCoreBosonNode{S, ZNIrrep{P}, N}}) where{S,P,N}
    st    = collect(0:N)
    names = string.(st)

    I = ZNIrrep{P}
    d = Dict{String, I}()

    for (n, s) in zip(names, st)
        chrg = mod(s, P)
        d[n] = I(chrg)
    end
    return d
end
