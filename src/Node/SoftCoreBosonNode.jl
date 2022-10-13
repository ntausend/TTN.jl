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


_parity(::Type{<:ZNIrrep{P}}) where{P} = P
function spaces(::SoftCoreBosonNode{S,I,N}) where{S,I,N}
    if I == Trivial
        return N + 1
    elseif I <: ZNIrrep
        parity = _parity(I)
        return [mod(n, parity) => 1 for n in 0:N]
    else
        return [n => 1 for n in 0:N]
    end
end

function state(::SoftCoreBosonNode{S,I,N}, ::Val{V}) where{S,I,N, V}
    occ = nothing
    try
        occ = parse(Int64, string(V))
    catch ArgumentError
        error("$V is not a valid integer needed for indicating occupation")
    end
    occ > N && error("Requested occupation number $occ larger than total number of bosons...")
    return [jj == occ ? 1 : 0 for jj in 0:N]
end