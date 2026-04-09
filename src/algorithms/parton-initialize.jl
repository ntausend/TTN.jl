

create_wavefunction(sz::NTuple{D, Int}) where{D} = create_wavefunction(ComplexF64, sz)
create_wavefunction(elT, sz::NTuple{D,Int}) where{D} = normalize!(randn(elT, sz))

function parton_application!(ttn::TTN.TreeTensorNetwork, wf_coefs::Array, op_ins::String; maxdim::Int = maxlinkdim(ttn), normalize::Bool = true)
    net = TTN.network(ttn)
    lat = TTN.physical_lattice(net)

    all(size(lat) .== size(wf_coefs)) || error("Trying to apply a parton wavefunction of dimensionality $(size(wf_coefs)) to a TTN defined on a lattice $(size(lat))")


    parton_mpo = wf_mpo(wf_coefs, net, op_ins)
    
    for p in eachindex(TTN.lattice(net, 1))
        ttn = TTN.move_ortho!(ttn, (1,p))
        pc = TTN.child_nodes(net, (1, p))
        Tpat = map(j -> parton_mpo[j], last.(pc))
        
        pr = TTN.parent_node(net, (1,p))
        #println("Site $p has parent ",pr)
        #println("Child nodes with lattice coords are $(pc[1][2]) at $(TTN.coordinate(lat,pc[1][2])) and $(pc[2][2]) at $(TTN.coordinate(lat,pc[2][2]))")
        Tt  = ttn[(1, p)]

        rind = TTN.commonind(Tt, ttn[pr])
        linds = TTN.uniqueinds(Tt, rind)
        Tn = TTN.noprime(TTN.contract(Tt, Tpat...))

        #println("Multiplication worked")
        #if p == 2
        #    return Tn,linds,rind
        #end
        A, R = TTN.factorize(Tn, linds, maxdim = maxdim, tags = TTN.tags(rind))
        
        #=
        println("Physical Site $p")
        println("Sizes: ",", Tn",length(TTN.inds(Tn)),", R",length(TTN.inds(R)),", Pr",length(TTN.inds(ttn[pr])),", A",length(TTN.inds(A)))
        println(", Tn",TTN.tags.(TTN.inds(Tn)),", R",TTN.tags.(TTN.inds(R)),", Pr",TTN.tags.(TTN.inds(ttn[pr])),", A",TTN.tags.(TTN.inds(A)))
        =#
        ttn[(1,p)] = A
        ttn[pr] = ttn[pr] * R    
        
        # moving the orhto center
        ttn.ortho_center[1] = pr[1]
        ttn.ortho_center[2] = pr[2]
        # this has to be deleted in the future... dont need this anymore
        ttn.ortho_direction[1][p] = TTN.number_of_child_nodes(net, (1,p)) + 1
        ttn.ortho_direction[pr[1]][pr[2]] = -1
    end
    # now move to the higher layers layer by layer, excluding the top node
    #TTN.ITensorMPS.set_warn_order(20)
    for ll in 2:TTN.number_of_layers(net)-1
        for p in 1:TTN.number_of_tensors(net, ll)
            pp = (ll, p)
            ttnc = TTN.move_ortho!(ttn, pp)
            
            Tt = ttn[pp]

            pr = TTN.parent_node(net, pp)

            pc = TTN.child_nodes(net, pp)
            
            linds = map(p -> TTN.commonind(Tt, ttn[p]), pc)
            ptags = "Link,nl=$(ll),np=$(p)"#tags(commonind(Tt, ttn[pr]))
        
            A, R = factorize(Tt, linds; maxdim = maxdim, tags = ptags)
            #=
            println(ll,", ",p)
            println("With Parent $pr")
            println("Sizes: ",", R",length(TTN.inds(R)),", Tt",length(TTN.inds(Tt)),", A",length(TTN.inds(A)))
            println(", R",TTN.tags.(TTN.inds(R)),", Tt",TTN.tags.(TTN.inds(Tt)),", A",TTN.tags.(TTN.inds(A)))
            =#
            ttn[pp] = A
            ttn[pr] = ttn[pr] * R
        
            # moving the orhto center
            ttn.ortho_center[1] = pr[1]
            ttn.ortho_center[2] = pr[2]
            # this has to be deleted in the future... dont need this anymore
            ttn.ortho_direction[ll][p] = TTN.number_of_child_nodes(net, (ll,p)) + 1
            ttn.ortho_direction[pr[1]][pr[2]] = -1
        end
    end

    if normalize
        tpnd = (TTN.number_of_layers(net), 1) 
        T = ttn[tpnd]
        ttn[tpnd] = T/norm(T)
    end

    return ttn
end

function get_2dttn_path(size) # written by ChapGPT 29.05.2023
    path = []

    function traverse_lattice(x_start, y_start, x_end, y_end)
        if x_start == x_end && y_start == y_end
            push!(path, (x_start, y_start))
        else
            x_mid = (x_start + x_end) ÷ 2
            y_mid = (y_start + y_end) ÷ 2
            traverse_lattice(x_start, y_start, x_mid, y_mid)
            traverse_lattice(x_mid + 1, y_start, x_end, y_mid)
            traverse_lattice(x_start, y_mid + 1, x_mid, y_end)
            traverse_lattice(x_mid + 1, y_mid + 1, x_end, y_end)
        end
    end

    traverse_lattice(1, 1, size, size)
    return path
end

function get_site_number_square(site_label, side_length)
    x,y = site_label
    site_number = (y - 1) * side_length + x
    if site_number > side_length^2
    	println("ERROR: Outside Square")
    	return Int(site_number)
    end
    return Int(site_number)
end

function get_site_number(x,y,a,b) # written by ChatGPT 12.06.2023
    # Check if the coordinates are within the lattice range
    if x < 1 || x > a || y < 1 || y > b
        println("Outside Lattice")
        return
    end

    # Calculate the number of complete rows and columns
    complete_rows = div(x - 1, a)
    complete_cols = div(y - 1, b)

    # Calculate the number of remaining elements in the incomplete row and column
    remaining_row = rem(x - 1, a)
    remaining_col = rem(y - 1, b)

    # Calculate the number of the lattice site
    if complete_rows % 2 == 0
        site_number = complete_rows * a + complete_cols * b + 1 + remaining_row * b + remaining_col
    else
        site_number = complete_rows * a + (a - remaining_row) * b + 1 + remaining_col
    end

    return site_number
end

function add_rect_block(path,edge_size)
    second_block = []
    for i in 1:length(path)
        push!(second_block,(path[i][1]+edge_size,path[i][2]))
    end
    return [path; second_block]
end

function ttn_2d_mapping(ttn_size)
    path::Vector{Tuple} = get_2dttn_path(minimum(ttn_size))
    if ttn_size[1] != ttn_size[2]
        for i in 1:Int(log2(maximum(ttn_size)/minimum(ttn_size)))
            path = add_rect_block(path,Int(minimum(ttn_size)*i))
        end
    end
    map2d::Vector{Int64} = []
    for i in 1:length(path)
        site_idx = get_site_number(path[i][2],path[i][1],ttn_size[2],ttn_size[1])
        append!(map2d,[Int(site_idx)])
    end
    return map2d
end

function construct_first_layer(mpo,mapping,net)
    bEnvironment = Vector{TTN.ITensor}(undef, 1) 

    bEnvironment = map(eachindex(net,1)) do pp
        chdnds = TTN.child_nodes(net, (1,pp))
        map(1:TTN.number_of_child_nodes(net, (1,pp))) do nn
          mpo[TTN.inverse_mapping(mapping)[chdnds[nn][2]]]
        end
    end
    
    return bEnvironment
end

function wf_mpo(wf, net, op_ins)
    ampo = TTN.OpSum()
    lat = TTN.physical_lattice(net)
    for pci in keys(wf)
        pt = to_indices(wf, (pci,))

        plin = TTN.linear_ind(lat, pt)
        ampo += (wf[pci], op_ins, plin)
    end
    
    mapping = ttn_2d_mapping(size(lat))
    
    # build MPO out of OpSum
    idx_lat = map(mapping) do pos
        TTN.hilbertspace(lat[pos])
    end
    mpo = TTN.MPO(ampo,idx_lat)
    
    rez_data = construct_first_layer(mpo,mapping,net)
    
    remapped_mpo::Vector{TTN.ITensor} = []
    for i in 1:size(rez_data,1)
        for j in 1:size(rez_data[i],1)
            append!(remapped_mpo,[rez_data[i][j]])
        end
    end
    return remapped_mpo
    
end

function initialize_ttn(ttn,maxdim,particle_count; kwargs...)
	particle_type = get(kwargs, :part_type, "Boson")
	if particle_type == "Fermion"
		creation = "Cdag"
	elseif particle_type == "Boson"
		creation = "Adag"
    else
        error("Unknown particle type: $particle_type. Supported types are 'Fermion' and 'Boson'.")
	end
	
    ttn0_net = TTN.network(ttn)
    ttn0_lat = TTN.physical_lattice(ttn0_net)

	phys_edge_length,virt_edge_length = size(ttn0_lat)
	site_count = TTN.number_of_sites(ttn0_net)
	wf_coefs = create_wavefunction(Float64,size(ttn0_lat))
	for i in 1:particle_count
		ttn = parton_application!(ttn,wf_coefs,creation;maxdim=maxdim)
	end
	return ttn
end
