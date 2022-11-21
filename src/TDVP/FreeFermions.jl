N = 4

function creation_op(j::Int)
    sigma_p = [0 1; 0 0]
    sigma_z = [1 0; 0 -1]

    println(sigma_p)
    println(sigma_z)
    return vcat(fill(sigma_z, j-1).*(-1), sigma_p)
end


function annihilation_op(j::Int)
    sigma_m = [[0, 0], [1, 0]]
    sigma_z = [[1, 0], [0, -1]]

    return vcat(fill(sigma_z, j-1).*(-1), sigma_m)
end

function occupation(j::Int)
    return creation_op(j)*annihilation_op(j)
end


states = fill([1/sqrt(2), 1/sqrt(2)], N)
# states = fill([1.; 0.], N)
fermStates = map(states) do s
    return 0.5*[1 1; 1 1]*s
end

println(states)
println(fermStates)


function fermHamiltonian()
    return
end
