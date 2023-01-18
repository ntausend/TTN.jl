
# todo include abstract model handling?
@with_kw struct TransverseFieldIsing
    #boundary_conditions <: BC # somehow encode if periodic or open ?
    J::Float64 = 1 # NN Interaction strength
    g::Float64 # Transverse Field interaction
end

function Hamiltonian(md::TransverseFieldIsing, lat::AbstractLattice{1})
    J = md.J
    g = md.g
    
    if sectortype(lat) == Trivial
        sx = [0 1; 1  0]
        sz = [1 0; 0 -1]

        sp = ComplexSpace(size(sx, 1))

        Sx = TensorMap(sx, sp, sp)
        Sz = TensorMap(sz, sp, sp)


        ham = MPOHamiltonian(LocalOperator(J * Sx ⊗ Sx, (1, 2)) +
                             LocalOperator(g * Sz, (1,)))

        #hamdat = Array{Union{Missing,typeof(sx)},3}(missing,1,3,3)
        #hamdat[1,1,1] = id;
        #hamdat[1,end,end] = id;
        #hamdat[1,1,2] = J*sx;
        #hamdat[1,2,end] = sx;
        #hamdat[1,1,end] = g*sz;

        #ham = MPOHamiltonian(hamdat);

    else
        sp = Rep[ℤ₂](0 => 1, 1 => 1)

        sz = TensorMap(zeros, ComplexF64, sp, sp)
        blocks(sz)[ℤ₂(1)] .= 1
        blocks(sz)[ℤ₂(0)] .= -1

        sx = TensorMap(ones, ComplexF64, sp * Rep[ℤ₂](1 => 1), sp)
        @tensor nn[-1 -2; -3 -4] := sx[-1 1; -3] * conj(sx[-4 1; -2])

        ham = MPOHamiltonian(J * nn) + MPOHamiltonian(g * sz)
    end
    return MPOWrapper(lat, ham)
end