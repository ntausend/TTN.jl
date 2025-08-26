# TTN.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14421855.svg)](https://zenodo.org/doi/10.5281/zenodo.14421855)

## Introduction

This is a package for working with tree tensor networks based on [ITensors.jl](https://github.com/ITensor/ITensors.jl) library. It provides basic tools for defining quantum states in arbitrary dimensions using a hierarchic structure without loops, called a tree. This includes the calculation of local observables, correlation functions, and entanglement entropy of bipartitions, as well as providing a toolbox for finding the ground state of local Hamiltonians and calculating the time evolution of quantum states with respect to local Hamiltonians.

Our code is compatible with all features provided by `ITensors.jl`. This includes abelian symmetries (CPU only) and GPU computations.


## Installation

The package is currently not registered. Please install via:

```julia
julia> using Pkg; Pkg.add(url="https://github.com/ntausend/TTN.jl.git")
```

## Examples

Here we provide basic examples on how to use our package. More involved examples involving a full time evolution or minimization of a Hamiltonian may be found in the folder `examples`.

### Spin wave function on a one dimensional chain of length $L$

Currently, only perfect binary trees are implemented. This requires $L = 2^{n_{\rm layer}}$ where $n_{\rm layer}$ is the number of layers contained in the binary tree.

```julia
using ITensors, ITensorMPS, TTN
nlayers = 4 # Number of layers in the tree

maxdims = 4 # maximal bond dimension of the random tensor network
# Creates a network of nodes with a local Hilbertspace of S=1/2 spins
# This is of type AbstractNetwork
net = BinaryChainNetwork(nlayers, "SpinHalf")
# Creates a random tree tensor network based on `net`.
ttn = RandomTreeTensorNetwork(net; maxdim = maxdims, elT = Float64)

# calculation of the local magnetization for every lattice point
zexpect = expect(ttn, "Z")
```

We can also use quantum numbers

```julia
using ITensors, ITensorMPS, TTN
nlayers = 4 # Number of layers in the tree
Lx = 2^nlayers
states = [isodd(j) ? "Up" : "Dn" for j in 1:Lx] # product state
# Creates a network of nodes with a local Hilbertspace of S=1/2 spins
net = BinaryChainNetwork(nlayers, "SpinHalf"; conserve_qns = true)
# Creates a tree tensor network based on `net` representing the product state `states`.
ttn = ProductTreeTensorNetwork(net states; elT = Float64)
# calculation of the local magnetization for every lattice point
zexpect = expect(ttn, "Z")
```

Each `AbstractNetwork` object `net` also provides information about the physical lattice of the system

```julia
# extract the physical lattice.
lat = physical_lattice(net)
```

which can be used to access the physical sites, indices etc.

```julia
foreach(eachindex(lat)) do j
    @info j
end
```

The index of the one dimensional system gets directly translated to the index returned by `eachindex` for the `Lattice` object. For two dimensional systems, the tuple `(x,y)` gets mapped onto a linearized index. The exact mapping is not unique and can varied by the user. The standard mapping is `j = x + Lx * y`.

### Spin wave function on a two dimensional rectangle of dimension $(L_x, L_y)$

We only allow two-dimensional systems with a total number of spins $L_x + L_y = 2^{n_{\rm layer}}$. Also, the current implementation requires that both $L_x$ and $L_y$ be powers of two, where $L_x$ must be greater than $L_y$. This is likely to change in the near future.

```julia
using ITensors, ITensorMPS, TTN
dims = (2^2,2^2) # dimensions of the rectangle.

maxdims = 4 # maximal bond dimension of the random tensor network
# Creates a network of nodes with a local Hilbertspace of S=1/2 spins
net = BinaryRectangularNetwork(dims, "SpinHalf")
# Creates a random tree tensor network based on `net`.
ttn = RandomTreeTensorNetwork(net; maxdim = maxdims, elT = Float64)

# calculation of the local magnetization for every lattice point
zexpect = expect(ttn, "Z")
```

### Defining a Hamiltonian

Defining a Hamiltonian is as easy as in `ITensors.jl` itself. Consider the case of a transverse field Ising model in two dimensions on a square lattice defined by the Hamiltonian
$$
H = J\sum_{\braket{j,k}}\sigma_{j}^x \sigma_{k}^x + g \sum_{j} \sigma_j^z
$$
where the sum is over all nearest neighbors in the square lattice for the $\sigma_j^x \sigma_k^x$ interaction, and over all lattice sites for the transverse field. We provide a function `nearest_neighbors` which returns a list of all nearest neighbors on the lattice. The TFI Hamiltonian can then be written as

```julia
function TFI(J, g, lattice; periodic = true)
	ampo = OpSum()
    for bond in TTN.nearest_neighbours(lattice, collect(eachindex(lattice)); periodic = periodic)
        b1 = coordinate(lat, first(bond))
        b2 = coordinate(lat, last(bond))
        ampo += J, "X", b1, "X", b2
    end
    for j in eachindex(lattice)
       ampo += g, "Z", coordinate(lat,j) 
    end
    return ampo
end
```

The first argument of the `nearest_neighbor` function is the physical lattice and the second argument is the specific mapping, here assumed to be the standard mapping as explained above.

To obtain a object which can be used for DMRG or TDVP, we now have to pass this `OpSum` object

```julia
using ITensors, ITensorMPS, TTN
dims = (2^2,2^2) # dimensions of the rectangle.
J,g = 1, 0.1
maxdims = 4 # maximal bond dimension of the random tensor network
# Creates a network of nodes with a local Hilbertspace of S=1/2 spins
net = BinaryRectangularNetwork(dims, "SpinHalf")

lat = physical_lattice(net)
ampo = TFI(J,g, lat; periodic = false)

H = TPO(ampo, lat)
```

which creates a tensor product object. Alternative, one can use the `Hamiltonian(ampo, lat)` function to create a matrix product operator.

### GPU

All objects defined in this package can be easily put on the GPU by wrapping each object with the `gpu` function. Currently only `CUDA.jl` is supported.

### Detailed Examples

- `examples/J1J2_DMRG/j1j2_dmrg.jl`: contains an example of minimizing the J1J2 model on a square lattice using DMRG


---

## Citation

We are happy if you decide to use our framework for your research. When citing our work, please use the following reference 

Tausendpfund, N., Rizzi, M., Krinitsin, W., & Schmitt, M. (2024). TTN.jl -- A tree tensor network library for calculating groundstates and solving time evolution (0.1). Zenodo. [https://doi.org/10.5281/zenodo.14421855](https://doi.org/10.5281/zenodo.14421855)

The BibTeX code for this reference is:

```bibtex
	@software{tausendpfund_2024_14421855,
	  author       = {Tausendpfund, Niklas and
	                  Rizzi, Matteo and
	                  Krinitsin, Wladislaw and
	                  Schmitt, Markus},
	  title        = {TTN.jl -- A tree tensor network library for
	                   calculating groundstates and solving time
	                   evolution
	                  },
	  month        = dec,
	  year         = 2024,
	  publisher    = {Zenodo},
	  version      = {0.1},
	  doi          = {10.5281/zenodo.14421855},
	  url          = {https://doi.org/10.5281/zenodo.14421855}
	}
```

## TODO's

If you think we miss a feature not listed below, feel free to write us/open a issue.

- Expectation values for global operators (aka $H^2$)
- Initialization using application to a product state (aka Patron initialization $\rightarrow$ nearly finished)
- Finish documentation
- Add tests for various features
- implement non-perfect trees
- Stable noise term in combination with subspace expansion
- Make abelian symmetries work properly on the GPU (current a problem in `ITensors.jl`)
- Implement a chached version for the GPU to handle larger bond dimensions on the GPU
