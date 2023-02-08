# TTNKit
Package of handling one- and two- dimensional Tree Tensor Networks based on TensorKit and ITensors


## TODO

---

### Features:

- Correlation functions (ITensors & TensorKit)
- Subspace expansion (TensorKit)
- Hamiltonians defined by sum over local terms as alternative to MPO's (TensorKit)

### Tests:

- Environments testing ->  none so far
- Subspace expansion
- Simple DMRG/TDVP test (TFI vs exact)?
- 2d mappings for MPO

### Aditional:

- More flexibility in choosing the lengths of the system -> allow for general dimensions
- QR/factorize wrapper to collapse nearly identical code (TDVP sweeps, for example)
     
### Examples:

- DMRG for one dimensional and two dimensional systems with and without QNS
- TDVP in one and two dimensional systems
- Preferable in ITensors and TensorKit
- If implemented, also for different Hamiltonian realizations
