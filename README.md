# TTNKit
Package of handling one- and two- dimensional Tree Tensor Networks based on TensorKit


## TODO

---

### Features:

- Correlation functions (ITensors & TensorKit)
- Subspace expansion (TensorKit)
- Hamiltonians defined by sum over local terms as alternative to MPO's (ITensors & TensorKit)

### Tests:

- Environments testing ->  none so far
- Subspace expansion
- Simple DMRG/TDVP test (TFI vs exact)?
- 2d mappings for MPO

### Aditional:

- Change nameing style for DMRG sweeper -> decouple the route from the rest of the sweeping handler
- add meta parameters (Krylov dimension, cutoff etc) to the DMRGSweeper. (Also in some way for TDVP ->  how to do generic?)
- Reworking Naming of TPOS to distingush more between MPO and other types. This involves:
  1) Finding a abstract `AbstractProjTPO` type and a nice API
  2) Rename the current `ProjectTensorProductOperator` to `ProjMatrixProductOperator` to resolve ambiguity since the
     current structure is only working with MPO's and will fail representing other structures.
  3) Define a function wich updates the environments according to a internal state referencing the old orthogonality centrum
     and the new orthogonality centrum of the ttn.
     
### Examples:

- DMRG for one dimensional and two dimensional systems with and without QNS
- TDVP in one and two dimensional systems
- Preferable in ITensors and TensorKit
- If implemented, also for different Hamiltonian realizations
