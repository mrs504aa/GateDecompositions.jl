using Pkg
Pkg.activate(".")
using GateDecompositions

U = zeros(ComplexF64, 4, 4)
U[1, 1] = exp(im * 0)
U[2, 3] = exp(im * 0)
U[3, 2] = exp(im * 0)
U[4, 4] = exp(im * 0)

decomp_result = KAK1(U)
decomp_result.interaction_angles