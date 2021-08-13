export haar_unitary, KAK1, to_circuit, to_circuit_XYX


#Using the algorithm given by Medrazzi, generate a Haar-random unitary U(N)

function haar_unitary(N::Int)
    Z = (randn(N,N) + 1im*randn(N,N)) / sqrt(2.0)
    F = qr(Z)
    D = Diagonal(F.R)
    Λ = D / abs.(D)
    U = F.Q * Λ * F.Q
    return UnitaryGate(U)
end

struct KAKDecomposition <: QuantumGate
    global_phase::ComplexF64
    left_gates::Array{UnitaryGate, 1}
    right_gates::Array{UnitaryGate, 1}
    interaction_angles::Array{Float64, 1}
end

function UnitaryGate(K::KAKDecomposition)
    A = kron(K.left_gates...)
    B = kron(K.right_gates...)
    U = Ud(K.interaction_angles...)
    return K.global_phase * A * U * B
end

"""
Given a matrix X in SO(4), this function decomposes it into matrices 
A, B in SU(2) such that X = M' (A⊗B) M

Code copied from Cirq: 
https://github.com/quantumlib/Cirq/blob/master/cirq/linalg/decompositions.py
"""
function decompose_product_gate(X::Matrix{T}; 
    tol=sqrt(eps(Float64)), debug=false) where T <: Number
    
    M = [[1. 0. 0. 1im]; #Magic basis
         [0. 1im 1. 0.]; 
         [0. 1im -1. 0.];
         [1. 0. 0. -1im];] / sqrt(2.0);
    
    @assert norm(X*X'-I) < tol "Input matrix U is not orthogonal!"
    @assert abs(det(X)-1) < tol "Input matrix U is not special!"
    @assert size(X)[1] == 4 && size(X)[1] == size(X)[2] "Matrix must be 4x4!"
    
    U = M*X*M'
    A = zeros(ComplexF64, (2,2))
    B = zeros(ComplexF64, (2,2))
    
    a11 = (U[1:2,1:2]*U[3:4,3:4]')[1,1]
    a12 = -(U[1:2,3:4]*U[3:4,1:2]')[1,1]
    a11a12 = (U[1:2,1:2]*U[1:2,3:4]')[1,1]
    
    A[1,1] = sqrt(a11)
    A[1,2] = sqrt(a12)
    
    #fix relative sign
    if abs(A[1,1]*A[1,2]' - a11a12) > tol
        A[1,2] = -A[1,2]
    end
    
    A[2,1] = -A[1,2]'
    A[2,2] = A[1,1]'
    
    if abs(A[1,1])>abs(A[1,2])
        B = U[1:2,1:2]/A[1,1]
    else
        B = U[1:2,3:4]/A[1,2]
    end
    
    err = norm(kron(A,B) - U)
    if err > tol
        @warn "Could not decompose SO(4) gate!"
    end
    
    if debug
        println("SO(4) to SU(2) error: ", err)
    end
    
    return A, B
end

"""
Check if a set of angles is in the Weyl chamber.

"""
function in_weyl_chamber(x::T, y::T, z::T) where T <: Real
    return (0 <= abs(z) <= abs(y) <= abs(x) <= π/4) && !(z ≈ -π/4)
end

"""
Canonicalizes the nonlocal part of a KAK decomposition 
A = exp(i⋅Σ kσₖₖ), k ∈ (x, y, z) such that it lies in the Weyl chamber:

0 ≤ |z| ≤ |y| ≤ |x| ≤ π/4
z ≠ -π/4

Code from Cirq's implementation: 
https://github.com/quantumlib/Cirq/blob/master/cirq/linalg/decompositions.py

See also:
B. Kraus and J. I. Cirac, "Optimal creation of entanglement 
using a two-qubit gate." Phys. Rev. A 63, 062309 (2001)

"""
function kak_canonicalize(x::T, y::T, z::T) where T <: Real
    
    phase = one(ComplexF64)
    Id = ComplexF64.(Matrix(I, (2,2)))
    left = [Id, Id]
    right = [Id, Id]
    v = [x, y, z]
   
    flippers = [ [[0. 1.]; [1. 0.]] * 1im,
                 [[0. -1im]; [1im 0.]] *1im,
                 [[1. 0.]; [0. -1.]] * 1im]
    
    swappers = [ [[1. -1im]; [1im -1]] * 1im/sqrt(2.),
                 [[1. 1.]; [1. -1.]] *1im/sqrt(2.),
                 [[0. 1-1im]; [1+1im 0.]] * 1im/sqrt(2.) ]
    
    function shift(k, step)
        v[k] += step * π/2.0
        phase *= (1im)^Float64(step)
        right[1] = (flippers[k]^(step%4)) * right[1]
        right[2] = (flippers[k]^(step%4)) * right[2]
    end
    
    other_index(k1,k2) = setdiff((1,2,3), (k1,k2))[1]
    
    function negate(k1, k2)
        v[k1] *= -1.
        v[k2] *= -1.
        phase *= -1.
        s = flippers[other_index(k1,k2)]
        left[2]  = left[2]*s
        right[2] = s*right[2]
    end
    
    function swap(k1, k2)
        v[k1], v[k2] = v[k2], v[k1]
        s = swappers[other_index(k1,k2)]
        left[1] = left[1]*s
        left[2] = left[2]*s
        right[1] = s*right[1]
        right[2] = s*right[2]
    end
    
    function canonical_shift(k)
        while v[k] <= -π/4
            shift(k, +1)
        end
        while v[k] > π/4
            shift(k, -1)
        end
    end
    
    function sort()
        if abs(v[1]) < abs(v[2])
            swap(1,2)
        end
        if abs(v[2]) < abs(v[3])
            swap(2,3)
        end
        if abs(v[1]) < abs(v[2])
            swap(1,2)
        end
    end
    
    canonical_shift(1)
    canonical_shift(2)
    canonical_shift(3)
    sort()
    if v[1] < 0
        negate(1,3)
    end
    if v[2] <0
       negate(2,3)
    end
    canonical_shift(2)
    
    @assert in_weyl_chamber(v...) "Something went wrong!"
    
    return KAKDecomposition(phase, [UnitaryGate(left[2]), UnitaryGate(left[1])], 
                            [UnitaryGate(right[2]), UnitaryGate(right[1])], v)
end 

""" K-A-K Decomposition of an arbitrary U(4) unitary.

Algorithm from:
R. Tucci. "An introduction to Cartan's KAK decomposition for QC Programmers." 
arXiv:quant-ph/0507171
"""
function KAK1(U0::Matrix{T}; tol = sqrt(eps(Float64)), debug=false, 
    force_gradient_descent=false) where T<:Complex

    M = [[1. 0. 0. 1im]; #Magic basis
         [0. 1im 1. 0.]; 
         [0. 1im -1. 0.];
         [1. 0. 0. -1im];] / sqrt(2.0);
    
    Γ = [[1 1 -1  1]; 
         [1 1  1  -1];
         [1 -1 -1 -1];
         [1 -1 1  1]];
    
    Xp = M'*U0*M
    XR = real.((Xp + conj(Xp))/2.0)
    XI = real.((Xp - conj(Xp))/2im)
    

    (U, V, dr, di, success) = joint_svd(XR, XI, tol=1e-13)
    
    #harder problem, attempt using gradient descent
    if (!success || force_gradient_descent)
        (U, V, dr, di, success) = joint_svd_gradient_descent(XR, XI, U, V)
    end
    
    if !success
        (U, V, dr, di, success) = 
            joint_svd_gradient_descent(XR, XI, nothing, nothing)
    end 
    
    if !success
        error("Was unable to find joint SVD of gate matrix real and 
            imaginary components.")
    end 
        
    QL = Matrix(U')
    QR = Matrix(V')
    
    Σ = QL'*Xp*QR
    θ = diag(angle.(Σ))
    
    k = Γ\θ
    
    #global phase
    δ = exp(1im*k[1])
    
    if debug
        printfmtln("Error in SVD: {:.4e}", norm(QL*Σ*QR' - Xp))
        printfmtln("Norm of LL†-I: {:.4e}", norm(QL*QL'-I))
        printfmtln("Norm of RR†-I: {:.4e}", norm(QR*QR'-I))
        printfmtln("Norm of ΣΣ†-I: {:.4e}", norm(Σ*Σ'-I))
        printfmtln("det QL: {:.4e}, det QR: {:.4e}", det(QL), det(QR))
    end
    
    A1,A0 = UnitaryGate.(decompose_product_gate(QL))
    B1,B0 = UnitaryGate.(decompose_product_gate(Matrix(QR')))
    
    if debug
        printfmtln("Error in A gates: {:.4e}", 
            norm((M'*kron(A1,A0)*M).U - QL))
        printfmtln("Error in B gates: {:.4e}", 
            norm((M'*kron(B1,B0)*M).U - QR'))   
        printfmtln("Nonlocal part error: {:.4e}", 
            norm(M*Σ*M' - δ*Ud(k[2:end]...).U))
        
        Ubefore = (A1⊗A0) * δ*Ud(k[2:end]...) * (B1⊗B0)
        printfmtln("Pre-caninicalization error: {:.4e}", norm(Ubefore.U - U0))
    end
    
    canonKAK = kak_canonicalize(k[2], k[3], k[4])
    
    B1 = canonKAK.right_gates[1] * B1
    B0 = canonKAK.right_gates[2] * B0
    A1 = A1 * canonKAK.left_gates[1]
    A0 = A0 * canonKAK.left_gates[2]
    
    final_decomp = KAKDecomposition(δ*canonKAK.global_phase, [A1, A0], 
        [B1, B0], canonKAK.interaction_angles)
    
    if debug
        U1 = UnitaryGate(canonKAK)
        U2 = Ud(k[2:end]...)
        U3 = UnitaryGate(final_decomp)
        
        println("Initial angles: ", k[2:end])
        println("Final angles: ", final_decomp.interaction_angles)
        
        printfmtln("Canonicaliztion error: {:.4e}", norm(U1 - U2))
        printfmtln("Output error to input: {:.4e}", norm(U3.U - U0))
    end
    
    return final_decomp
end

KAK1(G::UnitaryGate; tol=ϵ, debug=false, force_gradient_descent=false) = KAK1(G.U, tol=tol, debug=debug, force_gradient_descent=force_gradient_descent)

"""
Helper function to test that the KAK decomposition works 
(approximately, at least!)

Takes a unitary U and returns ‖KAK - U‖₂
"""
function test_KAK_decomp(U::UnitaryGate; gd=false, debug=false)
    return norm(UnitaryGate(KAK1(U, force_gradient_descent=gd, debug=debug)) - U)
end

"""
Convert a KAK-decomposed unitary to a circuit that contains 3 CNOT gates.
These CNOT gates are then broken apart into 3 CZ gates where we assume 
qubit 1 is the control.

Reference:
F. Vatan and C. Williams. "Optimal Quantum Circuits for General Two-Qubit Gates"
Phys. Rev. A 69, 032315 (2004)
"""
function to_circuit(Ut::UnitaryGate)

    K = KAK1(Ut)
    
    #for convenience
    Id = UnitaryGate(complex.([[1. 0.]; [0. 1.]]))
    H = UnitaryGate(complex.([[-1 1]; [1 1]]) / sqrt(2.0))
    
    Hcr = -1.0 * (0.00 + π/4) * Rz(π) ⊗ Rx(π) 
    Hzz = 0.0 * Rz(π) ⊗ Rz(π)

    CR = UnitaryGate(exp(-1im*(Hcr.U + Hzz.U)))    
        
    α, β, γ = K.interaction_angles
    
    T1 = Rz(2γ - π/2.)
    T2 = Ry(π/2. - 2α)
    T3 = Ry(2β - π/2)
    
    #Single qubit gates in the decomposition of N(α, β, γ)
    #So that we have -N1-CR-N2-CR-N3-CR-N4-
    N0 = [Rz(π/2), Id]
    
    N1 = [H*T2*Rz(-π/2), H*T1*Rx(π/2)]
    
    N2 = [T3*H*Rz(-π/2), H*Rx(π/2)]
    
    N3 = [Rz(-π/2), Rz(-π/2)*Rx(π/2)]
    
    L0 = [N0[1]*K.right_gates[1], N0[2]*K.right_gates[2]]
    L1 = N1
    L2 = N2
    L3 = [K.left_gates[1]*N3[1], K.left_gates[2]*N3[2]]
        
    #check that we got it right
    Ux = kron(L3...)*CR*kron(L2...)*CR*kron(L1...)*CR*kron(L0...)

    @assert abs(tr(Ux*Ut')) ≈ 4.0

    return [L0, :ZX90, L1, :ZX90, L2, :ZX90, L3]
    
end
