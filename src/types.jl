import Base: -, *, convert
using LinearAlgebra

export UnitaryGate, tracenorm, tracedist, average_fidelity, dim, ⊗

abstract type QuantumGate end

struct UnitaryGate{T<:Complex} <: QuantumGate
    U::Matrix{T}
end

function UnitaryGate(U::T) where {T<:AbstractMatrix}
    if !isunitary(U)
        error("Gate must be unitary!")
    end
    (n, m) = size(U)
    if !(n == m)
        error("Gate matrix must be square!")
    end
    UnitaryGate(complex.(U))
end

Matrix(G::UnitaryGate{T}) where {T} = G.U

*(A::T, B::T) where {T<:UnitaryGate} = UnitaryGate(A.U * B.U)
*(A::T, B::AbstractMatrix) where {T<:UnitaryGate} = UnitaryGate(A.U * B)
*(A::AbstractMatrix, B::T) where {T<:UnitaryGate} = UnitaryGate(A * B.U)
*(A::Tg, x::Tx) where {Tg<:UnitaryGate,Tx<:Number} = UnitaryGate(A.U * x)
*(x::Tx, A::Tg) where {Tg<:UnitaryGate,Tx<:Number} = UnitaryGate(x * A.U)
-(A::T, B::T) where {T<:UnitaryGate} = UnitaryGate(A.U - B.U)

LinearAlgebra.kron(A::T, B::T) where {T<:UnitaryGate} = UnitaryGate(kron(A.U, B.U))
⊗(A::T, B::T) where {T<:UnitaryGate} = kron(A, B)

LinearAlgebra.adjoint(A::UnitaryGate) = UnitaryGate(Matrix(A.U'))
LinearAlgebra.tr(A::UnitaryGate) = tr(A.U)

dim(G::T) where {T<:UnitaryGate} = size(G.U)[1]

function dim(U::T)::Float64 where {T<:AbstractMatrix}
    if size(U)[1] == size(U)[2]
        return size(U)[1]
    else
        error("Can only return dimension of square matrix!")
    end
end

tracenorm(G::UnitaryGate) = tracenorm(Matrix(G))
tracedist(A::UnitaryGate, B::UnitaryGate) = tracedist(Matrix(A), Matrix(B))
frobdist(A::UnitaryGate, B::UnitaryGate) = frobdist(Matrix(A), Matrix(B))

LinearAlgebra.norm(G::UnitaryGate) = norm(G.U, 2)
LinearAlgebra.ishermitian(G::T; tol=ϵ) where {T<:UnitaryGate} = ishermitian_tol(Matrix(G), tol=ϵ)

"""
Returns the average gate fidelity between two gates.
"""
function average_fidelity(A::T, B::T) where {T<:AbstractMatrix}
    @assert dim(A) == dim(B)
    d = dim(A)
    return (d + abs(tr(A * B'))^2) / (d * (d + 1))
end

average_fidelity(A::T, B::T) where {T<:UnitaryGate} = average_fidelity(A.U, B.U)