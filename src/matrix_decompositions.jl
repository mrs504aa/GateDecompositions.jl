import Optim
export joint_svd, joint_svd_gradient_descent

"""
Calculate the joint SVD of two real matrices using gradient descent on the 
manifold St(n,n)⊗St(n,n).  This is pretty slow... and not particularly reliable

Reference:

G. Hori. "A general framework for SVD flows and joint SVD flows" 
IEEE ICASSP'03 (2003)

"""
function joint_svd_gradient_descent(A::Matrix{T}, B::Matrix{T},
    U0::Union{Matrix{T},Nothing}, V0::Union{Matrix{T},Nothing};
    tol=sqrt(eps(T))) where {T<:Real}

    (N, M) = size(A)
    (n, nn) = size(B)

    D = N * M

    to_vector(U, V) = cat(reshape(U, :, 1), reshape(U, :, 1), dims=1)
    ϕ(X) = sum(abs.(diag(X)) .^ 2)
    function to_matrices(X)
        U = reshape(X[1:D], N, M)
        V = reshape(X[D+1:end], N, M)
        return U, V
    end
    function f(X)
        U, V = to_matrices(X)
        return -(ϕ(U' * A * V) + ϕ(U' * B * V))
    end
    U0 = nothing
    V0 = nothing
    if U0 == nothing
        U0 = Float64.(Matrix(I, (n, nn)))
    end
    if V0 == nothing
        V0 = Float64.(Matrix(I, (n, nn)))
    end
    X0 = to_vector(U0, V0)
    #manifold over which to optimize: Steifel manifold of NxN orthogonal matrices
    #Julia is amazing
    manif = Optim.ProductManifold(Optim.Stiefel(), Optim.Stiefel(), (N, M), (N, M))
    result = Optim.optimize(f, X0, Optim.GradientDescent(manifold=manif))

    U, V = to_matrices(Optim.minimizer(result))

    Λa = U' * A * V
    Λb = U' * B * V

    cost(X, Y) = offdiag(X) + offdiag(Y)
    start_cost = cost(A, B)
    final_cost = cost(Λa, Λb)

    success = true
    if (!Optim.converged(result) || (final_cost > tol))
        @warn "Was unable to find a simultaneous SVD using gradient descent: 
                   start cost function: $(start_cost), final: $(final_cost)."
        success = false
    end

    U = Matrix(U') #convention shuffling
    V = Matrix(V')

    return (U, V, Diagonal(Λa), Diagonal(Λb), success)
end

"""
Calculatesd the approximate joint singular value decomposition of two real 
matrices A, B such that: 

A = U' Λa V and B = U' Λb V 

Algorithm is from:

J. Miao, G. Cheng, Y. Cai and J. Xia. "Approximate Joint Singular Value 
Decomposition Based on Givens-Like Rotation"
IEEE Signal Processing Letter, 25(5) 620-624 (2018).

"""
function joint_svd(A::Matrix{T}, B::Matrix{T}; tol=sqrt(eps(T)),
    maxiter=1000) where {T<:Real}

    (n, nn) = size(A)

    U = Float64.(Matrix(I, (n, nn)))
    V = Float64.(Matrix(I, (n, nn)))

    Λa = A
    Λb = B


    function MM(X, Y, i, j)
        α1 = X[i, i] * X[j, j] + Y[i, i] * Y[j, j]
        α2 = X[i, i] * X[j, i] + Y[i, i] * Y[j, i]
        α3 = X[j, j] * X[i, j] + Y[j, j] * Y[i, j]
        α4 = X[i, i] * X[i, j] + Y[i, i] * Y[i, j]
        α5 = X[j, j] * X[j, i] + Y[j, j] * Y[j, i]
        β = X[i, i]^2 + X[j, j]^2 + Y[i, i]^2 + Y[j, j]^2
        m = [[β -2 * α1]; [-2 * α1 β]]
        f = [α2 - α3; α4 - α5]
        (x, y) = qr(m, Val(true)) \ f
        return (x, y)
    end

    function R(x, i, j)
        λ = 1 / sqrt(1 + x^2)
        r = Float64.(Matrix(I, (n, nn)))
        r[i, i] = λ
        r[j, j] = λ
        r[i, j] = λ * x
        r[j, i] = -λ * x
        return r
    end

    cost(X, Y) = offdiag(X) + offdiag(Y)

    start_cost = cost(A, B)


    target = tol * (norm(A, 2) + norm(B, 2))
    k = 1

    while ((cost(Λa, Λb) > tol) && (k < maxiter))
        for i = 1:n-1
            for j = i+1:n
                (x, y) = MM(Λa, Λb, i, j)
                R1 = R(x, i, j)
                R2 = R(y, i, j)
                U = R1 * U
                V = R2 * V
                Λa = R1 * Λa * R2'
                Λb = R1 * Λb * R2'
                k += 1
            end
        end
    end

    final_cost = cost(Λa, Λb)

    success = true
    if k >= maxiter
        @warn "Was unable to find a simultaneous SVD: start cost function: 
                    $(start_cost), final: $(final_cost)."
        success = false
    end

    return (U, V, Diagonal(Λa), Diagonal(Λb), success)
end

