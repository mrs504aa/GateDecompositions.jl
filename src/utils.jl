using LinearAlgebra, Optim

export tracenorm, tracedist, frobdist, isunitary, ishermitian_tol, 
        isspecial, offdiag, jacobi_simultaneous_diag

ϵ = sqrt(eps(Float64))

"""
Compute the trace norm of a matrix ``A``
"""
function tracenorm(A::Matrix{T}) where T<:Number
    #Compute the trace norm of a matrix A
    if ishermitian(A)
        return sum(abs.(eigvals(A)))
    else
        return sum(svd(A).S)
    end
end
   
"""
Compute the trace distance between matrices ``A``, ``B``.
"""     
function tracedist(A::Matrix{T}, B::Matrix{T}) where T<:Number
    #Compute trace distance between matrices A, B
    return 0.5*tracenorm(A-B)
end

"""
Compute the Frobenius distance between matrices ``A``, ``B``.
"""
function frobdist(A::Matrix{T}, B::Matrix{T}) where T<:Number
    return norm(A-B, p=2)
end

"""
Check if a matrix is unitary up to a tolerance ``\epsilon``.
"""
function isunitary(A::Matrix{T}; tol=ϵ) where T <: Number
    return tracenorm(A'A - I) < tol
end

"""
Check if a matrix is Hermitian up to a tolerance ``\epsilon``.
"""
function ishermitian_tol(A::Matrix{T}; tol=ϵ) where T <: Number
    return tracedist(A, Matrix(A')) < tol
end

"""
Check if a matrix is special up to a tolerance ``\epsilon``.
"""
function isspecial(A::Matrix{T}; tol=ϵ) where T <: Number
    return isapprox(det(A), 1.0, rtol=tol)
end

"""
Sum of the off-diagonal elements of a matrix.
"""
function offdiag(A::Matrix{T}) where T <: Number
    return sum(tril(A,-1).^2) + sum(triu(A,1).^2)
end

"""
Function which implements the Jacobi algorithm for simultaneous (possibly approxiamate)
diagonalization of two real, symmetric matrices A, B. 

Algorithm is from:

V. Kuleshov, A. Chaganty, P. Liang. "Simultaneous diagonalization: the asymmetric, 
low-rank, and noisy settings" arXiv: 1051.06318 (2015)

J.-F. Cardoso, A. Soulomiac. "Jacobi angles for simultaneous diagonalization" 
SIAM J. Matrix Anal. Appl. 17(1), 161-164. (1996)

"""
function jacobi_simultaneous_diag(A::Matrix{T}, B::Matrix{T}; 
        tol=sqrt(eps(T)), maxiter=1000) where T<:Real
    
    @assert ishermitian_tol(A) && ishermitian_tol(B) "A, B must be symmetric!"
    
    (m,mm) = size(A)
    done = false
    
    U = Float64.(Matrix(I, (m,mm)))
    Λa = A
    Λb = B
    
    #Cost function for minimization of "off-diagonal-ness"
    #TODO: I was not able to succesfully implement the exact solution for the minimizer from the 
    #Cardoso paper...
    
    cost(X,Y) = offdiag(X)+offdiag(Y)
    
    start_cost = cost(A,B)
    
    function minims_lsq(X, Y, i, j)
        function f(x)
            g = LinearAlgebra.Givens(i,j, cos(x), sin(x))
            return cost(g'*X*g, g'*Y*g)
        end
        xopt = Optim.minimizer(optimize(f, -π/4, π/4)) #are these the correct limits?
        return (cos(xopt), sin(xopt))
    end
    
    target = tol*(norm(A,2)+norm(B,2))
    
    n = 1
    while ((cost(Λa,Λb) > tol) && (n < maxiter))
        for j=1:m-1
            for k=j+1:m
                (c, s) = minims_lsq(Λa, Λb, j, k)
                Γ = LinearAlgebra.Givens(j, k, c, s)
                U = U*Γ
                Λa = Γ'*Λa*Γ
                Λb = Γ'*Λb*Γ
            end
        end
    end
        
    final_cost = cost(Λa,Λb)

    success = true
        
    if ((n >= maxiter) || (final_cost > tol))
        @warn "Was unable to find a simultaneous diagonalization: start 
                cost function: $(start_cost), final: $(final_cost)."
        success = false
    end
    
    return (U, Diagonal(Λa), Diagonal(Λb), success)
end 
