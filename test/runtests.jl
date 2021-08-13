using GateDecompositions
using Test

@testset "GateDecompositions.jl" begin
    
      U4 = haar_unitary(4)

      @test test_KAK_decomp(U4) ≈ 0.0  

end
