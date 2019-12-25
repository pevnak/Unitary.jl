using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryMatrix

@testset "Is transposed unitary matrix its inverse?" begin
	a = UnitaryMatrix([1])
	at = transpose(a);
	for x in [rand(2), rand(2,10)]
		@test at*(a*x) ≈ x
		@test a*(at*x) ≈ x
	end

	at = inv(a);
	for x in [rand(2), rand(2,10)]
		@test at*(a*x) ≈ x
		@test a*(at*x) ≈ x
	end

	@test inv(inv(a)) == a
end

@testset "Inversions of activation function" begin
	for f in [identity, selu, tanh, NNlib.σ, NNlib.leakyrelu]
		x = -10:1:10
		@test inv(f).(f.(x)) ≈ x
		@test inv(f).(f.(-x)) ≈ -x
		@test inv(inv(f)) == f
	end
end