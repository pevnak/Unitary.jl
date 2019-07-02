using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryMatrix, SVDDense

@testset "Testing that transposed unitary matrix is its inverse" begin
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

@testset "Testing that I can inverse SVDDense" begin
	m = SVDDense(identity)
	mi = inv(m)
	@test inv(mi) == m

	for x in [rand(2), rand(2,10), transpose(rand(10, 2))]
		@test Flux.data(mi(m(x))) ≈ x
	end
end
