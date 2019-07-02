using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryMatrix, SVDDense, invselu

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
	for f in [identity, selu]
		@test inv(f)(f(1)) ≈ 1
		@test inv(f)(f(-1)) ≈ -1
		@test inv(inv(f)) == f
	end
end

@testset "Can I invert SVDDense and its chain" begin
	for m in [SVDDense(identity), SVDDense(selu), Chain(SVDDense(selu), SVDDense(selu), SVDDense(identity))]
		mi = inv(m)
		@test inv(mi) == m

		for x in [rand(2), rand(2,10), transpose(rand(10, 2))]
			@test Flux.data(mi(m(x))) ≈ x
		end
	end
end
