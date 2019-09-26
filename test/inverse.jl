using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryMatrix, SVDDense

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

@testset "gradient with respect to the likelihood" begin 
	d = 2
	x = randn(2,3)
	model = Chain(SVDDense(d, selu), SVDDense(d, identity))
	ps = params(model)
	# gs = gradient(() -> sum(model((x,0.0))[2]), ps)
	gs = gradient(() -> sum(sum.(model((x,0.0)))), ps)
end

@testset "Inversions of activation function" begin
	for f in [identity, selu, tanh, NNlib.σ]
		x = -10:1:10
		@test inv(f).(f.(x)) ≈ x
		@test inv(f).(f.(-x)) ≈ -x
		@test inv(inv(f)) == f
	end
end

@testset "Can I invert SVDDense and its chain" begin
	for d in [2,3,4]
		for m in [SVDDense(d, identity), SVDDense(d, selu), Chain(SVDDense(d, identity), SVDDense(d, identity)), Chain(SVDDense(d, selu), SVDDense(d, selu))]
			mi = inv(m)
			@test inv(mi) == m
			for x in [rand(d), rand(d,10), transpose(rand(10, d))]
				@test isapprox(mi(m(x)),  x, atol = 1e-4)
			end
		end
	end
end

@testset "reducing number of operations in SVDDense" begin 
	@test length(Unitary.SVDDense(10, identity, maxn = 3).u.matrices) == 3
	@test length(Unitary.SVDDense(10, identity, maxn = 3).v.matrices) == 3
end