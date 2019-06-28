using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryMatrix, _∇mulax, _mulax, _mulatx, _mulxa, _mulxat
using Flux.Tracker: Params

@testset "Testing multiplication and transposed" begin
	a = UnitaryMatrix([1])
	ad = Matrix(a)

	for x in [randn(2),rand(2,10)]
		@test a * x ≈ ad * x
		@test transpose(a) * x ≈ transpose(ad) * x
		@test transpose(x) * transpose(a) ≈ transpose(x) * transpose(ad)
		@test transpose(x) * a ≈ transpose(x) * ad
	end

	for x in [rand(1,2), rand(10, 2)];
		@test x * a ≈ x * ad
		@test x * transpose(a) ≈ x * transpose(ad)
		@test transpose(a) * transpose(x) ≈ transpose(ad) * transpose(x)
		@test a * transpose(x) ≈ ad * transpose(x)
	end
end


@testset "Testing integration with Flux" begin
	for x in [randn(2), randn(2, 10)]
		θ = [1.0]
		pθ = param(θ)
		a = UnitaryMatrix(pθ)
		px = param(x)
		ps = Params(params(a))
		push!(ps, px)

		#testing gradient of a * x with respect to x and parameters of a
		grads = Flux.Tracker.gradient(() -> sum(sin.(a * px)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, Flux.Tracker.ngradient(x -> sum(sin.(Flux.data(a) * x)), x)[1], atol = 1e-6)
		@test isapprox(∇θ, Flux.Tracker.ngradient(θ -> sum(sin.(_mulax(θ, x))), θ)[1], atol = 1e-6)

		#testing gradient of transpose(a) * x with respect to x and parameters of a
		at = transpose(a)
		grads = Flux.Tracker.gradient(() -> sum(sin.(at * px)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, Flux.Tracker.ngradient(x -> sum(sin.(Flux.data(at) * x)), x)[1], atol = 1e-6)
		@test isapprox(∇θ, Flux.Tracker.ngradient(θ -> sum(sin.(_mulatx(θ, x))), θ)[1], atol = 1e-6)
	end

	for x in [rand(10, 2),rand(1, 2)]
		px = param(x)
		ps = Params(params(a))
		push!(ps, px)
		#testing gradient of a * x with respect to x and parameters of a
		grads = Flux.Tracker.gradient(() -> sum(sin.(px * a)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, Flux.Tracker.ngradient(x -> sum(sin.(x * Flux.data(a) )), x)[1], atol = 1e-6)
		@test isapprox(∇θ, Flux.Tracker.ngradient(θ -> sum(sin.(_mulxa(x, θ))), θ)[1], atol = 1e-6)

		#testing gradient of transpose(a) * x with respect to x and parameters of a
		at = transpose(a);
		grads = Flux.Tracker.gradient(() -> sum(sin.(px * at)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, Flux.Tracker.ngradient(x -> sum(sin.(x * Flux.data(at) )), x)[1], atol = 1e-6)
		@test isapprox(∇θ, Flux.Tracker.ngradient(θ -> sum(sin.(_mulxat(x, θ))), θ)[1], atol = 1e-6)
	end
end

@testset "Jacobian" begin 
	a = UnitaryMatrix([1.0])
	x = randn(2, 1)
	Flux.Tracker.jacobian(x -> a * x, x)
end

