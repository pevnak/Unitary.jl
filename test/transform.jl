using Unitary, Test, LinearAlgebra, Flux
using Unitary: Transform, ScaleShift
using FiniteDifferences

@testset "Can I invert Transform and its chain" begin
	for d in [1, 2, 3, 4]
		for decom in [:lu, :ldu, :svdgivens, :svdhouseholder]
			for m in [Transform(d, identity, decom), Transform(d, selu, decom), Chain(Transform(d, identity, decom), Transform(d, identity, decom)), Chain(Transform(d, selu, decom), Transform(d, selu, decom))]
				mi = inv(m)
				for x in [rand(d), rand(d,10), transpose(rand(10, d))]
					@test isapprox(mi(m(x)),  x, atol = 1e-2)
				end
			end
		end
	end

	for d in [1, 2, 3, 4]
		for m in [ScaleShift(d, identity), ScaleShift(d, selu), Chain(ScaleShift(d, identity), ScaleShift(d, identity)), Chain(ScaleShift(d, selu), ScaleShift(d, selu))]
			mi = inv(m)
			for x in [rand(d), rand(d,10), transpose(rand(10, d))]
				@test isapprox(mi(m(x)),  x, atol = 1e-3)
			end
		end
	end
end

@testset "testing the determinant" begin
	fdm = central_fdm(5, 1);
	x = randn(2)
	for decom in [:lu, :ldu, :svdgivens, :svdhouseholder]
		for m in [Transform(2, identity, decom), Transform(2, selu, decom), Chain(Transform(2, identity, decom), Transform(2, selu, decom))]
			@test isapprox(logabsdet(jacobian(fdm, m, x)[1])[1], m((x,0))[2][1], atol = 1e-3)
		end

		for m in [Transform(2, identity, decom), Transform(2, selu, decom), Chain(Transform(2, identity, decom), Transform(2, selu, decom))]
			@test m((x,0))[1] ≈ m(x)
		end
	end

	for m in [ScaleShift(2, identity), ScaleShift(2, selu), Chain(ScaleShift(2, identity), ScaleShift(2, selu))]
		@test isapprox(logabsdet(jacobian(fdm, m, x)[1])[1], m((x,0))[2][1], atol = 1e-3)
	end

	for m in [ScaleShift(2, identity), ScaleShift(2, selu), Chain(ScaleShift(2, identity), ScaleShift(2, selu))]
		@test m((x,0))[1] ≈ m(x)
	end
end

@testset "Testing calculation of the Jacobian" begin
	for decom in [:lu, :ldu, :svdgivens, :svdhouseholder]
		jacobian(f, x) = vcat([transpose(gradient(x -> f(x)[i], x)[1]) for i in 1:length(x)]...)
		for σ in [identity, selu]
			m = Transform(2,σ, decom)
			x = randn(2,1)
			@test isapprox(log(abs(det(jacobian(m, x)))), m((x,0))[2][1], atol = 1e-4)
		end
		for σ in [identity, selu]
			m = Chain(Transform(2,selu, decom), Transform(2,selu, decom), Transform(2,identity, decom))
			x = randn(2,1)
			@test isapprox(log(abs(det(jacobian(m, x)))), m((x,0))[2][1], atol = 1e-3)
		end
	end
end

@testset "Gradient of the likelihood" begin
	for decom in [:lu, :ldu, :svdgivens, :svdhouseholder]
		m = Transform(2,selu, decom)
		function lkl(m, x)
			x, l = m((x,0))
			exp.(- sum(x.^2, dims = 1)) .+ l
		end
		x = randn(2,10)
		fdm = central_fdm(5, 1)
		@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], grad(fdm, x -> sum(lkl(m, x)), x)[1], atol = 5e-4)
		m = Chain(Transform(2,selu, decom), Transform(2,selu, decom), Transform(2,identity, decom))
		@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], grad(fdm, x -> sum(lkl(m, x)), x)[1], atol = 5e-4)
	end
end
