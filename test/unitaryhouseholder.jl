using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryHouseholder, mulax, T_matrix
using FiniteDifferences
using Zygote: gradient



function HH_t(y::Vector)
	2 / (y' * y)
end

function HH_reflection(y::Vector)
	t = HH_t(y)
	I - t * y * y'
end

function HH_mul(Y::AbstractMatrix)
	U = I
	for i = 1:size(Y)[2]
		U = U * HH_reflection(Y[:, i])
	end
	U
end

@testset "Conversion of UnitaryHouseholder to matrix" begin
	Y = randn(5, 5)
	a = UnitaryHouseholder(Y)
	Y = LowerTriangular(Y)
	@test Matrix(a) ≈ HH_mul(Y)
	@test Matrix(transpose(a)) ≈ HH_mul(Y[:, end:-1:1])
end

@testset "Multiplication and T_update on multiplication" begin
	a = UnitaryHouseholder(5)
	x = randn(5, 5)
	@test a * x ≈ HH_mul(a.Y.Y) * x
	@test transpose(a) * x ≈ HH_mul(a.Y.Y)' * x
	@test x * a ≈ x * HH_mul(a.Y.Y)
	@test x * transpose(a) ≈ x * HH_mul(a.Y.Y)'
	a.Y .= LowerTriangular(rand(5, 5))
	x = randn(5, 5)
	@test a * x ≈ HH_mul(a.Y.Y) * x
	@test transpose(a) * x ≈ HH_mul(a.Y.Y)' * x
	@test x * a ≈ x * HH_mul(a.Y.Y)
	@test x * transpose(a) ≈ x * HH_mul(a.Y.Y)'
end

@testset "Test partial derivative of single reflection" begin
	y = randn(5)
	for i = 1:5
		δy = zeros(5)
		δy[i] = 10^(-6)
		num = (HH_reflection(y+δy)-HH_reflection(y))*10^6
		@test isapprox(num, Unitary.pdiff_reflect(y, i); atol = 10^(-5))
	end
end

@testset "Test partial derivative of UnitaryHouseholder" begin
	Y = LowerTriangular(randn(5, 5))
	for a = 1:5 #vector index
		for b = a:5 #vector element index
			δY = LowerTriangular(zeros(5, 5))
			δY[b, a] = 10^(-6)
			num = (HH_mul(Y+δY)-HH_mul(Y))*10^6
			num_tran = (HH_mul((Y+δY)[:, end:-1:1])-HH_mul(Y[:, end:-1:1]))*10^6
			@test isapprox(num, Unitary.pdiff(Y, Unitary.T_matrix(Y), false, a, b); atol = 10^(-4)) 
			@test isapprox(num_tran, Unitary.pdiff(Y, Unitary.T_matrix(Y)', true, a, b); atol = 10^(-4)) 
		end
	end
end

@testset "Test differential of UnitaryHouseholder" begin
	Y = LowerTriangular(randn(5, 5))
	δY = LowerTriangular(rand(5, 5))*10^(-6)
	num = (HH_mul(Y+δY)-HH_mul(Y))
	num_tran = (HH_mul((Y+δY)[:, end:-1:1])-HH_mul(Y[:, end:-1:1]))
	@test isapprox(num, Unitary.diff_U(Y, Unitary.T_matrix(Y), false, δY); atol = 10^(-4))
	@test isapprox(num_tran, Unitary.diff_U(Y, Unitary.T_matrix(Y)', true, δY); atol = 10^(-4))
end

@testset "Test gradient functions" begin
	cfdm = central_fdm(5, 1)
	Y = rand(5, 5)
	x = rand(5, 5)
	Δ = ones(size(Y*x))
	@test grad(cfdm, Y -> sum(UnitaryHouseholder(Y) * x), Y)[1] ≈
	Unitary.grad_mul_Y(LowerTriangular(Y), Unitary.T_matrix(LowerTriangular(Y)), false, x, Δ)
	@test grad(cfdm, Y -> sum(transpose(UnitaryHouseholder(Y)) * x), Y)[1] ≈
	Unitary.grad_mul_Y(LowerTriangular(Y), Unitary.T_matrix(LowerTriangular(Y))', true, x, Δ)
	@test grad(cfdm, x -> sum(UnitaryHouseholder(Y) * x), x)[1] ≈
	Unitary.grad_mul_x(LowerTriangular(Y), Unitary.T_matrix(LowerTriangular(Y)), x, Δ)
	@test grad(cfdm, x -> sum(transpose(UnitaryHouseholder(Y)) * x), x)[1] ≈
	Unitary.grad_mul_x(LowerTriangular(Y), Unitary.T_matrix(LowerTriangular(Y))', x, Δ)
end

@testset "Testing integration with Flux and updating T" begin
	Y = rand(5, 5)
	x = rand(5, 5)
	U = UnitaryHouseholder(Y)
	ps = Flux.params(U)
	cfdm = central_fdm(5, 1)
	@test gradient(() -> sum(sin.(U * x)), ps)[U.Y] ≈
	grad(cfdm, Y -> sum(sin.(UnitaryHouseholder(Y) * x)), Y)[1]
	@test gradient(x -> sum(sin.(U * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(UnitaryHouseholder(Y) * x)), x)[1]
	Y = rand(5, 5)
	x = rand(5, 5)
	U.Y.Y .= Y
	@test gradient(() -> sum(sin.(U * x)), ps)[U.Y] ≈
	grad(cfdm, Y -> sum(sin.(UnitaryHouseholder(Y) * x)), Y)[1]
	@test gradient(x -> sum(sin.(U * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(UnitaryHouseholder(Y) * x)), x)[1]
end
