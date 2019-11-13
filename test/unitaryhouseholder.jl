using Unitary, Test, LinearAlgebra
using Unitary: UnitaryHouseholder



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
	for _ = 1:10
		Y = LowerTriangular(randn(5, 5))
		@test Matrix(UnitaryHouseholder(Y)) ≈ HH_mul(Y)
		@test Matrix(transpose(UnitaryHouseholder(Y))) ≈ HH_mul(Y[:, end:-1:1])
	end
end

@testset "Multiplication" begin
	for _ = 1:5
		Y = LowerTriangular(randn(5, 5))
		x = randn(5, 5)
		@test UnitaryHouseholder(Y) * x ≈ HH_mul(Y) * x
		@test transpose(UnitaryHouseholder(Y)) * x ≈ HH_mul(Y)' * x
		@test x * UnitaryHouseholder(Y) ≈ x * HH_mul(Y)
		@test x * transpose(UnitaryHouseholder(Y)) ≈ x * HH_mul(Y)'
	end
end

@testset "Test partial derivative of single reflection" begin
	for _ = 1:5
		y = randn(5)
		for i = 1:5
			δy = zeros(5)
			δy[i] = 10^(-6)
			num = (HH_reflection(y+δy)-HH_reflection(y))*10^6
			@test isapprox(num, Unitary.pdiff_reflect(y, i); atol = 10^(-5))
		end
	end
end

@testset "Test partial derivative of UnitaryHouseholder" begin
	for _ = 1:5
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
end

@testset "Test differential of UnitaryHouseholder" begin
	for _ = 1:10
		Y = LowerTriangular(randn(5, 5))
		δY = LowerTriangular(rand(5, 5))*10^(-6)
		num = (HH_mul(Y+δY)-HH_mul(Y))
		num_tran = (HH_mul((Y+δY)[:, end:-1:1])-HH_mul(Y[:, end:-1:1]))
		@test isapprox(num, Unitary.diff_U(Y, Unitary.T_matrix(Y), false, δY); atol = 10^(-4))
		@test isapprox(num_tran, Unitary.diff_U(Y, Unitary.T_matrix(Y)', true, δY); atol = 10^(-4))
	end
end
