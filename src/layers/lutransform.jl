struct LUTransform{M, B, S}
	m::M
	b::B
	σ::S
end

Base.show(io::IO, a::LUTransform) = print(io, "LUTransform{$(size(a.m)), $(a.σ)}")

Flux.@functor LUTransform

"""
	LUTransform(n, σ)

	Transform layer with square weight matrix of dimension `n` parametrized in 
	LU or LDU decomposition.
	
	`σ` --- an invertible and transfer function, curently implemented `selu` and `identity`
"""
function LUTransform(n::Int, σ, decom = :ldu)
	n == 1 && return(ScaleShift(1, σ))
	if decom == :lu
		return(_lutransform(n, σ))
	elseif decom == :ldu
		return(_ldutransform(n, σ))
	else
		@error "unknown type of decompostion $decom"
	end
end


using LinearAlgebra

_lutransform(n::Int, σ) = LUTransform(lowup(Float32, n), 0.01f0.*randn(Float32,n), σ)
_ldutransform(n::Int, σ) = LUTransform(lowdup(Float32, n), 0.01f0.*randn(Float32,n), σ)

(a::LUTransform)(x::AbstractMatVec) = a.σ.((a.m * x) .+ a.b)

function (a::LUTransform)(xx::Tuple{A,B}) where {A,B}
	x, logdet = xx
	pre = (a.m * x) .+ a.b
	g = explicitgrad.(a.σ, pre)
	(a.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(a.m))
end

struct InvertedLUTransform{M, B, S}
	m::M
	b::B
	σ::S
end
Flux.@functor InvertedLUTransform

Base.inv(a::LUTransform) = InvertedLUTransform(inv(a.m), a.b, inv(a.σ))
Base.inv(a::InvertedLUTransform) = LUTransform(inv(a.m), a.b, inv(a.σ))

(a::InvertedLUTransform)(x::AbstractMatVec)  = (a.m * (a.σ.(x) .- a.b))
