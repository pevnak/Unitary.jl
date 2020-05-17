struct Transform{M, B, S}
	m::M
	b::B
	σ::S
end

Base.show(io::IO, a::Transform) = print(io, "Transform{$(size(a.m)), $(a.σ)}")

Flux.@functor Transform

"""
	Transform(n, σ, s)

	Transform layer with square weight matrix of dimension `n`
	
	`σ` --- an invertible and transfer function, curently implemented `selu` and `identity`

	`s` --- type of decomposition used
"""

function Transform(n::Int, σ, decom = :svdgivens)
	n == 1 && return(ScaleShift(1, σ))
	if decom == :lu
		_transform_lu(n, σ)
	elseif decom == :ldu
		_transform_ldu(n, σ)
	elseif decom == :svdgivens
		_transform_givens(n, σ)
	elseif decom == :svdhouseholder
		_transform_householder(n, σ)
	else
		@error "unknown type of decompostion $decom"
	end
end

_transform_lu(n::Int, σ) = Transform(lowup(Float32, n), 0.01f0.*randn(Float32,n), σ)
_transform_ldu(n::Int, σ) = Transform(lowdup(Float32, n), 0.01f0.*randn(Float32,n), σ)
_transform_givens(n::Int, σ) = Transform(Svd(n, :givens), 0.01f0.*randn(Float32,n), σ)
_transform_householder(n::Int, σ) = Transform(Svd(n, :householder), 0.01f0.*randn(Float32,n), σ)

(a::Transform)(x::AbstractMatVec) = a.σ.((a.m * x) .+ a.b)

function (a::Transform)(xx::Tuple{A,B}) where {A,B}
	x, logdet = xx
	pre = (a.m * x) .+ a.b
	g = explicitgrad.(a.σ, pre)
	(a.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(a.m))
end

struct InvertedTransform{M, B, S}
	m::M
	b::B
	σ::S
end
Flux.@functor InvertedTransform

Base.inv(a::Transform) = InvertedTransform(inv(a.m), a.b, inv(a.σ))
Base.inv(a::InvertedTransform) = Transform(inv(a.m), a.b, inv(a.σ))

(a::InvertedTransform)(x::AbstractMatVec)  = (a.m * (a.σ.(x) .- a.b))
