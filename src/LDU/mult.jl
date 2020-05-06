mulaxldu(m, x) = UnitLowerTriangular(m) *
                (diagsp(m) *
                (UnitUpperTriangular(m) *
                x))
mulaxildu(m, x) = UnitUpperTriangular(m) *
                 (diagsp(m) *
                 (UnitLowerTriangular(m) *
                 x))
 
mulxaldu(m, x) = x *
                UnitLowerTriangular(m) *
                diagsp(m) *
                UnitUpperTriangular(m)
mulxaildu(m, x) = x *
                 UnitUpperTriangular(m) *
                 diagsp(m) *
                 UnitLowerTriangular(m)

import Base: *
function *(a::lowdup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulaxldu(a.m, x)
end
function *(x::AbstractMatVec, a::lowdup)
	@assert size(x, 2) == a.n
	mulxaldu(a.m, x)
end
function *(a::inverted_lowdup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulaxildu(a.m, x)
end
function *(x::AbstractMatVec, a::inverted_lowdup)
	@assert size(x, 2) == a.n
	mulxaildu(a.m, x)
end

@inline function _lfill(a, b, ∇m, outer, inner)
	@inbounds for i = 1:a-1
		for j = i+1:a
			for k = 1:b
				∇m[j, i] += outer[k, i] * inner[k, j]
			end
		end
	end
end

@inline function _ufill(a, b, ∇m, outer, inner)
	@inbounds for i = 2:a
		for j = 1:i-1
			for k = 1:b
				∇m[j, i] += outer[k, i] * inner[k, j]
			end
		end
	end
end

@inline function _dfill(a, b, ∇m, m, x, Δ)
	@inbounds for i = 1:a
		for k = 1:b
			∇m[i, i] += σ(m[i, i]) * x[k, i] * Δ[k, i]
		end
	end
end

function ∇mulaxldu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	d = diagsp(m)
	u = UnitUpperTriangular(m)
	xloc = u*x
	Δloc = l'*Δ
	_dfill(a, b, ∇m, m, transpose(xloc), transpose(Δloc))
	Δloc = d*Δloc
	xloc = d*xloc
	_ufill(a, b, ∇m, transpose(x), transpose(Δloc))
	_lfill(a, b, ∇m, transpose(xloc), transpose(Δ))
	Δloc = u'*Δloc
	(∇m, Δloc)
end

function ∇mulaxildu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	d = diagsp(m)
	u = UnitUpperTriangular(m)
	xloc = l*x
	Δloc = u'*Δ
	_dfill(a, b, ∇m, m, transpose(xloc), transpose(Δloc))
	Δloc = d*Δloc
	xloc = d*xloc
	_lfill(a, b, ∇m, transpose(x), transpose(Δloc))
	_ufill(a, b, ∇m, transpose(xloc), transpose(Δ))
	Δloc = l'*Δloc
	(∇m, Δloc)
end

function ∇mulxaldu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	d = diagsp(m)
	u = UnitUpperTriangular(m)
	xloc = x*l
	Δloc = Δ*u'
	_dfill(b, a, ∇m, m, xloc, Δloc)
	Δloc = Δloc*d
	xloc = xloc*d
	_lfill(b, a, ∇m, Δloc, x)
	_ufill(b, a, ∇m, Δ, xloc)
	Δloc = Δloc*l'
	(∇m, Δloc)
end

function ∇mulxaildu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	d = diagsp(m)
	u = UnitUpperTriangular(m)
	xloc = x*u
	Δloc = Δ*l'
	_dfill(b, a, ∇m, m, xloc, Δloc)
	Δloc = Δloc*d
	xloc = xloc*d
	_ufill(b, a, ∇m, Δloc, x)
	_lfill(b, a, ∇m, Δ, xloc)
	Δloc = Δloc*u'
	(∇m, Δloc)
end

@adjoint function mulaxldu(m, x)
	return mulaxldu(m, x), Δ -> ∇mulaxldu(Δ, m, x)
end

@adjoint function mulxaldu(m, x)
	return mulxaldu(m, x), Δ -> ∇mulxaldu(Δ, m, x)
end

@adjoint function mulaxildu(m, x)
	return mulaxildu(m, x), Δ -> ∇mulaxildu(Δ, m, x)
end

@adjoint function mulxaildu(m, x)
	return mulxaildu(m, x), Δ -> ∇mulxaildu(Δ, m, x)
end
