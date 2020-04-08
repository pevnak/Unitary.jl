Base.Matrix(a::lowup) = Matrixlu(a.m)
Matrixlu(m) = UnitLowerTriangular(m) * UpperTriangular(m)
Base.Matrix(a::inverted_lowup) = Matrixilu(a.m)
Matrixilu(m) = UpperTriangular(m) * UnitLowerTriangular(m)

function ∇Matrixlu(Δ, m)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	(UnitLowerTriangular(Δ*u') + UpperTriangular(l'*Δ) - I, )
end

function ∇Matrixilu(Δ, m)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	(UnitLowerTriangular(u'*Δ) + UpperTriangular(Δ*l') - I, )
end

@adjoint function Matrixlu(m)
	return Matrixlu(m), Δ -> ∇Matrixlu(Δ, m)
end

@adjoint function Matrixilu(m)
	return Matrixilu(m), Δ -> ∇Matrixilu(Δ, m)
end

import Base: *
function *(a::Union{lowup, inverted_lowup}, b::Union{lowup, inverted_lowup})
	Matrix(a) * Matrix(b)
end
