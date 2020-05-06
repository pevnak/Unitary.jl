Matrix(a::lowdup) = Matrixldu(a.m)
Matrixldu(m) = UnitLowerTriangular(m) *
              diagsp(m) *
              UnitUpperTriangular(m)
Matrix(a::inverted_lowdup) = Matrixildu(a.m)
Matrixildu(m) = UnitUpperTriangular(m) *
               diagsp(m) *
               UnitLowerTriangular(m)

function ∇Matrixldu(Δ, m)
	l = UnitLowerTriangular(m)
	u = UnitUpperTriangular(m)
	d = diagsp(m)
	out = zero(m)
	out += UnitLowerTriangular(Δ*(d*u)')
	out += UnitUpperTriangular((l*d)'*Δ)
	for i in 1:size(m, 1)
		out[i, i] = σ(m[i, i])*sum(Δ.*(l[:, i]*u[i, :]'))
	end
	(out, )
end


function ∇Matrixildu(Δ, m)
	l = UnitLowerTriangular(m)
	u = UnitUpperTriangular(m)
	d = diagsp(m)
	out = zero(m)
	out += UnitLowerTriangular((u*d)'*Δ)
	out += UnitUpperTriangular(Δ*(d*l)')
	for i in 1:size(m, 1)
		out[i, i] = σ(m[i, i])*sum(Δ.*(u[:, i]*l[i, :]'))
	end
	(out, )
end

@adjoint function Matrixldu(m)
	return Matrixldu(m), Δ -> ∇Matrixldu(Δ, m)
end

@adjoint function Matrixildu(m)
	return Matrixildu(m), Δ -> ∇Matrixildu(Δ, m)
end

import Base: *
function *(a::Union{lowdup, inverted_lowdup}, b::Union{lowdup, inverted_lowdup})
	Matrix(a) * Matrix(b)
end
