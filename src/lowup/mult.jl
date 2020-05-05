mulaxlu(m, x) = UnitLowerTriangular(m) * (UpperTriangular(m) * x)
mulaxilu(m, x) = UpperTriangular(m) * (UnitLowerTriangular(m) * x)

mulxalu(m, x) = x * UnitLowerTriangular(m) * UpperTriangular(m)
mulxailu(m, x) = x * UpperTriangular(m) * UnitLowerTriangular(m)

import Base: *
function *(a::lowup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulaxlu(a.m, x)
end
function *(x::AbstractMatVec, a::lowup)
	@assert size(x, 2) == a.n
	mulxalu(a.m, x)
end
function *(a::inverted_lowup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulaxilu(a.m, x)
end
function *(x::AbstractMatVec, a::inverted_lowup)
	@assert size(x, 2) == a.n
	mulxailu(a.m, x)
end

function ∇mulaxlu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	Δloc = l'*Δ
	@inbounds for i = 1:a
		for j = 1:i
			for k = 1:b
				∇m[j, i] += x[i, k] * Δloc[j, k]
			end
		end
	end
	Δloc = u'*Δloc
	x = u*x
	@inbounds for i = 1:a-1
		for j = i+1:a
			for k = 1:b
				∇m[j, i] += x[i, k] * Δ[j, k]
			end
		end
	end
	(∇m, Δloc)
end

function ∇mulaxilu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	Δloc = u'*Δ
	@inbounds for i = 1:a-1
		for j = i+1:a
			for k = 1:b
				∇m[j, i] += x[i, k] * Δloc[j, k]
			end
		end
	end
	Δloc = l'*Δloc
	x = l*x
	@inbounds for i = 1:a
		for j = 1:i
			for k = 1:b
				∇m[j, i] += x[i, k] * Δ[j, k]
			end
		end
	end
	(∇m, Δloc)
end

function ∇mulxalu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	Δloc = Δ*u'
	@inbounds for i = 1:b-1
		for j = i+1:b
			for k = 1:a
				∇m[j, i] += x[k, j] * Δloc[k, i]
			end
		end
	end
	Δloc = Δloc*l'
	x = x*l
	@inbounds for i = 1:b
		for j = 1:i
			for k = 1:a
				∇m[j, i] += x[k, j] * Δ[k, i]
			end
		end
	end
	(∇m, Δloc)
end

function ∇mulxailu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	Δloc = Δ*l'
	@inbounds for i = 1:b
		for j = 1:i
			for k = 1:a
				∇m[j, i] += x[k, j] * Δloc[k, i]
			end
		end
	end
	Δloc = Δloc*u'
	x = x*u
	@inbounds for i = 1:b-1
		for j = i+1:b
			for k = 1:a
				∇m[j, i] += x[k, j] * Δ[k, i]
			end
		end
	end
	(∇m, Δloc)
end

@adjoint function mulaxlu(m, x)
	o = mulaxlu(m, x)
	return o, Δ -> ∇mulaxlu(Δ, m, x)
end

@adjoint function mulxalu(m, x)
	o = mulxalu(m, x)
	return o, Δ -> ∇mulxalu(Δ, m, x)
end

@adjoint function mulaxilu(m, x)
	o = mulaxilu(m, x)
	return o, Δ -> ∇mulaxilu(Δ, m, x)
end

@adjoint function mulxailu(m, x)
	o = mulxailu(m, x)
	return o, Δ -> ∇mulxailu(Δ, m, x)
end
