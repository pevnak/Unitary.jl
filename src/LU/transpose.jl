function transposelu(m)
	m = deepcopy(Matrix(transpose(m)))
	for i = 1:size(m, 1)-1
		for j = i+1:size(m, 1)
			m[j, i] /= m[i, i]
			m[i, j] *= m[i, i]
		end
	end
	m
end

function transposeilu(m)
	m = deepcopy(Matrix(transpose(m)))
	for i = 2:size(m, 1)
		for j = 1:i-1
			m[j, i] *= m[i, i]
			m[i, j] /= m[i, i]
		end
	end
	m
end

function transpose(a::Union{lowup, inverted_lowup})
	trans(a)
end

@adjoint function transpose(a::Union{lowup, inverted_lowup})
	Zygote.pullback(trans, a)
end

function trans(a::lowup)
	m = transposelu(a.m)
	lowup(m, size(m, 1))
end

function trans(a::inverted_lowup)
	m = transposeilu(a.m)
	inverted_lowup(m, size(m, 1))
end

function ∇transposelu(Δ, m)
	Δloc = deepcopy(Matrix(transpose(Δ)))
	for i = 1:size(m, 1)
		for j = i+1:size(m, 1)
			Δloc[j, i] *= m[i, i]
			Δloc[i, j] /= m[i, i]
			Δloc[i, i] += Δ[i, j]*m[j, i] - Δ[j, i]*m[i, j]/m[i, i]^2
		end
	end
	(Δloc, )
end

function ∇transposeilu(Δ, m)
	Δloc = deepcopy(Matrix(transpose(Δ)))
	for i = 1:size(m, 1)
		for j = 1:i-1
			Δloc[j, i] /= m[i, i]
			Δloc[i, j] *= m[i, i]
			Δloc[i, i] += Δ[j, i]*m[i, j] - Δ[i, j]*m[j, i]/m[i, i]^2
		end
	end
	(Δloc, )
end

@adjoint function transposelu(m)
	transposelu(m), Δ -> ∇transposelu(Δ, m)
end

@adjoint function transposeilu(m)
	transposeilu(m), Δ -> ∇transposeilu(Δ, m)
end
