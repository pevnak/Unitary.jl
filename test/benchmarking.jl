using Unitary, Flux, BenchmarkTools
using Unitary: UnitaryGivens, lowup

x = randn(Float32, 50, 100)
xx = randn(Float32, 100, 50)

a = UnitaryGivens(50)
@btime a * x;		# 217.676 μs (4 allocations: 20.00 KiB)
@btime xx * a;		# 244.549 μs (4 allocations: 20.00 KiB)

ps = Flux.params(a)
@btime gradient(() -> sum(a * x), ps);	# 890.323 μs (58 allocations: 71.52 KiB)
@btime gradient(() -> sum(xx * a), ps);	# 473.158 μs (58 allocations: 71.52 KiB)

a = lowup(50)
@btime a * x;		# 217.676 μs (4 allocations: 20.00 KiB)
@btime xx * a;		# 244.549 μs (4 allocations: 20.00 KiB)

ps = Flux.params(a)
@btime gradient(() -> sum(a * x), ps);	# 890.323 μs (58 allocations: 71.52 KiB)
@btime gradient(() -> sum(xx * a), ps);	# 473.158 μs (58 allocations: 71.52 KiB)

