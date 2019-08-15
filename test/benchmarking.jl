using Unitary, Test, Flux, BenchmarkTools, Statistics
using Unitary: UnitaryButterfly, Butterfly, randomgivenses


function benchmarksingle(d, l = 100)
	a = Butterfly(randn(div(d,2)), randomgivenses(d)[1]..., d)
	x = randn(d, l)
	ts = [@elapsed a*x for i in 1:100]
	println("d = ",d," median: ", median(ts), "  mean ", mean(ts))
end

function benchmarkall(d, l = 100)
	a = UnitaryButterfly(d)
	x = randn(d, l)
	ts = [@elapsed a*x for i in 1:100]
	println("d = ",d," median: ", median(ts), "  mean ", mean(ts))
end

# foreach(d -> benchmarksingle(2^d, 100), 2:9)
foreach(d -> benchmarkall(2^d, 100), 7:9)



d, l = 256, 100
a, x = UnitaryButterfly(d), randn(d, l)
@btime a * x;
# d = 256: 31.830 ms (1530 allocations: 50.48 MiB)

# 1 thread
# d = 128 median: 0.0039161935  mean 0.0048265533199999995
# d = 256 median: 0.036817067499999995  mean 0.03741094838
# d = 512 median: 0.138168415  mean 0.13870060212000002

# 2 threads
# d = 128 median: 0.0052984615  mean 0.013383723930000003
# d = 256 median: 0.0386333265  mean 0.039622609100000004
# d = 512 median: 0.137985156  mean 0.13912592216

# 4 threads
# d = 128 median: 0.005685899499999999  mean 0.01421938149
# d = 256 median: 0.0393440205  mean 0.040341021020000006
# d = 512 median: 0.1364668985  mean 0.13833719031000002

# 8 threads
# d = 128 median: 0.014580228  mean 0.019109436480000002
# d = 256 median: 0.09941320849999999  mean 0.12210359605000001
# d = 512 median: 0.2842547635  mean 0.3338837798
