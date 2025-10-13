# Generate heatmaps of the pseudo steady-state mean and Fano factor of the protein numbers for the negative feedback loop.
envpath = normpath(joinpath((@__DIR__, "../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../src/")))
datapath = normpath(joinpath((@__DIR__, "../data/")))
using Pkg; Pkg.activate(envpath)
include(srcpath*"plotting.jl")
include(srcpath*"feedback_loop.jl")

using JLD2
using OhMyThreads: tmap
using Base.Threads: nthreads
using LinearAlgebra
#using ThreadPinning
#BLAS.set_num_threads(1)
#pinthreads(:affinitymask)

# Tau leaping
tau = 0.01
# Dimensions of the 2D tissue
n = m = 32
# Gene switching off (unbinding) rate
σ_u = 0.1
# Gene switching on (binding) rate
σ_b = 0.03
# Translation rate (in the unbound state)
ρ_u = 100.0
# Translation rate ρ_b (in the bound state)
ρ_b = 0.0
# Decay rate
δ = 1.0
# Diffusion rate for the protein (movement from one cell to another)
η = 0.05
# Create parameter vector
# NOTE: order is fixed
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]

n_snapshots = 400
niter_per_snapshot = 500

# compute variance over all cells on the grid given the mean μ
get_var_protein_number_FL(cells::AbstractMatrix, μ::Real) = mapreduce(x -> (x[1] - μ)^2, +, cells) / length(cells)

# multithreaded code to measure protein counts (mean, variance, and Fano factor per cell and trajectories) over time in (η, σ_b) space
function get_timeseries_η_vs_σ_b(n_snapshots, niter_per_snapshot, η_values, σ_b_values, ntrajectories; ntasks=nthreads())
    lk = ReentrantLock()
    chnl_buf, chnl_ps, chnl_config, chnl_helper = init_sim_utils_FL(n, m, ps, ntasks)
    chnl_mean_mtx = Channel{MMatrix{ntrajectories, n_snapshots, Float64, ntrajectories*n_snapshots}}(ntasks)
    chnl_var_mtx = Channel{MMatrix{ntrajectories, n_snapshots, Float64, ntrajectories*n_snapshots}}(ntasks)
    foreach(1:ntasks) do _
        put!(chnl_mean_mtx, MMatrix{ntrajectories, n_snapshots, Float64}(undef))
        put!(chnl_var_mtx, MMatrix{ntrajectories, n_snapshots, Float64}(undef))
    end
    pmat = collect(Iterators.product(η_values, σ_b_values))

    all_means = tmap(pmat; ntasks) do (η_value, σ_b_value)
        buf = take!(chnl_buf)
        _ps = take!(chnl_ps) 
        config = take!(chnl_config)
        helpers = take!(chnl_helper)
        mean_mtx = take!(chnl_mean_mtx)
        var_mtx = take!(chnl_var_mtx)
        
        # update η and σ_b in the parameter vector
        _ps[5] = η_value
        _ps[2] = σ_b_value
        
        t = @elapsed for i in 1:ntrajectories
            # all genes are in the unbound state initially
            reset_grid_FL!(config)
            for j in 1:n_snapshots
                for _ in 1:niter_per_snapshot
                    run_iter_FL!(config, buf, _ps, tau, helpers...)
                end
                mean_mtx[i, j] = get_mean_protein_number_FL(config)
                var_mtx[i, j] = get_var_protein_number_FL(config, mean_mtx[i, j])
            end
        end

        lock(lk) do
            println("η = $η_value, σ_b = $σ_b_value, t = $t s")
        end

        res = vec(mean(mean_mtx, dims=1)), vec(mean(var_mtx, dims=1)), vec(mean(var_mtx ./ mean_mtx, dims=1))
        put!(chnl_buf, buf)
        put!(chnl_ps, _ps)
        put!(chnl_config, config)
        put!(chnl_helper, helpers)
        put!(chnl_mean_mtx, mean_mtx)
        put!(chnl_var_mtx, var_mtx)
        res
    end

    return all_means
end

η_values = 0:0.02:1.0
σ_b_values = 0:0.02:1.0
n_snapshots = 100
niter_per_snapshot = 50
ntrajectories = 100
println("Simulation time = ", n_snapshots * niter_per_snapshot * tau)

res = @time get_timeseries_η_vs_σ_b(n_snapshots, niter_per_snapshot, η_values, σ_b_values, ntrajectories)
@save datapath*"mean_timeseries/timeseries_NFL.jld2" η_values σ_b_values res