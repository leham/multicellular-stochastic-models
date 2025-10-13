# Generate heatmaps of the pseudo steady-state mean for the positive feedback loop over a 2D slice of (η, σ_b) parameter space.
# NOTE: some parameters must be readjusted when using different grid sizes and/or σ_u values.
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
σ_u = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 0.1
# Gene switching on (binding) rate
σ_b = 0.03
# Translation rate (in the unbound state)
ρ_u = 0.0
# Translation rate ρ_b (in the bound state)
ρ_b = 60.0
# Decay rate
δ = 1.0
# Diffusion rate for the protein (movement from one cell to another)
η = 0.05
# Create parameter vector
# NOTE: order is fixed
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]


# multithreaded code to mean protein counts over time in (η, σ_b) space
function get_mean_pss_η_vs_σ_b(niters, pset, ntrajectories; ntasks=nthreads())
    lk = ReentrantLock()
    chnl_buf, chnl_ps, chnl_config, chnl_helper = init_sim_utils_FL(n, m, ps, ntasks)
    
    all_means = tmap(pset; ntasks) do (η_value, σ_b_value)
        res = 0.0
        buf = take!(chnl_buf)
        _ps = take!(chnl_ps) 
        config = take!(chnl_config)
        helpers = take!(chnl_helper)

        # update η and σ_b in the parameter vector
        _ps[5] = η_value
        _ps[2] = σ_b_value
        
        t = @elapsed for _ in 1:ntrajectories
            init_cells_FL!(config)
            #init_cells_FL_small!(config) #NOTE: use this instead for 10 × 10 grid
            for _ in 1:niters
                run_iter_FL!(config, buf, _ps, tau, helpers...)
            end
            res += get_mean_protein_number_FL(config)
        end

        res /= ntrajectories
        lock(lk) do
            println("η = $η_value, σ_b = $σ_b_value, t = $t s")
        end

        put!(chnl_buf, buf)
        put!(chnl_ps, _ps)
        put!(chnl_config, config)
        put!(chnl_helper, helpers)
        res
    end

    return all_means
end

# Setup for the original 100k niter simulation 
η_values = 0:0.002:0.1
σ_b_values = 0:0.002:0.1
pset = collect(Iterators.product(η_values, σ_b_values))
ntrajectories = 10  # Number of trajectories
niters = 10^5  # Number of tau leaping iterations per simulation trajectory

# NOTE: 
# η_values = σ_b_values = 0:0.02:1.0 # used for σ_u = 10 case
# - 
# ntrajectories = 50 # used for the case of 10 × 10 grid size
# -
# The heatmap in Figure 4(D) is constructed from three separate blocks:
# 1. η_values = 0:0.0005:0.015; σ_b_values = 0:0.0005:0.1 
# 2. η_values = 0.0155:0.0005:0.1; σ_b_values = 0:0.0005:0.04
# 3. η_values = 0.0175:0.0025:0.1; σ_b_values = 0.042:0.002:0.1

# Setup for the follow-up longer simulations over a vector of selected (η, σ_b) parameter pairs
#pset = load(datapath*"rerun_1M_sigma_u_$(σ_u[1])_pset.jld2", "pset")
#niters = 10^6

#threadinfo(; slurm=true, masks=true)
#println(getcpuids())
#println("no double occupancies: ", length(unique(getcpuids())) == length(getcpuids()))
#println("in order: ", issorted(getcpuids()))
#println("----------------------------------------------------------------------")
println("Number of Julia threads: $(nthreads())")
println("System size: $n × $m")
println("System parameters:")
@show σ_u
@show ρ_u
@show ρ_b
@show δ
@show n_snapshots
@show niter_per_snapshot
println("Simulation over $(length(pset)) parameter pairs")
println("Run $ntrajectories trajectories up to $niters iterations")
println("Simulation time = ", n_iters * tau)
println("----------------------------------------------------------------------")
flush(stdout)

@time all_means = get_mean_pss_η_vs_σ_b(niters, pset, ntrajectories)
# NOTE: change filename according to the specific case
#@save "means_eta_vs_sigma_b_sigma_u_$(σ_u[1])_single_block_100k.jld2" η_values σ_b_values all_means