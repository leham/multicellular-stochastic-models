envpath = normpath(joinpath((@__DIR__, "../envs/env1")))
srcpath = normpath(joinpath((@__DIR__, "../src/")))
using Pkg; Pkg.activate(envpath)
include(srcpath*"plotting.jl")

using DifferentialEquations
using LinearAlgebra
using SparseArrays
using JumpProcesses
using Catalyst
using StatsBase
using Distributions

# Construct a feedback loop model:
# negative feedback back when ρu > ρb 
# positive feedback back when ρu < ρb 
# NOTE: G bound is state 0 and G unbound is state 1
# NOTE: we consider positive feedback throughout

@parameters ρu ρb σu σb δ

rs = @reaction_network begin
	σb, G + P → 0
	σu * (1 - G), 0 ⇒ G + P
	ρu, G → G + P
	ρb * (1 - G), 0 ⇒ P
	δ, P → 0
end

# Initial conditions for [G, P]
u0 = [:G => 0, :P => 0]

# --------------------------------------------------
# Figure 3(B) 
# Explore bursty protein production and gene activity.

p = (ρu => 0, ρb => 10.0, σu => 0.08, σb => 0.002, δ => 1.0)
tspan = (0.0, 60.0)
dprob = DiscreteProblem(rs, u0, tspan, p)
jprob = JumpProblem(rs, dprob, Direct())
sol = solve(jprob, SSAStepper())

ts = sol.t
protein_trace = sol[2, :]
gene_trace = 1 .- sol[1, :] # inverting: G = 0 (ON) → 1, G = 1 (OFF) → 0

f = CairoMakie.Figure(size = (size_pt[1]*1.1, size_pt[2]*1.3), figure_padding = 2)
ax = Axis(f[1, 1], xlabel = "", ylabel = "Protein", 
          xticksvisible=true, xticklabelsvisible=true,
          yticksvisible=true, yticklabelsvisible=true,
          limits = ((0, 60.1), (0, nothing)))
CairoMakie.lines!(ax, ts, protein_trace, linewidth=1.5, color=red_col)
ax= Axis(f[2, 1], xlabel = "Time", ylabel = "Gene", 
            limits = ((0, 60.1), (-0.1, 1.1)),
            height=Relative(1/3))
CairoMakie.lines!(ax, ts, gene_trace, linewidth=1.5, color=purple_col)
rowgap!(f.layout, -15)
f

# --------------------------------------------------
# Figure 3(C) 
# Low unbinding (L) regime

# Figure 3(C) top
# Plot example protein trace
p = (ρu => 0, ρb => 100.0, σu => 0.1, σb => 0.01, δ => 1.0)
tspan = (0.0, 150.0)
jprob = remake(jprob, tspan=tspan, p=p)
sol = solve(jprob, SSAStepper())
ts = sol.t
protein_trace = sol[2, :]
gene_trace = 1 .- sol[1, :]

f = CairoMakie.Figure(size = (size_pt[1]*0.9, size_pt[2]*1.1), figure_padding = 2)
ax = Axis(f[1, 1], xlabel = "", ylabel = "Protein", 
          xticksvisible=true, xticklabelsvisible=true,
          yticksvisible=true, yticklabelsvisible=true,
          limits = ((0, 150.1), (-0.3, nothing)))
CairoMakie.lines!(ax, ts, protein_trace, linewidth=1.5, color=red_col)
ax= Axis(f[2, 1], xlabel = "Time", ylabel = "Gene", 
            limits = ((-0.1, 150.1), (-0.1, 1.1)),
            height=Relative(1/3))
CairoMakie.lines!(ax, ts, gene_trace, linewidth=1.5, color=purple_col)
rowgap!(f.layout, -15)
f

# Figure 3(C) bottom
# Verify that the burst count C follows the the geometric distributon derived in the paper

rhou, rhob, sigmau, sigmab, delta = last.(p)
geom_parameter = exp((-rhob/delta)*sigmab/(sigmab+delta))

tspan = (0.0, 1000.0) # longer timespan ensures that activity has died out (for this parameter set)
jprob = remake(jprob, tspan=tspan)

# count the number of bursts depending from the gene activity
function burstcount(gene_state)
	count = 0
	current = 0
	for i in eachindex(gene_state)
		if isone(gene_state[i]) && iszero(current)
            # gene switches off
			current = 1
        elseif iszero(gene_state[i]) && isone(current)
			# gene switches on
			current = 0
            # new burst
            count += 1
		end
	end
    # add 1 for the initial burst
	return count+1
end

# Generate a number of SSA trajectories
ensembleprob = EnsembleProblem(jprob)
ntrajectories = 1000
sol_SSA = solve(ensembleprob, SSAStepper(), trajectories=ntrajectories)
burstlife = [burstcount(sol[1, :]) for sol in sol_SSA]

# plot the SSA histogram and the theoretical distribution
f = CairoMakie.Figure(size = (size_pt[1]*0.7, size_pt[2]*0.7), figure_padding = 6)
ax = Axis(f[1, 1], xlabel = "Burst count", ylabel = "Probability",
          xticks=(1:4:20))

function stairpts(s)
    pts = s.plots[1].converted[1][]
    [p[1] for p in pts], [p[2] for p in pts]
end

nmax = 20
ws = fit(Histogram, burstlife, 1:nmax+1, closed=:left)
ws = normalize(ws, mode=:probability)
ws = ProbabilityWeights(ws.weights)
s = stairs!(ax, 0:nmax+1, vcat(0, ws, 0), 
            step=:center, color=(purple_col, 1), linewidth=0)
xs′, ys′ = stairpts(s)
band!(ax, xs′, 0*ys′, ys′, color=(purple_col, 1), label="Data")
ylims!(0, nothing)
xlims!(0.5, 13.5)

geom(x) = pdf(Geometric(geom_parameter)+1, x)
CairoMakie.lines!(ax, 1:60, geom, linewidth=1, color=red_col)
f

# --------------------------------------------------
# Figure 3(D) 
# Low unbinding (L) regime

# Figure 3(D) top
# Plot examples of Ephemeral (L1) and Functionally Stable (L2) behaviour
 
function get_burst_times(gene_state, ts)
    # get burst initiation times (when gene switches to the bound state)
	current = 0
    ts_gene = Float64[]
    for i in eachindex(gene_state)
		if isone(gene_state[i]) && iszero(current)
            # gene switches off
			current = 1
        elseif iszero(gene_state[i]) && isone(current)
			# gene switches on
			current = 0
            append!(ts_gene, ts[i])
		end
	end
	return ts_gene
end

# Ephemeral (L1)
p = (ρu => 0, ρb => 100.0, σu => 0.1, σb => 0.01, δ => 1)
tspan = (0.0, 100.0)
dprob = DiscreteProblem(rs, u0, tspan, p)
jprob = JumpProblem(rs, dprob, Direct())

sol = solve(jprob, SSAStepper())
ts = sol.t
protein_trace = sol[2, :]
ts_gene = @time get_burst_times(sol[1, :], ts) 

f = CairoMakie.Figure(size = (size_pt[1]*0.85, size_pt[2]*0.75), figure_padding = 2)
ax = Axis(f[1, 1], xlabel = "Time", ylabel = "Protein",
            limits = ((0, ts_gene[end]+0.1), (-0.2, 152)))
CairoMakie.lines!(ax, ts, protein_trace, linewidth=1.0, color=red_col) 
CairoMakie.vlines!(ax, ts_gene, linewidth=1.0, color=(purple_col, 0.5)) # burst start times
f

# Functionally stable (L2)
p = (ρu => 0, ρb => 100.0, σu => 0.1, σb => 0.12, δ => 1)
tspan = (0.0, 400.0)
dprob = DiscreteProblem(rs, u0, tspan, p)
jprob = JumpProblem(rs, dprob, Direct())
sol = solve(jprob, SSAStepper())
ts_gene = get_burst_times(sol[1, :], sol.t)

# extract the state vector only at specific times
# otherwise DifferentialEquations saves the solution at the timepoint of each reaction,
# whih can become memory-intensive if reaction rates are higher
dt = 0.01
ts = 0:dt:tspan[end]
protein_trace = sol(ts)[2, :]

f = CairoMakie.Figure(size = (size_pt[1]*0.85, size_pt[2]*0.75), figure_padding = 2)
ax = Axis(f[1, 1], xlabel = "Time", ylabel = "Protein",
            limits = ((0, ts[end]+0.1), (-0.2, 152)))
CairoMakie.lines!(ax, ts, protein_trace, linewidth=1.0, color=red_col)
CairoMakie.vlines!(ax, ts_gene, linewidth=1.0, color=(purple_col, 0.5))
f

# Figure 3(D) bottom
# Exponential scaling of the expected burst count with increasing σb: 
rhou, rhob, sigmau, sigmab, delta = last.(p)
sigmab_values = 0.001:0.001:0.2
get_mean_burst_count(sigmab) = log10(exp((rhob/delta)*sigmab/(sigmab+delta)))

f = CairoMakie.Figure(size = (size_pt[1]*0.75, size_pt[2]*0.75), 
                      figure_padding = 5)
ax = Axis(f[1, 1], xlabel = "σ_b", ylabel = "log(E(C))")
xlims!(ax, (-0.005, 0.205))
CairoMakie.lines!(ax, sigmab_values, get_mean_burst_count, linewidth=1.5, color=purple_col)
# highlight the points corresponding to the σb values we used above
CairoMakie.scatter!(ax, [0.01, 0.12], get_mean_burst_count.([0.01, 0.12]), markersize=5)
f

# --------------------------------------------------
# Figure 3(F) 
# High unbinding (H) regime
# Plot examples of Ephemeral (H1), Bimodal (H2), and Functionally Stable (H3) behaviour

function plot_survival!(ax::Axis, rs, u0, tspan, p;
                        ntrajectories = 5, 
                        t0 = 1e-4, tmax = last(tspan),
                        y0 = 0, ymax = 1,
                        npoints=1000)
    empty!(ax) # remove previous plots from the axis object
    
    # Create discrete and jump problems
    dprob = DiscreteProblem(rs, u0, tspan, p)
    jprob = JumpProblem(rs, dprob, Direct(), save_positions=(false, false))
    ensembleprob = EnsembleProblem(jprob)

    _t0 = t0 < 1e-3 ? 1e-3 : t0 # likely that no reactions occured before then ()
    sol = solve(ensembleprob, SSAStepper(), 
                saveat=logrange(_t0, tmax, npoints), 
                trajectories=ntrajectories)
    fraction_alive = [count(!iszero, proteins)/ntrajectories for proteins in eachrow(sol[2, 2:end, :])]
    prepend!(fraction_alive, [1.0])
    ts = sol[1].t
    ts[1] = t0 < _t0 ? t0 : ts[1]
    CairoMakie.lines!(ax, log10.(ts), fraction_alive, linewidth=1.0, color=purple_col)
    CairoMakie.ylims!(ax, y0-0.1, ymax+0.1) 
    CairoMakie.xlims!(ax, log10(t0), log10(tmax)) 
    CairoMakie.hlines!(ax, 0, color=(:gray, 0.5))

    return ax
end


function plot_trajectories!(ax::Axis, rs, u0, tspan, p;
                            ntrajectories = 5, 
                            t0 = 1e-4, tmax = last(tspan),
                            y0 = 1, ymax = nothing)
 
    empty!(ax) # remove previous plots from the axis object
    
    # Create discrete and jump problems
    dprob = DiscreteProblem(rs, u0, tspan, p)
    jprob = JumpProblem(rs, dprob, Direct())
    _col_mix(t) = col_mix(t, tmax)   # custom colour function
    ts_log_range = logrange(t0, tmax, 5000)

    # color each trajectory depending on its survival time
    cgradient = cgrad([purple_col, red_col])
    cmap(t) = cgradient[t/tmax]

    # --- Trajectories
    
    for _ in 1:ntrajectories
        sol = solve(jprob, SSAStepper())
        protein_counts = last.(sol.u)
        ts = sol.t
        
        if length(ts) > 5000
            # some trajectories can explode in memory usage (due to saving each reaction time)
            ts = ts_log_range
            protein_counts = last.(sol(ts).u)
        end
        
        CairoMakie.lines!(ax, log10.(ts), log10.(protein_counts);
                          color = (cmap(ts[end-1]), 0.75),
                          linewidth = 0.5)
    end
    
    CairoMakie.xlims!(ax, log10(t0), log10(tmax))
    CairoMakie.ylims!(ax, log10(y0), log10(ymax))

    return ax
end

tspan = (0.0, 200.0)
get_θ(p) = (p[2][2]*p[4][2])/(p[3][2]*p[5][2]) #check gain-loss threshold θ value
get_b(p) = p[2][2]/p[3][2] # check mean burst size 

# H1 behaviour
p1 = (ρu => 0, ρb => 10000.0, σu => 1000, σb => 0.01, δ => 1.0)
get_θ(p1) # θ < 1

# H2 behaviour
p2 = (ρu => 0, ρb => 10000.0, σu => 1000, σb => 0.11, δ => 1.0)
get_θ(p2) # θ > 1
get_b(p2) # low b

# H3 behaviour
p3 = (ρu => 0, ρb => 10000.0, σu => 100, σb => 1, δ => 1.0)
get_θ(p3) # θ > 1
get_b(p3) # high b

f = CairoMakie.Figure(size = (size_pt[1]*2.2, size_pt[2]*1.3), figure_padding = 5)

ax11 = Axis(f[1, 1], xlabel = "", ylabel = "Survival", height=Relative(0.5), xticklabelsvisible=false, yticks=(0:1, ["0", "1"]))
ax12 = Axis(f[1, 2], xlabel = "", ylabel = "", yticklabelsvisible=false, xticklabelsvisible=false, height=Relative(0.5))
ax13 = Axis(f[1, 3], xlabel = "", ylabel = "", yticklabelsvisible=false, xticklabelsvisible=false, height=Relative(0.5))

ax21 = Axis(f[2, 1], xlabel = "", ylabel = "log(protein)")
ax22 = Axis(f[2, 2], xlabel = "log(time)", ylabel = "", yticklabelsvisible=false)
ax23 = Axis(f[2, 3], xlabel = "", ylabel = "", yticklabelsvisible=false)

plot_survival!(ax11, rs, u0, tspan, p1; ntrajectories = 1000)
plot_survival!(ax12, rs, u0, tspan, p2; ntrajectories = 1000)
plot_survival!(ax13, rs, u0, tspan, p3; ntrajectories = 1000)

plot_trajectories!(ax21, rs, u0, tspan, p1; ntrajectories = 100, ymax = 15000)
plot_trajectories!(ax22, rs, u0, tspan, p2; ntrajectories = 100, ymax = 15000)
plot_trajectories!(ax23, rs, u0, tspan, p3; ntrajectories = 100, ymax = 15000)

colgap!(f.layout, 12)
rowgap!(f.layout, -5)
f