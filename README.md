# multicellular-stochastic-models

This repository contains Julia code for the paper: &nbsp; L. Ham, M.G. Jackson, A. Sukys, and M.P.H Stumpf, "Intercellular signaling drives robust cell fate and patterning in multicellular systems" (2025).

The code is used to to simulate both single-cell and 2D spatial stochastic models that capture fine-scale gene regulatory mechanisms driving cell fate decision, exploring how local signalling and regulatory feedback shape multicellular dynamics and spatial organisation.

### Structure

- `analysis` - scripts used to reproduce the analysis and figures in the paper.
- `src` - main code for building stochastic gene models used throughout the analysis.
- `scripts` - Julia scripts used for longer multithreaded simulations.
- `envs` - Julia environments loading the relevant packages.

More details about specific files can be found within each folder.