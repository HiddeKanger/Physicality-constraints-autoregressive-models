# Classical quantum simulation: physicality constraints for autoregressive models

This repository contains various scripts and configuration files for quantum computing simulations and training. The files are organized to facilitate adiabatic quantum computation and physical quantum circuit simulations.

## File Structure

### Root Directory

- **config_adiabatic_GHZ.ini**: Configuration file for adiabatic simulation of the GHZ state.
- **config_adiabatic_GRAPH.ini**: Configuration file for adiabatic simulation of graph states.
- **embedding.py**: Embedding part of transformer architecture.
- **simulate_exactly.py**: Exact simulation of quantum circuits or states for metrics.
- **train_circuit_physical.py**: Training a circuit with physicality constraints.
- **train_gate_physical.py**: Training a gate with physicality constraints.
- **transformer.py**: Transformer architectures.

### utils Directory

- **MPS.py**: Matrix Product State (MPS) calculations.
- **POVM.py**: POVM related calculations.
- **ncon.py**: Tensor network contraction function.
- **plotting.py**: Generate plots.
- **povm_generation.py**: Generate random POVM.
- **slicetf.py**: Tensor slicing and transformations.
- **utils.py**: General utility functions.

## Running simulations

Enter details of simulation in the specific config.ini file. Config file is interpreted, accepts python code for easy of use. 

```bash
python train_gate_physical.py --save True/False
```
