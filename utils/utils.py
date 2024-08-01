import configparser
import argparse
from json import dump, dumps
from datetime import datetime
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="True",
                        help="Specify whether to save data or not. Default is True.")

    args = parser.parse_args()

    parsed_args = {
        "save": args.save.lower() == "true"
    }

    return parsed_args


def is_unitary(m):
    return np.isclose(m.dot(m.conj().T), np.eye(m.shape[0]), atol=1e-1).sum() == m.shape[0] * m.shape[1]


def read_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    # Transformer
    NUM_LAYERS = config.getint('Transformer', 'NUM_LAYERS')
    NUM_HEADS = config.getint('Transformer', 'NUM_HEADS')
    D_MODEL = config.getint('Transformer', 'D_MODEL')
    D_FFN = config.getint('Transformer', 'D_FFN')
    DROPOUT_RATE = config.getfloat('Transformer', 'DROPOUT_RATE')
    TARGET_VOCAB_SIZE = config.getint('Transformer', 'TARGET_VOCAB_SIZE')

    # Quantum circuit
    N_QUBITS = config.getint('Quantum Circuit', 'N_QUBITS')
    POVM = config.get('Quantum Circuit', 'POVM')
    GATES = eval(config.get("Quantum Circuit", "GATES"))
    GATE_TYPES = eval(config.get("Quantum Circuit", "GATE_TYPES"))
    SITES = eval(config.get("Quantum Circuit", "SITES"))
    START_STATE = config.get("Quantum Circuit", "START_STATE")
    TARGET_STATE = config.get("Quantum Circuit", "TARGET_STATE")

    # Training
    N_DATASET = config.getint("Training", "N_DATASET")
    N_EVAL = config.getint("Training", "N_EVAL")
    EPOCHS = eval(config.get('Training', 'EPOCHS'))
    BATCH_SIZE = config.getint('Training', 'BATCH_SIZE')
    MC_BATCH_SIZE = config.getint('Training', "MC_BATCH_SIZE")
    LEARNING_RATE = eval(config.get("Training", "LEARNING_RATE"))
    ALPHA = eval(config.get("Training", "ALPHA"))
    BETA = eval(config.get("Training", "BETA"))
    GAMMA = eval(config.get("Training", "GAMMA"))
    DELTA = eval(config.get("Training", "DELTA"))
    LAMBDA_TARGET = eval(config.get("Training", "LAMBDA_TARGET"))
    VERBOSE = config.getboolean("Training", "VERBOSE")
    EXACT_SIM = config.getboolean("Training", "EXACT_SIM")
    L2_EPOCHS = config.getint("Training", "L2_EPOCHS")
    PHYS_EPOCHS = config.getint("Training", "PHYS_EPOCHS")
    SUBSYS_RESTR = eval(config.get("Training", "SUBSYS_RESTR"))

    # Adiabatic settings
    STEPS = eval(config.get("Adiabatic", "STEPS"))

    # Return a dictionary with the retrieved values
    config_values = {
        'NUM_LAYERS': NUM_LAYERS,
        'NUM_HEADS': NUM_HEADS,
        'D_MODEL': D_MODEL,
        'D_FFN': D_FFN,
        'DROPOUT_RATE': DROPOUT_RATE,
        "TARGET_VOCAB_SIZE": TARGET_VOCAB_SIZE,
        "N_QUBITS": N_QUBITS,
        "GATES": GATES,
        "GATE_TYPES": GATE_TYPES,
        "SITES": SITES,
        "START_STATE": START_STATE,
        "TARGET_STATE": TARGET_STATE,
        "POVM": POVM,
        "N_DATASET": N_DATASET,
        "N_EVAL": N_EVAL,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "MC_BATCH_SIZE": MC_BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "ALPHA": ALPHA,
        "BETA": BETA,
        "GAMMA": GAMMA,
        "DELTA": DELTA,
        "LAMBDA_TARGET": LAMBDA_TARGET,
        "VERBOSE": VERBOSE,
        "EXACT_SIM": EXACT_SIM,
        "L2_EPOCHS": L2_EPOCHS,
        "PHYS_EPOCHS": PHYS_EPOCHS,
        "SUBSYS_RESTR": SUBSYS_RESTR,
        "STEPS": STEPS,
    }

    return config_values


def print_init_msg(config):
    for key, value in config.items():
        print(f"{key}: {value}, {type(value)}")


if __name__ == "__main__":
    config = read_config("config_adiabatic.ini")
    print_init_msg(config)
