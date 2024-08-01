import numpy as np
from time import time
import tensorflow as tf
from scipy.linalg import expm, logm
from simple_colors import *
from datetime import datetime
import os
from json import dump

from train_gate_physical import train_gate_
from utils.ncon import ncon
from utils.utils import is_unitary, read_config, get_args, print_init_msg
from utils.POVM import POVM
from utils.MPS import MPS
from transformer import Transformer
from simulate_exactly import ExactSimulator
from utils.plotting import plot_circuit_performance_eigvls_sub_mc


def save_weights(date, model, gate_idx, adiabatic_step, config):
    _dir = f"models/{date}"

    if not os.path.exists(_dir):
        os.mkdir(_dir)

    if not os.path.exists(f"{_dir}/config.json"):
        with open(f"{_dir}/config.json", "w") as file:
            dump(config, file, indent=4)

    filename = f"{_dir}/{gate_idx}_{adiabatic_step}.weights.h5"

    model.save_weights(filename)
    return filename


def save_performances(date, data, gate_idx, adiabatic_step):
    _dir = f"models/{date}"
    filename = f"{_dir}/performance_{gate_idx}_{adiabatic_step}.json"
    with open(filename, "w") as file:
        dump(data, file, indent=4)

    return filename


def load_weights(model, path):
    model.load_weights(path)
    return model


def discretize_matrix(U, steps):
    H = logm(U)
    dt = 1 / steps
    return expm(H * dt)


def train_circuit_(model, povm, mps, config, args, start_time, plot_performance=True, writer=None):
    GATES = config["GATES"]
    GATE_TYPES = config["GATE_TYPES"]
    SITES = config["SITES"]
    LEARNING_RATE = config["LEARNING_RATE"]
    EPOCHS = config["EPOCHS"]
    STEPS = config["STEPS"]

    exactSimulator = ExactSimulator(config=config, povm=povm)

    begin_training = time()

    for i, (gate, gate_type, site, epochs, steps) in enumerate(zip(
            GATES,
            GATE_TYPES,
            SITES,
            EPOCHS,
            STEPS)):
        gate_matrix = eval(f"povm.{gate.lower()}")
        if gate_type == 1:  # make 1 qubit gate a 2 qubit gate where the second is trivial
            gate_matrix = np.kron(gate_matrix, povm.I)
        gate_matrix = gate_matrix.reshape(4, 4)

        print("gate_matrix:", gate_matrix, gate_matrix.shape)

        dU = discretize_matrix(gate_matrix, steps=steps)
        assert is_unitary(dU), "GATE MATRIX NOT UNITARY, ABORTING!"
        print("dU:", dU, dU.shape)

        dU = dU.reshape(2, 2, 2, 2)
        dU_O_matrix = ncon((povm.M, povm.M, dU, povm.M, povm.M, povm.it, povm.it, np.conj(
            dU)), ([-1, 9, 1], [-2, 10, 2], [1, 2, 3, 4], [5, 3, 7], [6, 4, 8], [5, -3], [6, -4], [9, 10, 7, 8]))

        for j in range(steps):
            print(150 * "=")
            print(
                f"Initiating training of gate: {gate} @ {site}, adiabatic step number: {j + 1}/{steps}.")

            if config["EXACT_SIM"]:
                P_exact = exactSimulator.append_and_simulate(
                    gate_matrix=dU.reshape((4, 4)), sites=site)
            else:
                P_exact = None

            begin = time()
            if writer != None:
                with summary_writer.as_default():
                    performance = train_gate_(
                        model=model,
                        povm=povm,
                        mps=mps,
                        config=config,
                        gate=dU_O_matrix,
                        site=site,
                        lrs=LEARNING_RATE,
                        epochs=epochs,
                        gate_idx=i,
                        P_exact=P_exact,
                    )
            else:
                performance = train_gate_(
                    model=model,
                    povm=povm,
                    mps=mps,
                    config=config,
                    gate=dU_O_matrix,
                    site=site,
                    lrs=LEARNING_RATE,
                    epochs=epochs,
                    gate_idx=i,
                    P_exact=P_exact,
                )
            end = time()

            print(
                green(f"Learning gate {gate} took {(end - begin)/60:.2f} minutes."))

            if args["save"]:
                weights_filename = save_weights(
                    date=start_time,
                    model=model,
                    gate_idx=i,
                    adiabatic_step=j,
                    config=config
                )

                filename = save_performances(
                    date=start_time,
                    data=performance,
                    gate_idx=i,
                    adiabatic_step=j,
                )

    if plot_performance:
        plot_circuit_performance_eigvls_sub_mc(f"models/{start_time}")

    end_training = time()
    print(
        f"Overall training took: {(end_training - begin_training)/60:.2f} minutes.")


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    args = get_args()
    config = read_config("config_adiabatic.ini")
    print_init_msg(config)

    povm = POVM(POVM=config["POVM"], N_Qubits=config["N_QUBITS"])
    mps = MPS(POVM=config["POVM"], N_Qubits=config["N_QUBITS"],
              state=config["TARGET_STATE"])

    transformer = Transformer(
        num_layers=config["NUM_LAYERS"],
        d_model=config["D_MODEL"],
        num_heads=config["NUM_HEADS"],
        dff=config["D_FFN"],
        input_vocab_size=config["TARGET_VOCAB_SIZE"],  # same as target size
        target_vocab_size=config["TARGET_VOCAB_SIZE"],
        n_qubits=config["N_QUBITS"],
        dropout_rate=config["DROPOUT_RATE"],
        init_bias=povm.getinitialbias(config["START_STATE"]))

    # forces the model to be built
    input_tensor = tf.zeros([10, 1])
    out = transformer(inputs=input_tensor, training=False)

    print(transformer.summary())

    input("Press [ENTER] to continue...")
    summary_writer = None
    if args["save"]:
        _ = save_weights(
            date=now,
            model=transformer,
            gate_idx=0,
            adiabatic_step=0,
            config=config,
        )

        log_dir = f"models/{now}"
        summary_writer = tf.summary.create_file_writer(log_dir)

    log_dir = "logs/gradient_tracking"

    with tf.device("GPU:0"):
        train_circuit_(
            model=transformer,
            povm=povm,
            mps=mps,
            config=config,
            args=args,
            start_time=now,
            plot_performance=args["save"],
            writer=summary_writer
        )
