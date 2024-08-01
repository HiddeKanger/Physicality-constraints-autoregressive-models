import utils.slicetf as slicetf
from utils.utils import read_config, print_init_msg
from utils.MPS import MPS
from utils.POVM import POVM
from utils.ncon import ncon

from transformer import Transformer
from scipy.linalg import sqrtm

from simple_colors import *

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tqdm import tqdm
from time import time
import itertools as it
from math import isnan
from collections import defaultdict

# import matplotlib.pyplot as plt

epsilon = 1e-32
np.set_printoptions(linewidth=np.inf)
# tf.experimental.numpy.experimental_enable_numpy_behavior()


@tf.function
def P_to_rho_explicit(P: tf.Tensor, povm: POVM) -> tf.Tensor:
    """
    - P: (4, 4) tf.Tensor, probability of distribution (sub)system
    - povm: POVM object

    returns:
    - rho: (4, 4) tf.Tensor, density matrix of (sub)system

    Explicit implementation of tensor contraction for gradient
    calculation.

    """
    # povm.it contains imaginary elements so we cast P to complex
    # eventhough it contains real numbers only!
    P = tf.cast(P, tf.complex128)
    T_inv_M = tf.einsum('lij,lk->ijk', povm.M, povm.it)

    rho = tf.einsum("ij, lmi, pnj -> lpmn", P, T_inv_M, T_inv_M)
    return rho


@tf.function
def P_to_rho_explicit_one(P, povm):
    P = tf.cast(P, tf.complex128)

    T_inv_M = tf.einsum('lij,lk->ijk', povm.M, povm.it)
    rho_test = tf.einsum("i, lmi -> lm", P, T_inv_M)
    return rho_test


def P_to_rho(P: np.array, povm: POVM, N_qubits: int) -> np.array:
    """
    - P: tf.Tensor, probability distribution of system
    - povm: POVM object
    - N_qubits: int, number of qubits in system

    returns:
    - rho: tf.Tensor, density matrix of system

    Perform Born rule through tensor contraction, supports any
    system size.
    """
    P = P.reshape((4,)*N_qubits)

    it = povm.it
    M = povm.M

    tensors = (P, ) + N_qubits * (it, ) + N_qubits * (M, )
    indices = (range(1, N_qubits + 1),)  # P tensor
    indices += tuple([[x + N_qubits, x]
                     for x in range(1, N_qubits + 1)])  # it tensors
    indices += tuple([[x + N_qubits, -x, -x - N_qubits]
                     for x in range(1, N_qubits + 1)])  # POVM

    return ncon(tensors, indices)


def rho_to_P(rho: np.array, povm: POVM, N_qubits: int) -> np.array:
    """
    - rho: tf.Tensor, density matrix of system
    - povm: POVM object
    - N_qubits: int, number of qubits in system

    returns:
    - P: tf.Tensor, probability distribution of system

    Performs inverse operation of the Born rule, supports any system size
    """
    rho = rho.reshape((2,)*2*N_qubits)
    M = povm.M

    tensors = (rho, ) + (M,) * N_qubits

    indices = (np.arange(2*N_qubits) + 1,) + \
        tuple([[-i, i+N_qubits, i] for i in range(1, N_qubits + 1)])
    return ncon(tensors, indices)


def get_eigvals_from_model(
        model: Transformer,
        samples: np.array,
        povm: POVM,
        config: dict,
        subsystem_size: int = None) -> np.array:
    """
    - model: Transformer object, transformer model
    - samples: np.array (N_samples, N_qubits), povm measurement samples
    - povm: POVM object
    - config: dict, contains all settings for current run
    - subsystem_size: int, size of subsystem to consider for eigenvalues

    returns:
    - eigvals: np.array, numpy array containing eigenvalues of subsystem rho

    Calculates the eigenvalues of a (sub)system of the model, given a set
    of samples from the model. Uses NumPy.
    """
    if subsystem_size == None:  # if no subsystem, use entire system
        subsystem_size = model.n_qubits

    # get probability of configuration of subsystem
    lP = logP(
        model,
        tf.cast(samples[:, :subsystem_size], tf.int32),
        config,
        training=False,
        subsystem_size=subsystem_size
    )

    lP_unique, _ = get_unique_configs(
        samples[:, :subsystem_size], lP.numpy(), lP.numpy())

    P_model = tf.math.exp(lP_unique)

    rho = P_to_rho(P_model.numpy(), povm=povm, N_qubits=subsystem_size)

    eigvals, _ = tf.linalg.eig(rho.reshape(
        2**subsystem_size, 2**subsystem_size))

    return eigvals.numpy().real, P_model


def get_batch(x: np.array, N_b: int, batch_size: int) -> np.array:
    """
    - x: np.array, array to get batches of
    - N_b: int, batch index
    - batch_size: int, size of the batches

    returns:
    - x: np.array, batch of x


    Given an array x, the index of the current batch N_b
    and the batch size return the current batch.
    """
    return x[N_b*batch_size: N_b*batch_size+batch_size]


def get_unique_configs(samples: np.array, P_model: np.array, P_target: np.array):
    """
    - samples: np.array (N_samples, N_qubits), povm measurement samples
    - P_model: np.array, samples probabilities of the model
    - P_target: np.array, target probability array

    returns:
    - tuple(np.array, np.array): both unique configurations

    Get unique configurations of P_model and P_target based on unique entries
    in the samples array. This should only be used if N_samples is large enough
    such that all configurations are are sampled. NOT EFFICIENT!
    """
    unique_rows, indices = np.unique(samples, axis=0, return_index=True)

    P_model_unique = P_model[indices]
    P_target_unique = P_target[indices]

    return np.round(P_target_unique, 10), np.round(P_model_unique, 10)


@tf.function
def assign_unique_value(array):
    # Get the shape of the input array
    N, M = tf.shape(array)[0], tf.shape(array)[1]

    # Compute the unique values for each row
    base = tf.constant(4, dtype=array.dtype)
    powers = tf.range(M, dtype=array.dtype)
    # Reverse powers for correct exponentiation order
    powers_reversed = tf.reverse(powers, axis=[0])

    # Compute the unique values
    unique_values = tf.reduce_sum(
        array * tf.pow(base, powers_reversed), axis=1)

    return tf.cast(unique_values, tf.int64)


@tf.function
def unique_configs(samples: np.array, P: np.array):
    """
    - samples: np.array (N_samples, 2), povm measurement samples
    - P: np.array, samples probabilities of the model

    returns:
    - sorted_P_unique: np.array, unique configurations

    Get unique configurations of P_model and P_target based on unique entries
    in the samples array. This should only be used if N_samples is large enough
    such that all configurations are are sampled. NOT EFFICIENT! Full tensorflow
    implementation for subsystem_size = 2.
    """
    # tensorflow only supports 1D unique so build unique identifier
    samples_identifier = assign_unique_value(array=samples)

    unique_vals, _ = tf.unique(samples_identifier)

    # apply argmax on all functions to get single idx of unique_vals
    res = tf.map_fn(
        lambda x: tf.argmax(
            tf.cast(tf.equal(samples_identifier, x), tf.int64)),
        unique_vals)

    unique_samples = tf.gather(samples, res)
    P_unique = tf.gather(P, res)

    return unique_samples,  P_unique


def print_epoch_perf(N_e: int, epochs: int, perf: dict, delta_t: float) -> None:
    """
    - N_e: int, epoch index
    - perf: dict, dictionary containing performance metrics for epoch
    - delta_t: float, time it took for calculating epoch

    Print epoch performance for monitoring in terminal.

    """
    _l = ['%.3e' % elem for elem in perf["losses"][-1]]
    eig_full = ['%.3e' % elem for elem in perf['eigvals_model'][-1]]

    form_eig_sub_mcs = ["  ".join(f"{val:.3f}" for val in array)
                        for array in perf['eigvals_model_sub_mc'][-1]]
    s = green(f"[{N_e}/{epochs}]: ")
    s += f"losses: {_l} "
    # s += f" | cFID: mps: {perf['cFID_MPS'][-1]:.3e}, t: {perf['cFID_target'][-1]:.3e}, e: {perf['cFID_exact'][-1]:.3e} "
    # s += f" | KL div: mps: {perf['KL_MPS'][-1]:.3e}, t: {perf['KL_target'][-1]:.3e}, e: {perf['KL_exact'][-1]:.3e} "
    # s += f" | qFID: mps: {perf['qFID_MPS'][-1]:.3e}, t: {perf['qFID_target'][-1]:.3e}, e: {perf['qFID_exact'][-1]:.3e} "

    s += f" | t: {perf['cFID_target'][-1]:.3e}, e: {perf['cFID_exact'][-1]:.3e} "
    s += f" | t: {perf['KL_target'][-1]:.3e}, e: {perf['KL_exact'][-1]:.3e} "
    s += f" | t: {perf['qFID_target'][-1]:.3e}, e: {perf['qFID_exact'][-1]:.3e} "

    s += f" | dt = {delta_t:.3f}"
    print(s)

    s = yellow("EIGVALS_FULL: ")
    s += f"{eig_full}"
    s += yellow(" EIGVALS_sub: ")
    s += " | ".join(form_eig_sub_mcs) + "\n \n"
    print(s)
    return


def calculate_metrics(P_model_unique, P_target_unique, P_exact, povm, config):
    if config["VERBOSE"]:
        print(green("P_model:"), P_model_unique,
              P_model_unique.sum(), P_model_unique.shape)
        print(cyan("P_target:"), P_target_unique,
              P_target_unique.sum(), P_target_unique.shape)
        print(magenta("P_exact:"), P_exact, P_exact.sum(), P_exact.shape)

    if np.abs(1 - P_target_unique.sum()) > 0.01 or np.abs(1 - P_model_unique.sum()) > 0.01:
        print(red("ERROR: PROBABILITIES DO NOT ADD UP TO 1!"))
        print(green("P_model:"), P_model_unique,
              P_model_unique.sum(), P_model_unique.shape)
        print(cyan("P_target:"), P_target_unique,
              P_target_unique.sum(), P_target_unique.shape)
        print(magenta("P_exact:"), P_exact, P_exact.sum(), P_exact.shape)

    KL_div_target = KL_div(P_target_unique, P_model_unique)
    KL_div_exact = KL_div(P_exact, P_model_unique)
    cFID_target = cFidelity(P_target_unique, P_model_unique)
    cFID_exact = cFidelity(P_exact, P_model_unique)
    qFID_target = qFidelity(P_target_unique, P_model_unique, povm, config)
    qFID_exact = qFidelity(P_exact, P_model_unique, povm, config)

    return KL_div_target, KL_div_exact, cFID_target, cFID_exact, qFID_target, qFID_exact


def assess_performance(perf,
                       model,
                       config,
                       povm,
                       mps,
                       losses,
                       batch_samples,
                       batch_lP_model,
                       batch_P_target,
                       P_exact,
                       subsys_eigvals_mc,):
    """
    Given a model, calculates a set of performances and returns those in a dict.
    """

    # GET MPS METRICS
    begin = time()
    samples, lP = sample(model, config["N_EVAL"], training=False)
    lP = np.reshape(lP, [-1, 1])
    end = time()
    print(f"sampling in performance assessment: {(end - begin):.3f} s")

    # begin = time()
    # cFID_MPS, cFID_MPS_std, KL_MPS, KL_MPS_std = mps.cFidelity(
    #     tf.cast(samples, dtype=tf.int64), lP)
    # qFID_MPS, qFID_MPS_std = mps.Fidelity(tf.cast(samples, dtype=tf.int64))
    # end = time()
    print(f"sampling in MPS: {(end - begin):.3f} s")

    # GET TARGET AND EXACT METRICS
    P_target_unique, P_model_unique_full = get_unique_configs(
        samples=batch_samples,
        P_model=tf.math.exp(batch_lP_model).numpy(),
        P_target=batch_P_target,
    )

    KL_target, KL_exact, cFID_target, cFID_exact, qFID_target, qFID_exact = calculate_metrics(
        P_model_unique_full, P_target_unique, P_exact, povm, config)

    eigvals_model = get_density_eig_vals(
        P_model_unique_full, povm, config["N_QUBITS"]).numpy()
    eigvals_target = get_density_eig_vals(
        P_target_unique, povm, config["N_QUBITS"]).numpy()
    eigvals_exact = get_density_eig_vals(
        P_exact, povm, config["N_QUBITS"]).numpy()

    subsys_eigvals_mc = [list(x.numpy().astype(float))
                         for x in subsys_eigvals_mc]

    perf["losses"].append(list(np.array(losses, dtype=float)))

    # perf["cFID_MPS"].append(float(cFID_MPS))
    # perf["cFID_MPS_std"].append(float(cFID_MPS_std))
    perf["cFID_target"].append(float(cFID_target))
    perf["cFID_exact"].append(float(cFID_exact))

    # perf["KL_MPS"].append(float(KL_MPS))
    # perf["KL_MPS_std"].append(float(KL_MPS_std))
    perf["KL_target"].append(float(KL_target))
    perf["KL_exact"].append(float(KL_exact))

    # perf["qFID_MPS"].append(float(qFID_MPS))
    # perf["qFID_MPS_std"].append(float(qFID_MPS_std))
    perf["qFID_target"].append(float(qFID_target))
    perf["qFID_exact"].append(float(qFID_exact))

    perf["eigvals_model"].append(list(eigvals_model.astype(float)))
    perf["eigvals_model_sub_mc"].append(list(subsys_eigvals_mc))
    perf["eigvals_target"].append(list(eigvals_target.astype(float)))
    perf["eigvals_exact"].append(list(eigvals_exact.astype(float)))

    perf["all_P_models"].append(list(P_model_unique_full.astype(float)))

    return perf, P_model_unique_full, P_target_unique


def qFidelity(P, Q, povm, config):
    rho1 = P_to_rho(P, povm=povm, N_qubits=config["N_QUBITS"])
    rho2 = P_to_rho(Q, povm=povm, N_qubits=config["N_QUBITS"])

    rho1 = rho1.reshape((2**config["N_QUBITS"], 2**config["N_QUBITS"]))
    rho2 = rho2.reshape((2**config["N_QUBITS"], 2**config["N_QUBITS"]))

    sqrt_rho1 = sqrtm(rho1)
    sqrt_term = np.matmul(sqrt_rho1, np.matmul(rho2, sqrt_rho1))
    fidelity = np.trace(sqrtm(sqrt_term))
    return fidelity.real


def cFidelity(p, q):
    cFID = np.sum(np.sqrt(np.array(p) * np.array(q)))**2
    if isnan(cFID):
        print("Error calculating classical fidelity")
        print("P:", p)
        print("Q:", q)
        return 1
    return cFID


def KL_div(P, Q):
    P = np.clip(P, a_min=epsilon, a_max=1)
    Q = np.clip(Q, a_min=epsilon, a_max=1)
    return -np.sum(P * np.log(Q/P))


def build_dataset(model, config, gate, site, training=False):
    samples, _ = sample(
        model, Nsamples=config["BATCH_SIZE"], training=training)

    # calculate P^e_{i + 1}, implements equation 2: P^e_{i + 1} = O^i * P_model_i
    P_target = Pev(
        model=model,
        samples=samples,
        gate=gate,
        site=site,
        target_vocab_size=config["TARGET_VOCAB_SIZE"],
        config=config)

    N_calls = int(config["N_DATASET"]/config["BATCH_SIZE"])

    for call in tqdm(range(N_calls - 1)):

        s, _ = sample(
            model, Nsamples=config["BATCH_SIZE"], training=training)

        # calculate P^e_{i + 1}, implements equation 2: P^e_{i + 1} = O^i * P_model_i
        p = Pev(
            model=model,
            samples=s,
            gate=gate,
            site=site,
            target_vocab_size=config["TARGET_VOCAB_SIZE"],
            config=config)

        samples = np.concatenate((samples, s))
        P_target = np.concatenate((P_target, p))

    return samples, P_target


@tf.function
def sample(model, Nsamples, training=False):
    """
    Sample the current model and calculate the log probability of the
    configuration, given the model.

    We do this by iteratively calling the transformer model.
    """
    output = tf.zeros([Nsamples, 1]) - 1  # -1 is the start token

    for i in range(model.n_qubits):  # O(n)
        predictions = model(inputs=output, training=training)

        if i == model.n_qubits - 1:
            # to compute the logP of the sampled config after sampling
            logP = tf.math.log(tf.nn.softmax(predictions))

        # get last predicted token
        predictions = predictions[:, -1:, :]

        predictions = tf.reshape(
            predictions, [-1, model.target_vocab_size])

        # sample the conditional distribution
        predicted_id = tf.random.categorical(predictions, 1)

        output = tf.concat(
            [output, tf.cast(predicted_id, dtype=tf.float32)], axis=1)

    # Cut the input of the initial call (-1's)
    output = tf.slice(output, [0, 1], [-1, -1])

    # one hot vector of the sample
    oh = tf.one_hot(tf.cast(output, dtype=tf.int64),
                    model.target_vocab_size)

    # the log probability of the configuration, sum of logs is product rule of probs.
    logP = tf.reduce_sum(logP*oh, [1, 2])
    return output, logP


@tf.function
def logP(model, samples, config, training=False, subsystem_size=None):
    """
    Calculate the log probability of the samples, given the model.
    This can be interpreted as the likelihood of the samples.

    This is done by getting P(a_k|a_<k) and picking the right ones
    out using the samples that were given using one hot encoding.
    """
    if subsystem_size == None:
        subsystem_size = config["N_QUBITS"]

    Nsamples = tf.shape(samples)[0]
    init = tf.zeros([Nsamples, 1]) - 1
    output = tf.concat([init, tf.cast(samples, dtype=tf.float32)], axis=1)

    output = output[:, 0:subsystem_size]

    predictions = model(inputs=output, training=training)

    # get the probabilities of the next element in the sequence
    logP = tf.math.log(tf.nn.softmax(predictions))

    # pick the right probabilities using one hot encoding of samples
    oh = tf.one_hot(tf.cast(samples, tf.int32), config["TARGET_VOCAB_SIZE"])

    logP = tf.reduce_sum(logP*oh, [1, 2])

    return logP  # , attention_weights


@tf.function
def Pev(model, samples, gate, site, target_vocab_size, config):
    """
    This function implements P^e_i+1 = O^i * P_model for 2 qubit gate.
    1-qubit gates can be implemented as OxI where the second qubit is acted trivially on
    """
    S = samples
    O = gate
    K = target_vocab_size
    # print(S, O, K)
    Ns = tf.shape(S)[0]
    N = tf.shape(S)[1]
    flipped = tf.reshape(tf.keras.backend.repeat(S, K**2), (Ns*K**2, N))
    # possible combinations of outcomes on 2 qubits
    a = tf.constant(
        np.array(list(it.product(range(K), repeat=2)), dtype=np.float32))
    s0 = flipped[:, site[0]]
    s1 = flipped[:, site[1]]
    a0 = tf.reshape(tf.tile(a[:, 0], [Ns]), [-1])
    a1 = tf.reshape(tf.tile(a[:, 1], [Ns]), [-1])
    flipped = slicetf.replace_slice_in(
        flipped)[:, site[0]].with_value(tf.reshape(a0, [K**2*Ns, 1]))
    flipped = slicetf.replace_slice_in(
        flipped)[:, site[1]].with_value(tf.reshape(a1, [K**2*Ns, 1]))
    a = tf.tile(a, [Ns, 1])
    # indices_ = tf.cast(tf.concat([a,tf.reshape(s0,[tf.shape(s0)[0],1]),tf.reshape(s1,[tf.shape(s1)[0],1])],1),tf.int32)
    indices_ = tf.cast(tf.concat([tf.reshape(s0, [tf.shape(s0)[0], 1]), tf.reshape(
        s1, [tf.shape(s1)[0], 1]), a], 1), tf.int32)
    # indices_ = tf.reverse(indices_,[1])
    # getting the coefficients of the p-gates that accompany the flipped samples
    Coef = tf.gather_nd(O, indices_)
    # If some coefficients are zero, then eliminate those configurations (could be improved I believe)
    # mask = tf.where(np.abs(Coef)<1e-13,False,True)
    # Coef = tf.boolean_mask(Coef,mask)
    # flipped = tf.boolean_mask(flipped,mask)
    lP = logP(model, tf.cast(flipped, dtype=tf.int64), config, False)
    P_target = tf.reduce_sum(tf.cast(tf.reshape(
        Coef, [Ns, K**2]), dtype=tf.float32)*tf.reshape(tf.math.exp(lP), [Ns, K**2]), axis=1)
    return P_target  # ,indices


@tf.function
def loss_func(
        batch_samples,
        batch_lP_model,
        batch_P_target,
        alpha,
        beta,
        gamma,
        delta,
        sub_idxs,
        lambda_targets,
        rho_targets,
        povm):
    # L2 loss
    L_L2 = tf.cast(alpha, tf.float32) * tf.reduce_sum((tf.math.exp(
        batch_lP_model) - batch_P_target)**2)/batch_lP_model.shape[0]
    L_zero = 0
    L_eig = 0
    L_poly = 0
    # subsys_eigvals_mc = [[None], [None]]

    # if beta > 0:
    # phys_loss, subsys_eigvals_mc = physicality_loss_red(
    #     batch_samples=batch_samples,
    #     batch_lP_model=batch_lP_model,
    #     sub_idxs=sub_idxs,
    #     lambda_targets=lambda_targets,
    #     povm=povm,)

    # phys_loss_den = physicality_loss_den(
    #     batch_samples=batch_samples,
    #     batch_lP_model=batch_lP_model,
    #     sub_idxs=sub_idxs,
    #     rho_targets=rho_targets,
    #     povm=povm,)

    # if beta > 0:
    phys_loss_zero, subsys_eigvals_mc = physicality_loss_zero(
        batch_samples=batch_samples,
        batch_lP_model=batch_lP_model,
        sub_idxs=sub_idxs,
        povm=povm,)
    L_zero = tf.cast(beta, tf.float32) * phys_loss_zero

    if gamma > 0:
        phys_loss_eig, _ = physicality_loss_eig(
            batch_samples=batch_samples,
            batch_lP_model=batch_lP_model,
            sub_idxs=sub_idxs,
            lambda_targets=lambda_targets,
            povm=povm,)
        L_eig = tf.cast(gamma, tf.float32) * phys_loss_eig
    if delta > 0:
        phys_loss_poly = physicality_loss_poly(
            batch_samples=batch_samples,
            batch_lP_model=batch_lP_model,
            povm=povm,)
        L_poly = tf.cast(delta, tf.float32) * phys_loss_poly

    # we take th log to make the values a bit bigger, we add epsilon for errors.
    return L_L2 + L_zero + L_eig + L_poly, [L_L2, L_zero, L_eig, L_poly], subsys_eigvals_mc


def get_density_eig_vals(P, povm, N_qubits, explicit=False):
    if explicit:
        rho = P_to_rho_explicit(P=P, povm=povm)
    else:
        rho = P_to_rho(
            P=P,
            povm=povm,
            N_qubits=N_qubits
        )
    rho = rho.reshape((2**N_qubits, 2**N_qubits))

    eigvals, _ = tf.linalg.eig(rho)
    eigvals = tf.sort(tf.math.real(eigvals), direction="ASCENDING")

    return tf.cast(tf.math.real(eigvals), tf.float32)


@tf.function
def subsys_distr(samples, Ps, sub_idx):
    samples_unique, Ps_unique = unique_configs(samples=samples, P=Ps)

    # Select columns specified by sub_idx
    selected_columns = tf.gather(samples_unique, sub_idx, axis=1)
    # Initialize a dictionary to store sums for each combination
    # Assuming Ps_unique has dtype that matches
    P_tensor = tf.zeros((4, 4), dtype=Ps_unique.dtype)

    # Iterate over all possible combinations (0,0) to (3,3)
    for i in range(4):
        for j in range(4):
            # Create a mask for the current combination
            mask_i = tf.equal(selected_columns[:, 0], i)
            mask_j = tf.equal(selected_columns[:, 1], j)
            mask = tf.logical_and(mask_i, mask_j)

            # Use the mask to sum the corresponding values in Ps_unique
            P_sum = tf.reduce_sum(tf.boolean_mask(Ps_unique, mask))
            P_tensor = tf.tensor_scatter_nd_add(P_tensor, [[i, j]], [P_sum])

    return P_tensor


@tf.function
def subsys_distr_one(samples, Ps, sub_idx):
    samples_unique, Ps_unique = unique_configs(samples=samples, P=Ps)

    # Select columns specified by sub_idx
    selected_columns = tf.gather(samples_unique, sub_idx, axis=1)
    # Initialize a dictionary to store sums for each combination
    # Assuming Ps_unique has dtype that matches
    P_tensor = tf.zeros((4,), dtype=Ps_unique.dtype)

    # Iterate over all possible combinations (0,0) to (3,3)
    for i in range(4):
        # Create a mask for the current combination
        mask = tf.equal(selected_columns[:, 0], i)

        # Use the mask to sum the corresponding values in Ps_unique
        P_sum = tf.reduce_sum(tf.boolean_mask(Ps_unique, mask))
        P_tensor = tf.tensor_scatter_nd_add(P_tensor, [[i]], [P_sum])

    return P_tensor


@tf.function
def get_subsystem_mc(samples, logP, sub_idx, povm):
    P_sub = subsys_distr(
        samples=samples,
        Ps=tf.math.exp(logP),
        sub_idx=sub_idx,
    )

    P_sub = P_sub/tf.reduce_sum(P_sub)

    rho_mc = P_to_rho_explicit(P=P_sub, povm=povm)
    return tf.reshape(rho_mc, (4, 4))


@tf.function
def get_subsystem_mc_one(samples, logP, sub_idx, povm):
    P_sub = subsys_distr_one(
        samples=samples,
        Ps=tf.math.exp(logP),
        sub_idx=sub_idx,
    )

    P_sub = P_sub/tf.reduce_sum(P_sub)

    rho_mc = P_to_rho_explicit_one(P=P_sub, povm=povm)
    return tf.reshape(rho_mc, (2, 2))


@tf.function
def partial_trace(density_matrix, subsystem=0):
    """
    Compute the partial trace of a 4x4 density matrix over a specified subsystem.

    Args:
    density_matrix (tf.Tensor): Input density matrix of shape (4, 4).
    subsystem (int): The subsystem to trace out (0 or 1). Defaults to 0.

    Returns:
    tf.Tensor: The reduced density matrix after tracing out the specified subsystem.
    """
    if subsystem == 1:
        # Reshape the density matrix to (2, 2, 2, 2)
        reshaped_matrix = tf.reshape(density_matrix, (2, 2, 2, 2))
        # Perform the partial trace over the first subsystem
        partial_trace_matrix = tf.einsum('ijik->jk', reshaped_matrix)
    elif subsystem == 0:
        # Reshape the density matrix to (2, 2, 2, 2)
        reshaped_matrix = tf.reshape(density_matrix, (2, 2, 2, 2))
        # Perform the partial trace over the second subsystem
        partial_trace_matrix = tf.einsum('ijkj->ik', reshaped_matrix)
    else:
        raise ValueError("Subsystem must be 0 or 1.")

    return partial_trace_matrix


@tf.function
def physicality_loss_red(batch_samples, batch_lP_model, sub_idxs, lambda_targets, povm):
    loss = 0
    subsys_eigvals_mc = []
    rho_subs = []

    # calculate loss per reduced P using eigenvalues and add to total
    for i, sub_idx in enumerate(sub_idxs):

        rho_sub_monte = get_subsystem_mc(
            samples=batch_samples,
            logP=batch_lP_model,
            sub_idx=sub_idx,
            povm=povm,
        )

        eigvals, _ = tf.linalg.eig(rho_sub_monte)
        eigvals = tf.sort(tf.math.real(eigvals), direction="ASCENDING")

        subsys_eigvals_mc.append(eigvals)
        rho_subs.append(rho_sub_monte)

    # rho_qs = {f"{x}": [] for x in range(batch_samples.shape[1])}

    for i, (rho_sub, sub_idx) in enumerate(zip(rho_subs, sub_idxs)):
        rho_sub_sub_0 = partial_trace(rho_sub, 0)
        rho_sub_sub_1 = partial_trace(rho_sub, 1)

        # rho_qs[f"{sub_idx[0]}"].append(rho_sub_sub_0)
        # rho_qs[f"{sub_idx[1]}"].append(rho_sub_sub_1)

        # print(sub_idx)
        # print(rho_sub)
        # print(rho_sub_sub_0)
        # print(rho_sub_sub_1)

        loss += tf.reduce_sum((tf.cast(tf.math.real(rho_sub_sub_0), dtype=tf.float32) -
                               tf.constant([[0.5, 0], [0, 0.5]], dtype=tf.float32))**2)

        if sub_idx[1] == 2:
            # print("USING 2")
            loss += tf.reduce_sum((tf.cast(tf.math.real(rho_sub_sub_1), dtype=tf.float32) -
                                   tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32))**2)
        else:
            loss += tf.reduce_sum((tf.cast(tf.math.real(rho_sub_sub_1), dtype=tf.float32) -
                                   tf.constant([[0.5, 0], [0, 0.5]], dtype=tf.float32))**2)
    return tf.cast(loss, tf.float32), subsys_eigvals_mc


@tf.function
def physicality_loss_den(batch_samples, batch_lP_model, sub_idxs, rho_targets, povm):
    loss = 0
    # calculate loss per reduced P using eigenvalues and add to total
    for i, sub_idx in enumerate(sub_idxs):
        # print(batch_lP_model)
        rho_sub_monte = get_subsystem_mc(
            samples=batch_samples,
            logP=batch_lP_model,
            sub_idx=sub_idx,
            povm=povm,
        )

        rho_real = tf.math.real(rho_sub_monte)
        rho_imag = tf.math.imag(rho_sub_monte)

        rho_real = tf.reshape(rho_real, (16, ))
        rho_imag = tf.reshape(rho_imag, (16, ))
        # print(rho_sub_monte)
        target_rho = rho_targets[i]

        loss += tf.norm(target_rho - rho_real, ord="euclidean")
        loss += tf.norm(tf.zeros((16,), dtype=tf.float64) -
                        rho_imag + epsilon, ord="euclidean")  # epsilon for numerical stability
    # print(f"loss: {loss}")
    return tf.cast(loss, tf.float32)


@tf.function
def physicality_loss_zero(batch_samples, batch_lP_model, sub_idxs, povm):
    loss = 0
    subsys_eigvals_mc = []

    # calculate loss per reduced P using eigenvalues and add to total
    for i, sub_idx in enumerate(sub_idxs):

        rho_sub_monte = get_subsystem_mc(
            samples=batch_samples,
            logP=batch_lP_model,
            sub_idx=sub_idx,
            povm=povm,
        )

        eigvals, _ = tf.linalg.eig(rho_sub_monte)
        vals_real = tf.sort(tf.math.real(eigvals), direction="ASCENDING")
        # vals_imag = tf.math.imag(eigvals)

        subsys_eigvals_mc.append(eigvals)

        eigen_real_neg_mask = tf.less(vals_real, 0)
        negative_values = tf.boolean_mask(vals_real, eigen_real_neg_mask)

        # negative because sum is negative.
        loss -= tf.reduce_sum(negative_values)
        # loss += tf.reduce_sum()

    return tf.cast(loss, tf.float32), subsys_eigvals_mc


@tf.function
def physicality_loss_eig(batch_samples, batch_lP_model, sub_idxs, lambda_targets, povm):
    loss = 0
    subsys_eigvals_mc = []

    # calculate loss per reduced P using eigenvalues and add to total
    for i, sub_idx in enumerate(sub_idxs):

        rho_sub_monte = get_subsystem_mc(
            samples=batch_samples,
            logP=batch_lP_model,
            sub_idx=sub_idx,
            povm=povm,
        )

        eigvals, _ = tf.linalg.eig(rho_sub_monte)
        vals_real = tf.sort(tf.math.real(eigvals), direction="ASCENDING")
        vals_imag = tf.math.imag(eigvals)

        subsys_eigvals_mc.append(eigvals)

        loss += tf.norm(tf.constant(
            lambda_targets[i], dtype=tf.float64) - vals_real, ord="euclidean")
        loss += tf.norm(tf.zeros((4, ), dtype=tf.float64) -
                        vals_imag + epsilon, ord="euclidean")

    return tf.cast(loss, tf.float32), subsys_eigvals_mc


@tf.function
def physicality_loss_poly(batch_samples, batch_lP_model, povm):
    loss = 0.0

    smaller_eigvals = []

    # calculate loss per reduced P using eigenvalues and add to total
    for i in range(batch_samples.shape[1]):

        rho_sub_monte = get_subsystem_mc_one(
            samples=batch_samples,
            logP=batch_lP_model,
            sub_idx=[i],
            povm=povm,
        )

        eigvals, _ = tf.linalg.eig(rho_sub_monte)
        # tf.print(f"EIGENVALUES {i}: {eigvals}")
        vals_real = tf.sort(tf.math.real(eigvals), direction="ASCENDING")

        smaller_eigvals.append(tf.cast(vals_real[0], tf.float32))

    if smaller_eigvals[0] > (smaller_eigvals[1] + smaller_eigvals[2]):
        loss += smaller_eigvals[0] - \
            (smaller_eigvals[1] + smaller_eigvals[2])

    if smaller_eigvals[1] > (smaller_eigvals[0] + smaller_eigvals[2]):
        loss += smaller_eigvals[1] - \
            (smaller_eigvals[0] + smaller_eigvals[2])

    if smaller_eigvals[2] > (smaller_eigvals[0] + smaller_eigvals[1]):
        loss += smaller_eigvals[2] - \
            (smaller_eigvals[0] + smaller_eigvals[1])
    # print("poly violation: ", smaller_eigvals, loss)

    return tf.cast(loss, tf.float32)


@tf.function
def step(model, batch_samples, batch_P_target, povm, alpha, beta, gamma, delta, sub_idxs, lambda_targets, rho_targets, config):
    with tf.GradientTape() as tape:
        batch_lP_model = logP(
            model=model,
            samples=batch_samples,
            config=config,
        )

        loss, losses, subsys_eigvals_mc = loss_func(
            batch_samples=batch_samples,
            batch_lP_model=batch_lP_model,
            batch_P_target=batch_P_target,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            sub_idxs=sub_idxs,
            lambda_targets=lambda_targets,
            rho_targets=rho_targets,
            povm=povm,
        )

    gradients = tape.gradient(loss, model.trainable_variables)

    return losses, gradients, batch_lP_model, subsys_eigvals_mc


def train_gate_(model, mps, povm, config, gate, site, lrs, epochs, gate_idx, P_exact=None):
    class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_lr, decay_period, decay_rate, proj_epoch):
            super(CustomLearningRateSchedule, self).__init__()
            self.initial_lr = initial_lr
            self.decay_period = decay_period
            self.decay_rate = decay_rate

            self.proj_epoch = proj_epoch

            self.lr = self.initial_lr
            self.is_projecting = False

        def __call__(self, N_e):
            # if N_e == self.proj_epoch and not self.is_projecting:
            #     self.lr = self.lr * 0.5
            #     self.is_projecting = True

            if N_e % self.decay_period == 0:
                self.lr = self.lr * self.decay_rate

            return self.lr

    # lr_schedule = CustomLearningRateSchedule(
    #     initial_lr=lr,
    #     decay_period=10,
    #     decay_rate=0.99,
    #     proj_epoch=int(config["EPOCHS"][gate_idx]) - 1,
    # )
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=lr,
    #     decay_steps=10,
    #     decay_rate=0.99)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lrs[0])
    # get dataset
    dataset_samples, P_target = build_dataset(
        model=model,
        config=config,
        gate=gate,
        site=site,
    )

    BATCHES_PER_EPOCH = int(config["N_DATASET"]/config["BATCH_SIZE"])
    train_losses = []

    performance = {
        "config": config,
        "losses": [],
        "cFID_MPS": [],
        "cFID_MPS_std": [],
        "cFID_target": [],
        "cFID_exact": [],
        "KL_MPS": [],
        "KL_MPS_std": [],
        "KL_target": [],
        "KL_exact": [],
        "qFID_MPS": [],
        "qFID_MPS_std": [],
        "qFID_target": [],
        "qFID_exact": [],
        "eigvals_model": [],
        "eigvals_model_sub": [],
        "eigvals_model_sub_mc": [],
        "eigvals_target": [],
        "eigvals_exact": [],
        "all_P_models": [],
    }

    lambda_targets = config["LAMBDA_TARGET"][gate_idx]
    tf.print(f"LAMBDA_TARGET: {lambda_targets}")

    RHO_TARGETS = tf.constant([[
        [0.25, 0.25, 0.25, -0.25, 0.25, 0.25, 0.25, -0.25,
            0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25],
        [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0,
            0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25],
        [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0,
            0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25],
    ], [
        [0.25, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, -0.25,
            0.25, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.25],
        [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0,
            0.0, 0.25, -0.25, 0.0, 0.0, -0.25, 0.25],
        [0.25, 0.0, 0.0, 0.25, 0.0, 0.25, 0.25, 0.0,
            0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.25],
    ],
        [
        [0.25, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, -0.25,
            0.25, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.25],
        [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0,
            0.0, 0.25, -0.25, 0.0, 0.0, -0.25, 0.25],
        [0.25, 0.0, 0.0, 0.25, 0.0, 0.25, 0.25, 0.0,
            0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.25],
    ]], dtype=tf.float64)

    rho_targets = RHO_TARGETS[gate_idx]

    for N_e in range(epochs):
        l = lrs[N_e]
        # learning_rate = lr_schedule(N_e, performance)
        # print(learning_rate)
        optimizer.learning_rate = l
        # tf.print(green(f"LR: {optimizer.learning_rate}, {lr}"))
        tf.print(green(f"LR: {l}"))

        begin = time()

        alpha = config["ALPHA"][gate_idx][N_e]
        beta = config["BETA"][gate_idx][N_e]
        gamma = config["GAMMA"][gate_idx][N_e]
        delta = config["DELTA"][gate_idx][N_e]

        tf.print(
            f"ALPHA: {alpha}, BETA: {beta}, GAMMA: {gamma}, DELTA: {delta}")

        for N_b in tqdm(range(BATCHES_PER_EPOCH)):

            batch_samples = get_batch(
                x=dataset_samples, N_b=N_b, batch_size=config["BATCH_SIZE"])

            batch_P_target = get_batch(
                x=P_target, N_b=N_b, batch_size=config["BATCH_SIZE"])

            losses, gradients, batch_lP_model, subsys_eigvals_mc = step(
                model=model,
                batch_samples=batch_samples,
                batch_P_target=batch_P_target,
                povm=povm,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                sub_idxs=config["SUBSYS_RESTR"],
                lambda_targets=lambda_targets,
                rho_targets=rho_targets,
                config=config,
            )
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            # for idx, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
            #     tf.summary.histogram(f"gradients/{var.path}", grad, step=N_e)
            #     tf.summary.histogram(f"variables/{var.path}", var, step=N_e)

        end = time()
        dt = end - begin

        begin = time()
        performance, P_model_unique, P_target_unique = assess_performance(
            perf=performance,
            model=model,
            config=config,
            povm=povm,
            mps=mps,
            losses=losses,
            batch_samples=batch_samples,
            batch_lP_model=batch_lP_model,
            batch_P_target=batch_P_target,
            P_exact=P_exact,
            subsys_eigvals_mc=subsys_eigvals_mc,
        )
        end = time()
        print(f"Assessing performance took: {(end - begin):.3f}s")
        print_epoch_perf(N_e, epochs, performance, dt)

    # save last prob. distributions for comparison
    performance["P_model"] = list(P_model_unique.astype(float))
    performance["P_target"] = list(P_target_unique.astype(float))
    performance["P_exact"] = list(P_exact.astype(float))

    return performance


if __name__ == "__main__":
    config = read_config("config.ini")
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

    # forces model to be built
    input_tensor = tf.zeros([10, 1])
    out = transformer(inputs=input_tensor, training=False)

    tf.print(transformer.summary())

    input(green("Press [ENTER] to continue..."))

    gate_str = config["GATES"][0]
    gate_type = config["GATE_TYPE"][0]

    # get quasi-stochastic matrix representing gate
    if gate_type == 1:
        gate = povm.single_qubit_dict[gate_str]
    elif gate_type == 2:
        gate = povm.two_qubit_dict[gate_str]

    with tf.device("GPU:0"):
        train_gate_(
            model=transformer,
            nmps=mps,
            config=config,
            gate=gate,
            site=config["SITES"][0]
        )
