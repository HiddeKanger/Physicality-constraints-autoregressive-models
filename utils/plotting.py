from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps
# print(colormaps)
blues = colormaps["Blues"]
reds = colormaps["Reds"]
greens = colormaps["Greens"]

# Set plotting parameters
plt.rcParams.update({'mathtext.fontset': 'cm'})
plt.rcParams.update({'font.family': 'STIXGeneral'})
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.xmargin': 0.01})


def numerical_sort(s):
    parts = []
    current_part = ""
    for char in s:
        if char.isdigit():
            current_part += char
        else:
            if current_part:
                parts.append(int(current_part))
                current_part = ""
    if current_part:
        parts.append(int(current_part))

    return tuple(parts)


def load_statistics_from_json(filename):
    with open(filename, 'r') as f:
        statistics = json.load(f)

    statistics_converted = {
        key: np.array(value) if isinstance(value, list) else value
        for key, value in statistics.items()
    }

    return statistics_converted


def plot_circuit_performance(dir_):
    config = load_statistics_from_json(f"models/{dir_}/config.json")

    perf_filenames = glob(f"models/{dir_}/performance_*_*.json")
    perf_filenames = sorted(perf_filenames, key=numerical_sort)
    print(perf_filenames)

    performances = []
    for filename in perf_filenames:
        performances.append(load_statistics_from_json(filename))

    losses = []
    KLs = []
    KL_exacts = []
    cFIDs = []
    cFID_exacts = []
    FIDs = []

    for perf in performances:
        losses.extend(perf["loss"])
        KLs.extend(perf["KL"])
        KL_exacts.extend(perf["KL_exact"])
        cFIDs.extend(np.abs(1 - perf["cFID"]))
        cFID_exacts.extend(np.abs(1 - perf["cFID_exact"]))
        FIDs.extend(np.abs(1 - perf["FID"]))

    epochs = np.arange(len(losses))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))

    # ax1.plot(epochs, KLs)
    ax1.plot(epochs, KL_exacts)
    # ax2.plot(epochs, cFIDs)
    ax2.plot(epochs, cFID_exacts)
    ax3.plot(epochs, losses)
    ax4.plot(epochs, FIDs)

    # ax1.scatter(epochs, KLs, s=3)
    ax1.scatter(epochs, KL_exacts, s=3)
    # ax2.scatter(epochs, cFIDs, s=3)
    ax2.scatter(epochs, cFID_exacts, s=3)
    ax3.scatter(epochs, losses, s=3)
    ax4.scatter(epochs, FIDs, s=3)

    ax1.set_title("KL div")
    ax2.set_title("|1 - cFID|")
    ax3.set_title("Loss function (L2)")
    ax4.set_title("|1 - FID|")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")

    fig.suptitle(
        f"{config['N_QUBITS']} Qubit {config['TARGET_STATE']} state, POVM: {config['POVM']}", fontsize=22)

    plt.tight_layout()

    plt.show()

    return performances


def plot_circuit_performance_eigvls_sub(dir_):
    config = load_statistics_from_json(f"{dir_}/config.json")

    perf_filenames = glob(f"{dir_}/performance_*_*.json")
    perf_filenames = sorted(perf_filenames, key=numerical_sort)
    print(perf_filenames)

    performances = []
    for filename in perf_filenames:
        performances.append(load_statistics_from_json(filename))

    losses = []

    cFID_MPSs = []
    cFID_targets = []
    cFID_exacts = []

    KL_MPSs = []
    KL_targets = []
    KL_exacts = []

    qFID_MPSs = []
    qFID_targets = []
    qFID_exacts = []

    eigvals_models = []
    eigvals_models_sub = []
    eigvals_targets = []
    eigvals_exacts = []

    for perf in performances:
        losses.extend(perf["losses"])

        cFID_MPSs.extend(np.abs(1 - perf["cFID_MPS"]))
        cFID_targets.extend(np.abs(1 - perf["cFID_target"]))
        cFID_exacts.extend(np.abs(1 - perf["cFID_exact"]))

        KL_MPSs.extend(perf["KL_MPS"])
        KL_targets.extend(perf["KL_target"])
        KL_exacts.extend(perf["KL_exact"])

        qFID_MPSs.extend(np.abs(1 - perf["qFID_MPS"]))
        qFID_targets.extend(np.abs(1 - perf["qFID_target"]))
        qFID_exacts.extend(np.abs(1 - perf["qFID_exact"]))

        eigvals_models.extend(perf["eigvals_model"])
        eigvals_models_sub.extend(perf["eigvals_model_sub"])
        eigvals_targets.extend(perf["eigvals_target"])
        eigvals_exacts.extend(perf["eigvals_exact"])

    losses = np.array(losses)

    eigvals_models = np.array(eigvals_models)
    eigvals_models_sub = np.array(eigvals_models_sub)
    eigvals_targets = np.array(eigvals_targets)
    eigvals_exacts = np.array(eigvals_exacts)

    epochs = np.arange(len(losses))

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        1, 5, figsize=(18, 3.5), sharex=True)

    for i in range(losses.shape[1]):
        if i == 0:
            ax1.plot(epochs, losses[:, i], label="L2")
        else:
            ax1.plot(epochs, losses[:, i], label="physicality")
        ax1.scatter(epochs, losses[:, i], s=3)

    ax2.plot(epochs, KL_MPSs, label=r"$P_{MPS}$")
    ax2.plot(epochs, KL_targets, label=r"$P_{target}$")
    ax2.plot(epochs, KL_exacts, label=r"$P_{exact}$")
    ax2.scatter(epochs, KL_MPSs, s=3)
    ax2.scatter(epochs, KL_targets, s=3)
    ax2.scatter(epochs, KL_exacts, s=3)

    ax3.plot(epochs, cFID_MPSs)
    ax3.plot(epochs, cFID_targets)
    ax3.plot(epochs, cFID_exacts)
    ax3.scatter(epochs, cFID_MPSs, s=3)
    ax3.scatter(epochs, cFID_targets, s=3)
    ax3.scatter(epochs, cFID_exacts, s=3)

    ax4.plot(epochs, qFID_MPSs)
    ax4.plot(epochs, qFID_targets)
    ax4.plot(epochs, qFID_exacts)
    ax4.scatter(epochs, qFID_MPSs, s=3)
    ax4.scatter(epochs, qFID_targets, s=3)
    ax4.scatter(epochs, qFID_exacts, s=3)

    for i in range(eigvals_models.shape[1]):
        color = blues(float(i) / eigvals_models.shape[1])

        if i == eigvals_models.shape[1] - 1:
            ax5.plot(epochs, eigvals_models[:, i],
                     color=color, label=r"$\lambda_{model}$")
        else:
            ax5.plot(epochs, eigvals_models[:, i], color=color)

        ax5.scatter(epochs, eigvals_models[:, i], s=3, color=color)

    for i in range(eigvals_targets.shape[1]):
        color = reds(float(i) / eigvals_models.shape[1])

        if i == eigvals_targets.shape[1] - 1:
            ax5.plot(epochs, eigvals_targets[:, i],
                     color=color, label=r"$\lambda_{target}$")
        else:
            ax5.plot(epochs, eigvals_targets[:, i], color=color)

        ax5.scatter(epochs, eigvals_targets[:, i], s=3, color=color)

    for i in range(eigvals_models_sub.shape[1]):
        color = greens(float(i) / eigvals_models_sub.shape[1])

        if i == eigvals_models_sub.shape[1] - 1:
            ax5.plot(epochs, eigvals_models_sub[:, i],
                     color=color, label=r"$\lambda_{sub}$")
        else:
            ax5.plot(epochs, eigvals_models_sub[:, i], color=color)

        ax5.scatter(epochs, eigvals_models_sub[:, i], s=3, color=color)

    ax1.set_title("Loss function (L2)")
    ax2.set_title("KL div")
    ax3.set_title("|1 - cFID|")
    ax4.set_title("|1 - FID|")
    ax5.set_title(r"$\lambda_i$")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")

    handles, labels = ax1.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.11, 0.85), ncol=4)

    handles, labels = ax2.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.5, 0.85), ncol=4)

    handles, labels = ax5.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.9, 0.85), ncol=2)

    fig.suptitle(
        f"{config['N_QUBITS']} Qubit {config['TARGET_STATE']} state, POVM: {config['POVM']}", fontsize=22, y=0.99)

    plt.tight_layout()

    plt.subplots_adjust(top=0.7)

    plt.show()

    return performances


def plot_circuit_performance_eigvls_sub_mc(dir_):
    config = load_statistics_from_json(f"{dir_}/config.json")

    perf_filenames = glob(f"{dir_}/performance_*_*.json")
    perf_filenames = sorted(perf_filenames, key=numerical_sort)
    print(perf_filenames)

    performances = []
    for filename in perf_filenames:
        performances.append(load_statistics_from_json(filename))

    losses = []

    cFID_MPSs = []
    cFID_targets = []
    cFID_exacts = []

    KL_MPSs = []
    KL_targets = []
    KL_exacts = []

    qFID_MPSs = []
    qFID_targets = []
    qFID_exacts = []

    eigvals_models = []
    eigvals_models_sub_mc = []
    eigvals_targets = []
    eigvals_exacts = []

    for perf in performances:
        losses.extend(perf["losses"])

        cFID_MPSs.extend(np.abs(1 - perf["cFID_MPS"]))
        cFID_targets.extend(np.abs(1 - perf["cFID_target"]))
        cFID_exacts.extend(np.abs(1 - perf["cFID_exact"]))

        KL_MPSs.extend(perf["KL_MPS"])
        KL_targets.extend(perf["KL_target"])
        KL_exacts.extend(perf["KL_exact"])

        qFID_MPSs.extend(np.abs(1 - perf["qFID_MPS"]))
        qFID_targets.extend(np.abs(1 - perf["qFID_target"]))
        qFID_exacts.extend(np.abs(1 - perf["qFID_exact"]))

        eigvals_models.extend(perf["eigvals_model"])
        eigvals_models_sub_mc.extend(perf["eigvals_model_sub_mc"])
        eigvals_targets.extend(perf["eigvals_target"])
        eigvals_exacts.extend(perf["eigvals_exact"])

    losses = np.array(losses)

    eigvals_models = np.array(eigvals_models)
    eigvals_models_sub_mc = np.array(eigvals_models_sub_mc)
    print(eigvals_models_sub_mc.shape)
    eigvals_targets = np.array(eigvals_targets)
    eigvals_exacts = np.array(eigvals_exacts)

    epochs = np.arange(len(losses))

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
        1, 6, figsize=(18, 3.5), sharex=True)

    for i in range(losses.shape[1]):
        if i == 0:
            ax1.plot(epochs, losses[:, i], label="L2")
        else:
            ax1.plot(epochs, losses[:, i], label="physicality")
        ax1.scatter(epochs, losses[:, i], s=3)

    # ax2.plot(epochs, KL_MPSs, label=r"$P_{MPS}$")
    ax2.plot(epochs, KL_targets, label=r"$P_{target}$")
    ax2.plot(epochs, KL_exacts, label=r"$P_{exact}$")
    # ax2.scatter(epochs, KL_MPSs, s=3)
    ax2.scatter(epochs, KL_targets, s=3)
    ax2.scatter(epochs, KL_exacts, s=3)

    # ax3.plot(epochs, cFID_MPSs)
    ax3.plot(epochs, cFID_targets)
    ax3.plot(epochs, cFID_exacts)
    # ax3.scatter(epochs, cFID_MPSs, s=3)
    ax3.scatter(epochs, cFID_targets, s=3)
    ax3.scatter(epochs, cFID_exacts, s=3)

    # ax4.plot(epochs, qFID_MPSs)
    ax4.plot(epochs, qFID_targets)
    ax4.plot(epochs, qFID_exacts)
    # ax4.scatter(epochs, qFID_MPSs, s=3)
    ax4.scatter(epochs, qFID_targets, s=3)
    ax4.scatter(epochs, qFID_exacts, s=3)

    for i in range(eigvals_models.shape[1]):
        color = blues(0.5 + float(i)/(2 * eigvals_models_sub_mc.shape[2]))

        if i == eigvals_models.shape[1] - 1:
            ax5.plot(epochs, eigvals_models[:, i],
                     color=color, label=r"$\lambda_{model}$")
        else:
            ax5.plot(epochs, eigvals_models[:, i], color=color)

        ax5.scatter(epochs, eigvals_models[:, i], s=3, color=color)

    for i in range(eigvals_targets.shape[1]):
        color = reds(0.5 + float(i)/(2 * eigvals_models_sub_mc.shape[2]))

        if i == eigvals_targets.shape[1] - 1:
            ax5.plot(epochs, eigvals_targets[:, i],
                     color=color, label=r"$\lambda_{target}$")
        else:
            ax5.plot(epochs, eigvals_targets[:, i], color=color)

        ax5.scatter(epochs, eigvals_targets[:, i], s=3, color=color)

    for i in range(eigvals_models_sub_mc.shape[1]):
        for j in range(eigvals_models_sub_mc.shape[2]):
            if i == 0:
                color = blues(
                    0.5 + float(j)/(2 * eigvals_models_sub_mc.shape[2]))
            if i == 1:
                color = reds(0.5 + float(j) /
                             (2 * eigvals_models_sub_mc.shape[2]))
            if i == 2:
                color = greens(0.5 + float(j) /
                               (2 * eigvals_models_sub_mc.shape[2]))
            if j == eigvals_models_sub_mc.shape[2] - 1:
                ax6.plot(epochs, eigvals_models_sub_mc[:, i, j],
                         color=color, label=r"$\lambda_{MC,\ sub}$")
            else:
                ax6.plot(epochs, eigvals_models_sub_mc[:, i, j], color=color)

            ax6.scatter(
                epochs, eigvals_models_sub_mc[:, i, j], s=3, color=color)

    ax1.set_title("Loss function (L2)")
    ax2.set_title("KL div")
    ax3.set_title("|1 - cFID|")
    ax4.set_title("|1 - FID|")
    ax5.set_title(r"$\lambda_i$")
    ax6.set_title(r"$\lambda_{MC,\ sub}$")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")

    handles, labels = ax1.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.11, 0.85), ncol=4)

    handles, labels = ax2.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.5, 0.85), ncol=4)

    handles, labels = ax5.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.7, 0.85), ncol=2)

    handles, labels = ax6.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.9, 0.85), ncol=2)

    fig.suptitle(
        f"{config['N_QUBITS']} Qubit {config['TARGET_STATE']} state, POVM: {config['POVM']}", fontsize=22, y=0.99)

    plt.tight_layout()

    plt.subplots_adjust(top=0.7)

    plt.show()

    return performances


def plot_circuit_performance_eigvls(dir_):
    config = load_statistics_from_json(f"{dir_}/config.json")

    perf_filenames = glob(f"{dir_}/performance_*_*.json")
    perf_filenames = sorted(perf_filenames, key=numerical_sort)
    print(perf_filenames)

    performances = []
    for filename in perf_filenames:
        performances.append(load_statistics_from_json(filename))

    losses = []

    cFID_MPSs = []
    cFID_targets = []
    cFID_exacts = []

    KL_MPSs = []
    KL_targets = []
    KL_exacts = []

    qFID_MPSs = []
    qFID_targets = []
    qFID_exacts = []

    eigvals_models = []
    eigvals_targets = []
    eigvals_exacts = []

    for perf in performances:
        losses.extend(perf["losses"])

        cFID_MPSs.extend(np.abs(1 - perf["cFID_MPS"]))
        cFID_targets.extend(np.abs(1 - perf["cFID_target"]))
        cFID_exacts.extend(np.abs(1 - perf["cFID_exact"]))

        KL_MPSs.extend(perf["KL_MPS"])
        KL_targets.extend(perf["KL_target"])
        KL_exacts.extend(perf["KL_exact"])

        qFID_MPSs.extend(np.abs(1 - perf["qFID_MPS"]))
        qFID_targets.extend(np.abs(1 - perf["qFID_target"]))
        qFID_exacts.extend(np.abs(1 - perf["qFID_exact"]))

        eigvals_models.extend(perf["eigvals_model"])
        eigvals_targets.extend(perf["eigvals_target"])
        eigvals_exacts.extend(perf["eigvals_exact"])

    losses = np.array(losses)

    eigvals_models = np.array(eigvals_models)
    eigvals_targets = np.array(eigvals_targets)
    eigvals_exacts = np.array(eigvals_exacts)

    epochs = np.arange(len(losses))

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        1, 5, figsize=(18, 3.5), sharex=True)

    for i in range(losses.shape[1]):
        if i == 0:
            ax1.plot(epochs, losses[:, i], label="L2")
        else:
            ax1.plot(epochs, losses[:, i], label="physicality")
        ax1.scatter(epochs, losses[:, i], s=3)

    ax2.plot(epochs, KL_MPSs, label=r"$P_{MPS}$")
    ax2.plot(epochs, KL_targets, label=r"$P_{target}$")
    ax2.plot(epochs, KL_exacts, label=r"$P_{exact}$")
    ax2.scatter(epochs, KL_MPSs, s=3)
    ax2.scatter(epochs, KL_targets, s=3)
    ax2.scatter(epochs, KL_exacts, s=3)

    ax3.plot(epochs, cFID_MPSs)
    ax3.plot(epochs, cFID_targets)
    ax3.plot(epochs, cFID_exacts)
    ax3.scatter(epochs, cFID_MPSs, s=3)
    ax3.scatter(epochs, cFID_targets, s=3)
    ax3.scatter(epochs, cFID_exacts, s=3)

    ax4.plot(epochs, qFID_MPSs)
    ax4.plot(epochs, qFID_targets)
    ax4.plot(epochs, qFID_exacts)
    ax4.scatter(epochs, qFID_MPSs, s=3)
    ax4.scatter(epochs, qFID_targets, s=3)
    ax4.scatter(epochs, qFID_exacts, s=3)

    for i in range(eigvals_models.shape[1]):
        color = blues(float(i) / eigvals_models.shape[1])

        if i == eigvals_models.shape[1] - 1:
            ax5.plot(epochs, eigvals_models[:, i],
                     color=color, label=r"$\lambda_{model}$")
        else:
            ax5.plot(epochs, eigvals_models[:, i], color=color)

        ax5.scatter(epochs, eigvals_models[:, i], s=3, color=color)

    for i in range(eigvals_targets.shape[1]):
        color = reds(float(i) / eigvals_models.shape[1])

        if i == eigvals_targets.shape[1] - 1:
            ax5.plot(epochs, eigvals_targets[:, i],
                     color=color, label=r"$\lambda_{target}$")
        else:
            ax5.plot(epochs, eigvals_targets[:, i], color=color)

        ax5.scatter(epochs, eigvals_targets[:, i], s=3, color=color)

    ax1.set_title("Loss function (L2)")
    ax2.set_title("KL div")
    ax3.set_title("|1 - cFID|")
    ax4.set_title("|1 - FID|")
    ax5.set_title(r"$\lambda_i$")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")

    handles, labels = ax1.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.11, 0.85), ncol=4)

    handles, labels = ax2.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.5, 0.85), ncol=4)

    handles, labels = ax5.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.9, 0.85), ncol=2)

    fig.suptitle(
        f"{config['N_QUBITS']} Qubit {config['TARGET_STATE']} state, POVM: {config['POVM']}", fontsize=22, y=0.99)

    plt.tight_layout()

    plt.subplots_adjust(top=0.7)

    plt.show()

    return performances


def plot_performance(filename):
    data = load_statistics_from_json(filename)
    config = data["config"]

    loss = data["loss"]
    cFID = np.abs(1 - data["cFID"])
    cFID_std = data["cFID_std"]
    cFID_exact = np.abs(1 - data["cFID_exact"])
    KL = data["KL"]
    KL_std = data["KL_std"]
    KL_exact = data["KL_exact"]
    FID = np.abs(1 - data["FID"])
    FID_std = data["FID_std"]

    epochs = range(len(loss))

    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 4, 1)
    ax2 = plt.subplot(2, 4, 2)
    ax3 = plt.subplot(2, 4, 3)
    ax4 = plt.subplot(2, 4, 4)
    ax5 = plt.subplot(2, 1, 2)

    ax1.plot(epochs, KL)
    ax1.plot(epochs, KL_exact)
    ax2.plot(epochs, cFID)
    ax2.plot(epochs, cFID_exact)
    ax3.plot(epochs, loss)
    ax4.plot(epochs, FID)

    # ax1.scatter(epochs, KL, s=3)
    ax1.scatter(epochs, KL_exact, s=3)
    # ax2.scatter(epochs, cFID, s=3)
    ax2.scatter(epochs, cFID_exact, s=3)
    ax3.scatter(epochs, loss, s=3)
    ax4.scatter(epochs, FID, s=3)

    w, spacing = 0.2, 0.25

    ax5.bar(np.arange(16) - spacing, data["Q_start"], alpha=0.75, width=w,
            label=r"$P_{\theta, i = 0}(\boldsymbol{a})$")
    ax5.bar(np.arange(16), data["Q"], alpha=0.75,
            width=w, label=r"$P_{\theta, i = 1}(\boldsymbol{a})$")
    ax5.bar(np.arange(16) + spacing, data["P"], alpha=0.75,
            width=w, label=r"$P^{(e)}_{i = 0}(\boldsymbol{a})$")
    ax5.legend(fontsize=14)

    axis_labels = [
        '(1,1)', '(1,2)', '(1,3)', '(1,4)',
        '(2,1)', '(2,2)', '(2,3)', '(2,4)',
        '(3,1)', '(3,2)', '(3,3)', '(3,4)',
        '(4,1)', '(4,2)', '(4,3)', '(4,4)'
    ]
    ax5.set_xticks(np.arange(16))
    ax5.set_xticklabels(axis_labels)

    ax1.set_title("KL div")
    ax2.set_title("|1 - cFID|")
    ax3.set_title("Loss function (L2)")
    ax4.set_title("|1 - FID|")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")
    # ax5.set_yscale("log")

    # ax5.set_xlabel(r"$\boldsymbol{a}$")

    fig.suptitle(
        f"gate: {config['GATE']}, POVM: {config['POVM']}", fontsize=22)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # name = "performance_2024-04-24_16-52-12"
    # plot_performance(f"data/{name}.json")

    plot_circuit_performance_eigvls("models/2024-05-27_17-15-56")
