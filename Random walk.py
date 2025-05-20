import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =======================
#  Configurable settings
# =======================
DIM = 2            # Walk dimension (2 or 3)
NUM_WALKS = 10000  # Number of parallel walkers
NUM_STEPS = 2000   # Number of steps per walker
NUM_REPEATS = 20   # Number of independent repetitions for stats
PLOT_SAMPLE_PATHS = True  # Show random walk samples

# =======================
#   Core simulation
# =======================
def get_directions(dim):
    """Return array of allowed directions in dim dimensions."""
    dirs = []
    for axis in range(dim):
        for sign in [-1, 1]:
            v = np.zeros(dim, dtype=int)
            v[axis] = sign
            dirs.append(v)
    return np.array(dirs)

def simulate_walks(dim, num_walks, num_steps, directions):
    """Simulate many random walks of given parameters, return full trajectory."""
    start = np.zeros(dim, dtype=int)
    choices = np.random.randint(0, len(directions), size=(num_walks, num_steps))
    steps = directions[choices]
    cum_steps = np.concatenate(
        (np.zeros((num_walks, 1, dim), dtype=int), np.cumsum(steps, axis=1)),
        axis=1
    )
    positions = start + cum_steps
    distances = np.linalg.norm(positions - start, axis=2)
    return positions, distances

def sqrt_model(n, A, B):
    return A * np.sqrt(n) + B

def fit_sqrt(steps, mean_dist):
    popt, _ = curve_fit(sqrt_model, steps, mean_dist, p0=[1,0])
    y_fit = sqrt_model(steps, *popt)
    mse = np.mean((mean_dist - y_fit)**2)
    return popt, mse, y_fit

# =======================
#   Run simulation
# =======================

directions = get_directions(DIM)
all_A = []
all_B = []

for repeat in range(NUM_REPEATS):
    positions, distances = simulate_walks(DIM, NUM_WALKS, NUM_STEPS, directions)
    mean_distance = distances.mean(axis=0)
    steps_arr = np.arange(mean_distance.size)
    popt, mse, y_fit = fit_sqrt(steps_arr, mean_distance)
    all_A.append(popt[0])
    all_B.append(popt[1])

    # Only plot for the first run to avoid clutter
    if repeat == 0:
        fig = plt.figure(figsize=(12, 5)) if DIM == 2 else plt.figure(figsize=(13, 5))
        # ---- Plot sample random walks ----
        if PLOT_SAMPLE_PATHS:
            if DIM == 2:
                ax1 = plt.subplot(1, 2, 1)
                for i in range(10):
                    ax1.plot(positions[i,:,0], positions[i,:,1], alpha=0.7, lw=1)
                ax1.set(
                    title=f'Sample random walks ({DIM}D)',
                    xlabel='x', ylabel='y'
                )
                ax1.text(0.02, 0.98, f'{10} sample walks', transform=ax1.transAxes, va='top', ha='left', fontsize=9, color='gray')
                ax1.grid()
            elif DIM == 3:
                from mpl_toolkits.mplot3d import Axes3D
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                for i in range(3):
                    ax1.plot(positions[i,:,0], positions[i,:,1], positions[i,:,2], alpha=0.8, lw=1)
                ax1.set(
                    title=f'Sample random walks ({DIM}D)',
                    xlabel='x', ylabel='y', zlabel='z'
                )
                ax1.text2D(0.02, 0.98, f'{3} sample walks', transform=ax1.transAxes, va='top', ha='left', fontsize=9, color='gray')
                ax1.grid()
        # ---- Plot mean distance vs. steps ----
        if DIM == 2:
            ax2 = plt.subplot(1, 2, 2)
        else:
            ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(steps_arr, mean_distance, label='Mean distance ⟨r⟩')
        ax2.plot(steps_arr, y_fit, '--', label=fr'Fit: $A\sqrt{{n}}+B$\nA={popt[0]:.4f}, B={popt[1]:.4f}, MSE={mse:.3g}')
        ax2.set(
            title=f'Mean distance from start vs. steps (n), {DIM}D',
            xlabel='Number of steps (n)', ylabel='Mean distance ⟨r⟩'
        )
        ax2.legend()
        ax2.grid()
        plt.tight_layout()
        plt.show()

# =======================
#   Statistical summary
# =======================
A_mean, A_std = np.mean(all_A), np.std(all_A)
B_mean, B_std = np.mean(all_B), np.std(all_B)

print(f"\n==== Fitting results after {NUM_REPEATS} repeats ====")
print(f"Fitted sqrt model:    ⟨r⟩ ≈ A·sqrt(n) + B")
print(f"Parameter A: {A_mean:.5f} ± {A_std:.5f}")
print(f"Parameter B: {B_mean:.5f} ± {B_std:.5f}")

# Optional: print all values
#print("All A:", np.round(all_A, 5))
#print("All B:", np.round(all_B, 5))
