# optimize_915.py
# ================================================================
# Meander Antenna Optimizer for 915 MHz
# ================================================================
# Strategy:
#   1. Scout sweep: wide band (200 MHz - 2 GHz) with initial geometry
#      to find where the antenna actually resonates
#   2. Optimize: use EMerge's built-in optimizer to tune
#      feed_penetration, gnd_height, and meander_offset to hit 915 MHz
#      with the best possible S11
# ================================================================

from pathlib import Path
import numpy as np
import emerge as em
from emerge.plot import plot_sp, plot_vswr, smith, plot_ff, plot_ff_polar

mm = 0.001
C0 = 299_792_458.0

outdir = Path("out")
outdir.mkdir(exist_ok=True)


def set_best_solver(sim):
    """Try MUMPS first (multi-threaded, best for macOS ARM),
    then UMFPACK (reuses symbolic factorization),
    then fall back to SuperLU (default)."""
    for solver, name in [
        (em.EMSolver.MUMPS, "MUMPS"),
        (em.EMSolver.UMFPACK, "UMFPACK"),
    ]:
        try:
            sim.mw.set_solver(solver)
            print(f"  Solver: {name}")
            return
        except Exception:
            pass
    print("  Solver: SuperLU (default)")


def build_full_model(
    sim,
    N=4,
    L_mm=50.0,
    W_mm=60.0,
    trace_w_mm=0.5,
    gnd_height_mm=20.0,
    feed_gap_mm=1.0,
    feed_penetration_mm=10.0,
    feed_clearance_mm=0.5,
    offset_x_mm=2.0,
    offset_y_mm=2.0,
    f0=915e6,
    margin_lambda=0.25,
    reset_model=True,
):
    """
    Builds the complete antenna model inside the given Simulation object:
    geometry + airbox + port + BCs + mesh.
    
    If reset_model=True, clears the simulation first (needed for optimization).

    Returns metadata dict and key objects.
    """
    if reset_model:
        sim.reset(all=True)
    L = L_mm * mm
    W = W_mm * mm
    trace_w = trace_w_mm * mm
    gnd_height = gnd_height_mm * mm
    feed_gap = feed_gap_mm * mm
    feed_penetration = feed_penetration_mm * mm
    feed_clearance = feed_clearance_mm * mm
    offset_x = offset_x_mm * mm
    offset_y = offset_y_mm * mm
    pcb_thickness = 1.6 * mm
    copper_thick = 0.035 * mm

    # --------------------------------------------------------
    # Meander layout
    # --------------------------------------------------------
    meander_y_bottom = gnd_height + feed_gap
    meander_y_top = W - offset_y
    n_legs = 2 * N

    meander_x_start = max(offset_x, feed_clearance)
    meander_x_end = L - offset_x
    meander_avail_w = meander_x_end - meander_x_start

    trace_gap = (meander_avail_w - n_legs * trace_w) / max(n_legs - 1, 1)
    leg_pitch = trace_w + trace_gap

    meander_arm_length = meander_y_top - meander_y_bottom - trace_w

    # Feed
    feed_x_centre = meander_x_start + trace_w / 2
    feed_y_bottom = gnd_height - feed_penetration

    # --------------------------------------------------------
    # Substrate
    # --------------------------------------------------------
    substrate = em.geo.Box(L, W, pcb_thickness, position=(0, 0, -pcb_thickness))
    substrate.set_material(em.Material(er=4.4, tand=0.02, color="#2d8c2d", opacity=0.6))

    # --------------------------------------------------------
    # Bottom ground plane (underneath the substrate, full board)
    # This provides the return current path for the microstrip feed
    # and gives the lumped port a ground reference.
    # --------------------------------------------------------
    bottom_gnd = em.geo.XYPlate(L, W, position=(0, 0, -pcb_thickness))
    bottom_gnd.set_material(em.lib.PEC)

    # --------------------------------------------------------
    # Ground plane + slot
    # --------------------------------------------------------
    gnd_full = em.geo.Box(L, gnd_height, copper_thick, position=(0, 0, 0))
    slot_left = feed_x_centre - trace_w / 2 - feed_clearance
    slot_width = trace_w + 2 * feed_clearance
    slot_y_bot = max(feed_y_bottom - feed_clearance, 0)
    slot_h = gnd_height - slot_y_bot

    slot = em.geo.Box(slot_width, slot_h, copper_thick, position=(slot_left, slot_y_bot, 0))
    gnd = em.geo.subtract(gnd_full, slot)
    gnd.set_material(em.lib.MET_COPPER)

    # --------------------------------------------------------
    # Antenna traces
    # --------------------------------------------------------
    antenna_parts = []

    # Feed line
    feed_len = meander_y_bottom - feed_y_bottom
    feed = em.geo.Box(
        trace_w, feed_len, copper_thick,
        position=(feed_x_centre - trace_w / 2, feed_y_bottom, 0)
    )
    antenna_parts.append(feed)

    # Overlap for robust boolean union
    join = 0.05 * mm
    ov = min(join, 0.25 * trace_w)
    x0 = meander_x_start

    for k in range(n_legs):
        xk = x0 + k * leg_pitch
        leg = em.geo.Box(
            trace_w, meander_arm_length + ov, copper_thick,
            position=(xk, meander_y_bottom, 0)
        )
        antenna_parts.append(leg)

    for k in range(n_legs - 1):
        x_left = x0 + k * leg_pitch
        conn_w = 2 * trace_w + trace_gap
        if (k % 2) == 0:
            top_conn = em.geo.Box(
                conn_w, trace_w + ov, copper_thick,
                position=(x_left, meander_y_top - trace_w - ov, 0)
            )
            antenna_parts.append(top_conn)
        else:
            bot_conn = em.geo.Box(
                conn_w, trace_w + ov, copper_thick,
                position=(x_left, meander_y_bottom, 0)
            )
            antenna_parts.append(bot_conn)

    antenna = em.geo.unite(*antenna_parts)
    antenna.set_material(em.lib.MET_COPPER)

    # --------------------------------------------------------
    # Port (vertical, from bottom ground up to feed trace)
    # --------------------------------------------------------
    # Matches the patch antenna example: port connects the feed trace
    # at z=0 down through the substrate to the bottom ground at z=-pcb_thickness.
    port_face = em.geo.Plate(
        np.array([feed_x_centre - trace_w / 2, feed_y_bottom, -pcb_thickness]),
        np.array([trace_w, 0, 0]),
        np.array([0, 0, pcb_thickness])
    )

    # --------------------------------------------------------
    # Airbox
    # --------------------------------------------------------
    lam = C0 / f0
    m = margin_lambda * lam
    air = em.geo.Box(
        L + 2 * m, W + 2 * m, pcb_thickness + 2 * m,
        position=(-m, -m, -pcb_thickness - m)
    )
    air.background()

    # --------------------------------------------------------
    # Commit + Mesh
    # --------------------------------------------------------
    sim.commit_geometry()
    sim.generate_mesh()

    # --------------------------------------------------------
    # Boundary conditions (after mesh)
    # --------------------------------------------------------
    sim.mw.bc.AbsorbingBoundary(air.boundary(), order=2, abctype='B')
    sim.mw.bc.LumpedPort(
        port_face, 1,
        width=trace_w, height=pcb_thickness,
        direction=em.ZAX, Z0=50.0
    )

    return {
        "air": air, "port_face": port_face,
        "trace_w": trace_w, "L": L, "W": W,
    }


# ================================================================
# STEP 1: SCOUT SWEEP — find where it resonates
# ================================================================
print("=" * 60)
print("STEP 1: Scout sweep (200 MHz - 2 GHz)")
print("=" * 60)

scout = em.Simulation("Scout", loglevel="INFO")
set_best_solver(scout)
scout.mw.set_frequency_range(200e6, 2e9, 51)
scout.mw.set_resolution(0.4)  # Coarser mesh for speed

meta = build_full_model(
    scout, N=2, L_mm=50.0, W_mm=60.0,
    trace_w_mm=0.5, gnd_height_mm=50.0,
    feed_penetration_mm=20.0,
)

data = scout.mw.run_sweep(parallel=True)
grid = data.scalar.grid
freqs = grid.freq
s11 = grid.S(1, 1)
s11_dB = 20 * np.log10(np.abs(s11))

# Find resonances (local minima)
print("\nScout S11 results:")
print(f"  Min S11: {np.min(s11_dB):.2f} dB @ {freqs[np.argmin(s11_dB)]/1e6:.0f} MHz")

# Save scout data
np.savetxt(
    outdir / "scout_s11.csv",
    np.column_stack([freqs / 1e6, s11_dB]),
    delimiter=",", header="freq_MHz,S11_dB", comments=""
)

# Plot scout
freq_dense = np.linspace(200e6, 2e9, 1001)
s11_dense = grid.model_S(1, 1, freq_dense)
plot_sp(freq_dense, s11_dense)

scout.clean()

# ================================================================
# STEP 2: OPTIMIZE for 915 MHz
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Optimizing for 915 MHz")
print("=" * 60)

sim = em.Simulation("MeanderOpt", loglevel="INFO")
set_best_solver(sim)

# Optimization parameters:
# N: number of meanders (rounded to integer)
# L_mm: PCB length
# W_mm: PCB width (total board height)
# gnd_height_mm: ground plane height — bigger GND = smaller meander region
sim.opt.add_param('N', 4.0, (2.0, 8.0))
sim.opt.add_param('L_mm', 50.0, (30.0, 100.0))
sim.opt.add_param('W_mm', 60.0, (40.0, 120.0))
sim.opt.add_param('gnd_height_mm', 20.0, (10.0, 50.0))
sim.opt.method = 'Powell'

# Target: minimize S11 at 915 MHz
f_target = 915e6

for N_opt, L_opt, W_opt, gnd_h_opt in sim.opt.run(max_iter=60):
    N_int = max(2, int(round(N_opt)))  # N must be integer >= 2
    print(f"\n  Trying: N={N_int}, L={L_opt:.1f}mm, W={W_opt:.1f}mm, GND={gnd_h_opt:.1f}mm")

    # Use a narrow band around 915 MHz for speed during optimization
    sim.mw.set_frequency_range(750e6, 1100e6, 21)
    sim.mw.set_resolution(0.4)

    try:
        meta = build_full_model(
            sim, N=N_int, L_mm=L_opt, W_mm=W_opt,
            gnd_height_mm=gnd_h_opt,
        )

        data = sim.mw.run_sweep(parallel=True)
        grid = data.scalar.slice_set(-21).grid

        # Metric: best S11 within ±25 MHz of 915 MHz
        # This finds designs where the resonance actually lands near 915 MHz,
        # not ones where a distant resonance has a weak tail at 915 MHz.
        eval_freqs = np.linspace(890e6, 940e6, 51)
        s11_band = grid.model_S(1, 1, eval_freqs)
        s11_band_dB = 20 * np.log10(np.abs(s11_band))

        # Best S11 in the target window
        best_idx = int(np.argmin(s11_band_dB))
        best_s11_in_band = float(s11_band_dB[best_idx])
        best_freq_in_band = eval_freqs[best_idx] / 1e6

        # Also get S11 at exactly 915 MHz for reporting
        s11_at_915 = grid.model_S(1, 1, np.array([f_target]))[0]
        s11_915_dB = 20 * np.log10(np.abs(s11_at_915))

        print(f"  S11 @ 915 MHz: {s11_915_dB:.2f} dB | "
              f"Best in band: {best_s11_in_band:.2f} dB @ {best_freq_in_band:.0f} MHz")

        # Cost: use the best S11 in the ±25 MHz window
        sim.opt.update(best_s11_in_band)

    except Exception as e:
        print(f"  ERROR: {e}")
        sim.opt.update(0.0)  # Worst possible S11 = 0 dB

# ================================================================
# STEP 3: FINAL SIMULATION with best parameters
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Final simulation with optimal parameters")
print("=" * 60)

# IMPORTANT: extract best parameters BEFORE reset clears the cache
solution, best_s11 = sim.opt.best
best_N = max(2, int(round(float(solution['N']))))
best_L = float(solution['L_mm'])
best_W = float(solution['W_mm'])
best_gnd_h = float(solution['gnd_height_mm'])

print(f"Best parameters from optimizer:")
print(f"  N:                {best_N}")
print(f"  L:                {best_L:.2f} mm")
print(f"  W:                {best_W:.2f} mm")
print(f"  GND height:       {best_gnd_h:.2f} mm")
print(f"  Meander region:   {best_W - best_gnd_h - 1:.1f} mm (W - GND - feed_gap)")
print(f"  Best S11 (optim): {best_s11:.2f} dB")

sim.reset(all=True)
set_best_solver(sim)

# Run detailed final simulation with SAME resolution as optimization
sim.mw.set_frequency_range(700e6, 1300e6, 101)
sim.mw.set_resolution(0.4)

print(f"\nBuilding final model with: N={best_N}, L={best_L:.2f}, W={best_W:.2f}, GND={best_gnd_h:.2f}")

meta = build_full_model(
    sim,
    N=best_N,
    L_mm=best_L,
    W_mm=best_W,
    gnd_height_mm=best_gnd_h,
)

data = sim.mw.run_sweep(parallel=True)

# After reset(all=True), data should only contain the final sim's 101 points
# Use grid directly
grid = data.scalar.grid
freqs = grid.freq
s11 = grid.S(1, 1)

# ================================================================
# EXPORT RESULTS
# ================================================================
# Flatten arrays in case they have extra dimensions from optimizer
freqs_flat = np.array(freqs).flatten()
s11_flat = np.array(s11).flatten()

# Manual Touchstone .s1p export (avoids EMerge grid shape issues)
with open(outdir / "optimized.s1p", "w") as f:
    f.write("! EMerge Meander Antenna - Optimized for 915 MHz\n")
    f.write(f"! N={best_N}, L={best_L:.2f}mm, W={best_W:.2f}mm, GND={best_gnd_h:.2f}mm\n")
    f.write("# MHZ S RI R 50.0\n")
    for i in range(len(freqs_flat)):
        f.write(f"{freqs_flat[i]/1e6:.6f} {np.real(s11_flat[i]):.8f} {np.imag(s11_flat[i]):.8f}\n")
print(f"Touchstone exported to {outdir / 'optimized.s1p'}")

# CSV export
s11_dB = 20 * np.log10(np.abs(s11_flat))
np.savetxt(
    outdir / "optimized_s11.csv",
    np.column_stack([freqs_flat / 1e6, np.real(s11_flat), np.imag(s11_flat), np.abs(s11_flat), s11_dB]),
    delimiter=",",
    header="freq_MHz,Re_S11,Im_S11,Mag_S11,S11_dB",
    comments=""
)
print(f"CSV exported to {outdir / 'optimized_s11.csv'}")

# Print final summary
idx_915 = int(np.argmin(np.abs(freqs_flat - 915e6)))
idx_best = int(np.argmin(np.abs(s11_flat)))
print(f"\nFinal Results:")
print(f"  S11 @ 915 MHz:  {s11_dB[idx_915]:.2f} dB")
print(f"  Best match:     {20*np.log10(np.abs(s11_flat[idx_best])):.2f} dB @ {freqs_flat[idx_best]/1e6:.1f} MHz")
print(f"  Optimal params: N={best_N}, L={best_L:.1f}mm, W={best_W:.1f}mm, GND={best_gnd_h:.1f}mm")

# ================================================================
# PLOTS
# ================================================================
freq_dense = np.linspace(700e6, 1300e6, 1001)
try:
    s11_dense = grid.model_S(1, 1, freq_dense)
except Exception:
    # Fallback: just use the raw data points
    s11_dense = s11_flat
    freq_dense = freqs_flat

plot_sp(freq_dense, s11_dense)
plot_vswr(freq_dense, s11_dense)

# Custom Smith chart with 915 MHz marker
import matplotlib.pyplot as plt

def plot_smith_with_marker(s11_data, freqs_hz, target_freq=915e6, best_match_freq=None):
    """Draw a Smith chart with the S11 trace, 915 MHz marker, and optional best-match marker."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    # Draw Smith chart circles
    # Constant resistance circles
    for r in [0, 0.2, 0.5, 1, 2, 5]:
        cx = r / (1 + r)
        cr = 1 / (1 + r)
        circle = plt.Circle((cx, 0), cr, fill=False, color='#334155', linewidth=0.5)
        ax.add_patch(circle)

    # Constant reactance arcs
    for x in [0.2, 0.5, 1, 2, 5]:
        theta = np.linspace(0, np.pi, 200)
        # Positive reactance
        cx_pos = 1 + 1j * x
        zc = 1 / x
        arc_x = 1 + zc * np.cos(theta)
        arc_y = zc + zc * np.sin(theta - np.pi)
        # Clip to unit circle
        mask = arc_x**2 + arc_y**2 <= 1.01
        ax.plot(arc_x[mask], arc_y[mask], color='#334155', linewidth=0.5)
        ax.plot(arc_x[mask], -arc_y[mask], color='#334155', linewidth=0.5)

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#475569', linewidth=1)
    # Real axis
    ax.plot([-1, 1], [0, 0], color='#475569', linewidth=0.5)

    # Plot S11 trace
    s11_arr = np.array(s11_data).flatten()
    freq_arr = np.array(freqs_hz).flatten()
    re = np.real(s11_arr)
    im = np.imag(s11_arr)

    # Color the trace by frequency
    for i in range(len(re) - 1):
        t = i / max(len(re) - 1, 1)
        r_c = 0.3 + 0.7 * (1 - t)
        b_c = 0.3 + 0.7 * t
        ax.plot(re[i:i+2], im[i:i+2], color=(r_c, 0.4, b_c), linewidth=2, alpha=0.8)

    # Mark start and end frequencies
    ax.plot(re[0], im[0], 'o', color='#f87171', markersize=6, zorder=10)
    ax.annotate(f'{freq_arr[0]/1e6:.0f} MHz', (re[0], im[0]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=9, color='#f87171', fontweight='bold')

    ax.plot(re[-1], im[-1], 'o', color='#60a5fa', markersize=6, zorder=10)
    ax.annotate(f'{freq_arr[-1]/1e6:.0f} MHz', (re[-1], im[-1]),
                textcoords="offset points", xytext=(10, -15),
                fontsize=9, color='#60a5fa', fontweight='bold')

    # Mark 915 MHz with a big marker — bright cyan for visibility
    idx_target = int(np.argmin(np.abs(freq_arr - target_freq)))
    tx, ty = re[idx_target], im[idx_target]

    # Glowing ring effect
    for size, alpha in [(24, 0.1), (20, 0.2), (16, 0.3), (12, 0.5)]:
        ax.plot(tx, ty, 'o', color='#00ff88', markersize=size, alpha=alpha, zorder=11)
    ax.plot(tx, ty, 'o', color='#00ff88', markersize=8, zorder=12)
    ax.plot(tx, ty, 'o', color='white', markersize=3, zorder=13)

    s11_db_at_target = 20 * np.log10(np.abs(s11_arr[idx_target]))
    ax.annotate(
        f'915 MHz\nS11 = {s11_db_at_target:.1f} dB',
        (tx, ty),
        textcoords="offset points", xytext=(20, 20),
        fontsize=12, fontweight='bold', color='#00ff88',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', edgecolor='#00ff88', linewidth=2, alpha=0.95),
        arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2),
        zorder=14,
    )

    # Mark the best-match (resonant) frequency in magenta
    if best_match_freq is not None:
        idx_best = int(np.argmin(np.abs(freq_arr - best_match_freq)))
        bx, by = re[idx_best], im[idx_best]
        s11_db_best = 20 * np.log10(np.abs(s11_arr[idx_best]))

        for size, alpha in [(22, 0.1), (18, 0.2), (14, 0.35)]:
            ax.plot(bx, by, 'o', color='#ff44ff', markersize=size, alpha=alpha, zorder=11)
        ax.plot(bx, by, 's', color='#ff44ff', markersize=8, zorder=12)
        ax.plot(bx, by, 's', color='white', markersize=3, zorder=13)

        ax.annotate(
            f'{best_match_freq/1e6:.0f} MHz (resonance)\nS11 = {s11_db_best:.1f} dB',
            (bx, by),
            textcoords="offset points", xytext=(-25, -30),
            fontsize=11, fontweight='bold', color='#ff44ff',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', edgecolor='#ff44ff', linewidth=2, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#ff44ff', lw=2),
            zorder=14,
        )

    # Labels
    ax.set_title('Smith Chart — S11', fontsize=16, color='#e2e8f0', fontweight='bold', pad=15)
    ax.text(1, -0.05, '∞', fontsize=12, color='#64748b', ha='center')
    ax.text(-1, -0.05, '0', fontsize=12, color='#64748b', ha='center')
    ax.text(0, -0.05, '1', fontsize=12, color='#64748b', ha='center')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(str(outdir / "smith_chart.png"), dpi=150, facecolor='#0f172a', bbox_inches='tight')
    plt.show()

# Find the actual resonant frequency (best S11 in the data)
best_match_freq_hz = float(freqs_flat[idx_best])
plot_smith_with_marker(s11_dense, freq_dense, target_freq=915e6, best_match_freq=best_match_freq_hz)

# Far-field at 915 MHz
try:
    abc_sel = meta["air"].boundary()

    # Find the frequency index closest to 915 MHz in our sweep
    idx_ff = int(np.argmin(np.abs(freqs_flat - 915e6)))
    field_915 = data.field[idx_ff]

    ff_xz = field_915.farfield_2d((0, 0, 1), (1, 0, 0), abc_sel, (-180, 180))
    ff_yz = field_915.farfield_2d((0, 0, 1), (0, 1, 0), abc_sel, (-180, 180))

    plot_ff(
        ff_xz.ang * 180 / np.pi,
        [ff_xz.gain.norm, ff_yz.gain.norm],
        dB=True, ylabel='Gain [dBi]',
        labels=['XZ plane', 'YZ plane']
    )

    plot_ff_polar(
        ff_xz.ang,
        [ff_xz.gain.norm, ff_yz.gain.norm],
        dB=True, dBfloor=-20,
        labels=['XZ plane', 'YZ plane']
    )

    # 3D radiation - center on antenna
    sim.display.populate()
    ant_cx = meta["L"] / 2
    ant_cy = meta["W"] / 2
    ff3d = field_915.farfield_3d(abc_sel, origin=(ant_cx, ant_cy, 0))
    # rmax scales the radiation pattern size, no offset so it's centered on the antenna
    surf = ff3d.surfplot('normE', rmax=meta["L"] * 3, isotropic=True)
    sim.display.add_surf(*surf.xyzf)
    sim.display.show()
except Exception as e:
    print(f"Far-field plotting error: {e}")
    print("S-parameter plots completed successfully. Far-field skipped.")
