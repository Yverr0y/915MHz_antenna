# verify.py
# Re-run the optimised PIFA simulation to verify results.
# Saves plots and data for the report.

from pathlib import Path
import numpy as np
import emerge as em
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mm = 0.001
C0 = 299_792_458.0

F_BAND_LO = 915e6
F_BAND_HI = 928e6
F_TARGET = 925e6
F_CENTER = 0.5 * (F_BAND_LO + F_BAND_HI)

outdir = Path("out")
outdir.mkdir(exist_ok=True)
reportdir = Path("report")
reportdir.mkdir(exist_ok=True)

# Optimised parameters from optimise_915.py
best_pL = 34.42
best_pW = 26.00
best_sW = 12.00
best_fo = 7.00

print("Verification run with optimised parameters:")
print(f"  patch_length: {best_pL:.2f} mm")
print(f"  patch_width:  {best_pW:.2f} mm")
print(f"  short_span:   {best_sW:.2f} mm")
print(f"  feed_offset:  {best_fo:.2f} mm")
print()


def set_best_solver(sim):
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
    L_mm=100.0, W_mm=80.0,
    patch_length_mm=45.0, patch_width_mm=25.0,
    patch_margin_x_mm=5.0, patch_margin_y_mm=5.0,
    short_span_mm=12.0, n_short_vias=6, via_pad_mm=0.6,
    feed_offset_mm=6.0, feed_trace_w_mm=1.5,
    f0=915e6, margin_lambda=0.25,
):
    L = L_mm * mm
    W = W_mm * mm
    patch_length = patch_length_mm * mm
    patch_width = patch_width_mm * mm
    patch_margin_x = patch_margin_x_mm * mm
    patch_margin_y = patch_margin_y_mm * mm
    short_span = short_span_mm * mm
    via_pad = via_pad_mm * mm
    feed_offset = feed_offset_mm * mm
    feed_trace_w = feed_trace_w_mm * mm
    pcb_thickness = 1.6 * mm
    copper_thick = 0.035 * mm

    patch_x0 = patch_margin_x
    patch_y1 = W - patch_margin_y
    patch_y0 = patch_y1 - patch_width
    patch_y_center = 0.5 * (patch_y0 + patch_y1)
    short_y0 = patch_y_center - short_span / 2.0

    substrate = em.geo.Box(L, W, pcb_thickness, position=(0, 0, -pcb_thickness))
    substrate.set_material(em.Material(er=4.4, tand=0.02, color="#2d8c2d", opacity=0.6))

    bottom_gnd = em.geo.XYPlate(L, W, position=(0, 0, -pcb_thickness))
    bottom_gnd.set_material(em.lib.PEC)

    patch_box = em.geo.Box(
        patch_length, patch_width, copper_thick,
        position=(patch_x0, patch_y0, 0)
    )
    if n_short_vias == 1:
        via_centres_y = np.array([short_y0 + short_span / 2.0])
    else:
        via_centres_y = np.linspace(
            short_y0 + via_pad / 2.0,
            short_y0 + short_span - via_pad / 2.0,
            n_short_vias,
        )
    via_height = pcb_thickness + 2 * copper_thick
    patch = patch_box
    for y_c in via_centres_y:
        via = em.geo.Box(
            via_pad, via_pad, via_height,
            position=(patch_x0, float(y_c) - via_pad / 2.0,
                      -pcb_thickness - copper_thick),
        )
        patch = em.geo.unite(patch, via)
    patch.set_material(em.lib.MET_COPPER)

    feed_x = patch_x0 + feed_offset
    port_face = em.geo.Plate(
        np.array([feed_x, patch_y_center - feed_trace_w / 2.0, -pcb_thickness]),
        np.array([0.0, feed_trace_w, 0.0]),
        np.array([0.0, 0.0, pcb_thickness])
    )

    lam = C0 / f0
    m = margin_lambda * lam
    air = em.geo.Box(
        L + 2 * m, W + 2 * m, pcb_thickness + 2 * m,
        position=(-m, -m, -pcb_thickness - m)
    )
    air.background()

    sim.commit_geometry()
    sim.generate_mesh()

    sim.mw.bc.AbsorbingBoundary(air.boundary(), order=2, abctype='B')
    sim.mw.bc.LumpedPort(
        port_face, 1,
        width=feed_trace_w, height=pcb_thickness,
        direction=em.ZAX, Z0=50.0
    )

    return {
        "air": air, "port_face": port_face, "patch": patch,
        "L": L, "W": W,
        "patch_length": patch_length, "patch_width": patch_width,
        "short_span": short_span, "feed_offset": feed_offset,
    }


# --- Run simulation ---
sim = em.Simulation("PIFAVerify", loglevel="INFO")
set_best_solver(sim)
sim.mw.set_frequency_range(700e6, 1300e6, 121)
sim.mw.set_resolution(0.25)

meta = build_full_model(
    sim,
    patch_length_mm=best_pL,
    patch_width_mm=best_pW,
    short_span_mm=best_sW,
    feed_offset_mm=best_fo,
)

data = sim.mw.run_sweep(parallel=True)
grid = data.scalar.grid
freqs_flat = np.array(grid.freq).flatten()
s11_flat = np.array(grid.S(1, 1)).flatten()
s11_dB = 20.0 * np.log10(np.abs(s11_flat))

# --- Results ---
idx_best = int(np.argmin(s11_dB))
res_freq_mhz = freqs_flat[idx_best] / 1e6

ism_freqs = np.linspace(F_BAND_LO, F_BAND_HI, 27)
s11_ism = grid.model_S(1, 1, ism_freqs)
s11_ism_dB = 20.0 * np.log10(np.abs(s11_ism))
ism_worst = float(np.max(s11_ism_dB))
ism_worst_mhz = float(ism_freqs[int(np.argmax(s11_ism_dB))] / 1e6)
ism_best = float(np.min(s11_ism_dB))
ism_best_mhz = float(ism_freqs[int(np.argmin(s11_ism_dB))] / 1e6)

below_10 = s11_dB <= -10.0
if np.any(below_10):
    below_idx = np.where(below_10)[0]
    bw_lo_mhz = freqs_flat[below_idx[0]] / 1e6
    bw_hi_mhz = freqs_flat[below_idx[-1]] / 1e6
    bw_10 = bw_hi_mhz - bw_lo_mhz
else:
    bw_lo_mhz = bw_hi_mhz = bw_10 = float('nan')

idx_925 = int(np.argmin(np.abs(freqs_flat - 925e6)))
s11_at_925 = s11_dB[idx_925]

# VSWR
vswr = (1 + np.abs(s11_flat)) / (1 - np.abs(s11_flat))

in_band = (F_BAND_LO / 1e6) <= res_freq_mhz <= (F_BAND_HI / 1e6)
matched = ism_worst <= -10.0
status = "PASS" if (in_band and matched) else "FAIL"

print("\nResults:")
print(f"  Resonance:         {res_freq_mhz:.1f} MHz ({s11_dB[idx_best]:+.2f} dB)")
print(f"  S11 @ 925 MHz:     {s11_at_925:+.2f} dB")
print(f"  -10 dB bandwidth:  {bw_lo_mhz:.0f} - {bw_hi_mhz:.0f} MHz ({bw_10:.0f} MHz)")
print(f"  ISM worst S11:     {ism_worst:+.2f} dB @ {ism_worst_mhz:.0f} MHz")
print(f"  ISM best S11:      {ism_best:+.2f} dB @ {ism_best_mhz:.0f} MHz")
print(f"  Spec:              {status}")

# --- Dense interpolation for smooth plots ---
freq_dense = np.linspace(700e6, 1300e6, 1001)
try:
    s11_dense = grid.model_S(1, 1, freq_dense)
except Exception:
    s11_dense = s11_flat
    freq_dense = freqs_flat

s11_dense_dB = 20.0 * np.log10(np.abs(s11_dense))
vswr_dense = (1 + np.abs(s11_dense)) / (1 - np.abs(s11_dense))

# --- S11 Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freq_dense / 1e6, s11_dense_dB, 'b-', linewidth=1.5, label='S11')
ax.axhline(-10, color='r', linestyle='--', alpha=0.7, label='-10 dB threshold')
ax.axvspan(915, 928, alpha=0.15, color='green', label='AU915 band (915-928 MHz)')
ax.axvline(res_freq_mhz, color='orange', linestyle=':', alpha=0.7,
           label=f'Resonance ({res_freq_mhz:.1f} MHz)')
ax.set_xlabel('Frequency (MHz)', fontsize=12)
ax.set_ylabel('S11 (dB)', fontsize=12)
ax.set_title('PIFA 925 MHz - S11 Return Loss', fontsize=14)
ax.set_xlim(700, 1300)
ax.set_ylim(-40, 0)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(reportdir / 's11_plot.png', dpi=150)
print(f"  Saved {reportdir / 's11_plot.png'}")

# --- VSWR Plot ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(freq_dense / 1e6, vswr_dense, 'b-', linewidth=1.5, label='VSWR')
ax2.axhline(2.0, color='r', linestyle='--', alpha=0.7, label='VSWR 2:1')
ax2.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='VSWR 3:1')
ax2.axvspan(915, 928, alpha=0.15, color='green', label='AU915 band')
ax2.set_xlabel('Frequency (MHz)', fontsize=12)
ax2.set_ylabel('VSWR', fontsize=12)
ax2.set_title('PIFA 925 MHz - VSWR', fontsize=14)
ax2.set_xlim(700, 1300)
ax2.set_ylim(1, 10)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(reportdir / 'vswr_plot.png', dpi=150)
print(f"  Saved {reportdir / 'vswr_plot.png'}")

# --- Smith Chart ---
fig3, ax3 = plt.subplots(figsize=(7, 7))
theta = np.linspace(0, 2 * np.pi, 200)
ax3.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
for r in [0.2, 0.5, 1.0, 2.0, 5.0]:
    cx = r / (1 + r)
    cr = 1 / (1 + r)
    ax3.add_patch(plt.Circle((cx, 0), cr, fill=False, color='gray',
                              linewidth=0.3))
for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
    arc_theta = np.linspace(0, np.pi / 2, 100)
    cx, cy = 1, 1 / x
    r_arc = 1 / x
    ax3.plot(cx + r_arc * np.cos(np.pi + arc_theta),
             cy + r_arc * np.sin(np.pi + arc_theta), 'gray', linewidth=0.3)
    ax3.plot(cx + r_arc * np.cos(np.pi - arc_theta),
             -cy - r_arc * np.sin(np.pi - arc_theta), 'gray', linewidth=0.3)

# Normalised reflection coefficient
gamma = s11_dense
ax3.plot(np.real(gamma), np.imag(gamma), 'b-', linewidth=1.5)

# Mark ISM band points
ism_mask = (freq_dense >= F_BAND_LO) & (freq_dense <= F_BAND_HI)
ax3.plot(np.real(gamma[ism_mask]), np.imag(gamma[ism_mask]),
         'r-', linewidth=3, label='AU915 band')

# Mark 925 MHz
s11_925_pt = grid.model_S(1, 1, np.array([925e6]))[0]
ax3.plot(np.real(s11_925_pt), np.imag(s11_925_pt), 'go', markersize=8,
         label='925 MHz')

ax3.set_xlim(-1.1, 1.1)
ax3.set_ylim(-1.1, 1.1)
ax3.set_aspect('equal')
ax3.set_title('Smith Chart', fontsize=14)
ax3.legend()
ax3.grid(False)
fig3.tight_layout()
fig3.savefig(reportdir / 'smith_chart.png', dpi=150)
print(f"  Saved {reportdir / 'smith_chart.png'}")

# --- Save Touchstone ---
with open(reportdir / 'verified.s1p', 'w') as f:
    f.write("! EMerge PIFA Verification - AU915 (915-928 MHz)\n")
    f.write(f"! patch_length={best_pL:.2f}mm, patch_width={best_pW:.2f}mm, "
            f"short_span={best_sW:.2f}mm, feed_offset={best_fo:.2f}mm\n")
    f.write("# MHZ S RI R 50.0\n")
    for i in range(len(freqs_flat)):
        f.write(f"{freqs_flat[i]/1e6:.6f} "
                f"{np.real(s11_flat[i]):.8f} {np.imag(s11_flat[i]):.8f}\n")
print(f"  Saved {reportdir / 'verified.s1p'}")

print("\nDone.")
