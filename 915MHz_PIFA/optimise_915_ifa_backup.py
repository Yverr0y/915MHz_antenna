# optimise_915.py
# ================================================================
# Printed PIFA Optimiser for 915 MHz (AU915 ISM band: 915–928 MHz)
# ================================================================
# Strategy:
#   1. Scout sweep (300 MHz – 2 GHz) with nominal geometry to find
#      where the untuned design actually resonates.
#   2. Seed: rescale bar_length using f ∝ 1/L so the starting point
#      for Powell is already in the right neighbourhood.
#   3. Stage-1 optimise (coarse mesh res=0.4): Powell over
#      bar_length, bar_width, feed_offset, cost = max S11 across the
#      AU915 ISM band [915, 928] MHz. Rewards designs that cover the
#      *whole* band, not just one narrow dip inside it.
#   4. Stage-2 polish (fine mesh res=0.25): Powell over the same
#      parameters but with ±10 % bounds around the stage-1 optimum,
#      to correct the coarse-mesh frequency bias.
#   5. Final sweep + Touchstone / CSV / far-field exports.
# ================================================================

from pathlib import Path
import numpy as np
import emerge as em
from emerge.plot import plot_sp, plot_vswr, plot_ff, plot_ff_polar

mm = 0.001
C0 = 299_792_458.0

# AU915 ISM band: 915–928 MHz
F_BAND_LO = 915e6
F_BAND_HI = 928e6
F_TARGET = 915e6                          # label frequency (low edge)
F_CENTER = (F_BAND_LO + F_BAND_HI) / 2    # 921.5 MHz, used for seeding

# Cost assigned when build_full_model fails geometric validity or the
# solver blows up. S11 = 0 dB is "perfectly reflective" — worst case.
FAIL_COST = 0.0

outdir = Path("out")
outdir.mkdir(exist_ok=True)


def set_best_solver(sim):
    """MUMPS > UMFPACK > SuperLU fallback (same as the meander script)."""
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
    L_mm=100.0,
    W_mm=80.0,
    gnd_height_mm=45.0,
    bar_length_mm=40.0,
    bar_width_mm=4.0,
    bar_margin_x_mm=3.0,
    stub_gap_mm=3.0,
    feed_offset_mm=4.0,
    feed_penetration_mm=5.0,
    trace_w_mm=1.0,
    feed_clearance_mm=0.5,
    n_meander_legs=3,
    meander_gap_mm=1.5,
    f0=915e6,
    margin_lambda=0.25,
    reset_model=True,
):
    """
    Build the complete printed PIFA model inside the given Simulation:
    geometry + airbox + port + BCs + mesh.
    """
    if reset_model:
        sim.reset(all=True)

    L = L_mm * mm
    W = W_mm * mm
    gnd_height = gnd_height_mm * mm
    bar_length = bar_length_mm * mm
    bar_width = bar_width_mm * mm
    bar_margin_x = bar_margin_x_mm * mm
    stub_gap = stub_gap_mm * mm
    feed_offset = feed_offset_mm * mm
    feed_penetration = feed_penetration_mm * mm
    trace_w = trace_w_mm * mm
    feed_clearance = feed_clearance_mm * mm
    meander_gap = meander_gap_mm * mm
    if n_meander_legs < 1:
        raise ValueError("n_meander_legs must be >= 1")

    pcb_thickness = 1.6 * mm
    copper_thick = 0.035 * mm
    join = 0.05 * mm

    # --------------------------------------------------------
    # Derived positions + geometric validity
    # --------------------------------------------------------
    bar_x0 = bar_margin_x
    bar_y0 = gnd_height + stub_gap
    bar_x1 = bar_x0 + bar_length
    bar_y1 = bar_y0 + n_meander_legs * bar_width + (n_meander_legs - 1) * meander_gap
    short_x_centre = bar_x0 + trace_w / 2.0
    feed_x_centre = short_x_centre + feed_offset
    feed_y_bottom = gnd_height - feed_penetration

    if bar_x1 > L - bar_margin_x:
        raise ValueError(f"bar_length {bar_length_mm:.1f}mm overflows board X")
    if bar_y1 > W:
        raise ValueError(f"bar overflows board Y (bar_y1={bar_y1*1e3:.1f}mm > W={W*1e3:.1f}mm)")
    if feed_x_centre + trace_w / 2.0 > bar_x1:
        raise ValueError(
            f"feed_offset {feed_offset_mm:.1f}mm pushes feed past bar open end"
        )
    if feed_y_bottom < 0:
        raise ValueError("feed_penetration extends below board")

    # --------------------------------------------------------
    # Substrate
    # --------------------------------------------------------
    substrate = em.geo.Box(L, W, pcb_thickness, position=(0, 0, -pcb_thickness))
    substrate.set_material(em.Material(er=4.4, tand=0.02, color="#2d8c2d", opacity=0.6))

    # --------------------------------------------------------
    # Bottom ground plane
    # --------------------------------------------------------
    bottom_gnd = em.geo.XYPlate(L, W, position=(0, 0, -pcb_thickness))
    bottom_gnd.set_material(em.lib.PEC)

    # --------------------------------------------------------
    # Top ground plane with feed slot
    # --------------------------------------------------------
    gnd_full = em.geo.Box(L, gnd_height, copper_thick, position=(0, 0, 0))
    slot_left = feed_x_centre - trace_w / 2.0 - feed_clearance
    slot_y_bot = feed_y_bottom - feed_clearance
    slot_width = trace_w + 2 * feed_clearance
    slot_h = gnd_height - slot_y_bot
    slot = em.geo.Box(
        slot_width, slot_h, copper_thick,
        position=(slot_left, slot_y_bot, 0)
    )
    gnd = em.geo.subtract(gnd_full, slot)

    # --------------------------------------------------------
    # F-shape antenna trace
    # --------------------------------------------------------
    ov = min(join, 0.25 * trace_w)

    # Meandered radiating element (N horizontal legs + alternating connectors)
    meander_parts = []
    for i in range(n_meander_legs):
        leg_y = bar_y0 + i * (bar_width + meander_gap)
        leg = em.geo.Box(
            bar_length, bar_width, copper_thick,
            position=(bar_x0, leg_y, 0)
        )
        meander_parts.append(leg)
    for i in range(n_meander_legs - 1):
        conn_x = (bar_x1 - trace_w) if (i % 2 == 0) else bar_x0
        conn_y_bot = bar_y0 + i * (bar_width + meander_gap) + bar_width - ov
        conn = em.geo.Box(
            trace_w, meander_gap + 2 * ov, copper_thick,
            position=(conn_x, conn_y_bot, 0)
        )
        meander_parts.append(conn)
    bar = em.geo.unite(*meander_parts) if len(meander_parts) > 1 else meander_parts[0]

    short_stub = em.geo.Box(
        trace_w, stub_gap + 2 * ov, copper_thick,
        position=(short_x_centre - trace_w / 2.0, gnd_height - ov, 0)
    )

    feed_stub_len = (bar_y0 - feed_y_bottom) + ov
    feed_stub = em.geo.Box(
        trace_w, feed_stub_len, copper_thick,
        position=(feed_x_centre - trace_w / 2.0, feed_y_bottom, 0)
    )

    top_copper = em.geo.unite(gnd, bar, short_stub, feed_stub)
    top_copper.set_material(em.lib.MET_COPPER)

    # --------------------------------------------------------
    # Vertical port face
    # --------------------------------------------------------
    port_face = em.geo.Plate(
        np.array([feed_x_centre - trace_w / 2.0, feed_y_bottom, -pcb_thickness]),
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
    # Commit + mesh
    # --------------------------------------------------------
    sim.commit_geometry()
    sim.generate_mesh()

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------
    sim.mw.bc.AbsorbingBoundary(air.boundary(), order=2, abctype='B')
    sim.mw.bc.LumpedPort(
        port_face, 1,
        width=trace_w, height=pcb_thickness,
        direction=em.ZAX, Z0=50.0
    )

    return {
        "air": air, "port_face": port_face,
        "L": L, "W": W,
        "bar_length": bar_length, "bar_width": bar_width,
        "feed_offset": feed_offset,
    }


# ----------------------------------------------------------------
# Inner evaluation: one build + sweep, returns ISM-band cost.
# Shared by both optimisation stages.
# ----------------------------------------------------------------
def evaluate(sim, bar_L_mm, bar_W_mm, feed_off_mm, stub_gap_mm, gnd_h_mm,
             resolution, sweep_lo=820e6, sweep_hi=1010e6, n_pts=31):
    """
    Build the model with the given parameters, run a narrow sweep,
    and return (cost, s11_at_915_dB, resonance_MHz).

    cost = max(S11_dB) across 915-928 MHz (AU915). Lower is better.
    Designs where the whole ISM band sits below -10 dB score < -10.
    """
    sim.mw.set_frequency_range(sweep_lo, sweep_hi, n_pts)
    sim.mw.set_resolution(resolution)

    build_full_model(
        sim,
        gnd_height_mm=gnd_h_mm,
        bar_length_mm=bar_L_mm,
        bar_width_mm=bar_W_mm,
        feed_offset_mm=feed_off_mm,
        stub_gap_mm=stub_gap_mm,
    )

    data = sim.mw.run_sweep(parallel=True)
    grid = data.scalar.slice_set(-n_pts).grid

    # Evaluate S11 at 1 MHz spacing across the ISM band.
    ism_freqs = np.linspace(F_BAND_LO, F_BAND_HI, 14)  # 915..928, 1 MHz steps
    s11_ism = grid.model_S(1, 1, ism_freqs)
    s11_ism_dB = 20 * np.log10(np.abs(s11_ism))
    worst_in_ism = float(np.max(s11_ism_dB))

    # Also report the 915 MHz value and the actual resonance location
    # (useful for diagnostic printing, not used as cost).
    s11_at_915 = grid.model_S(1, 1, np.array([F_TARGET]))[0]
    s11_915_dB = float(20 * np.log10(np.abs(s11_at_915)))

    wide = np.linspace(sweep_lo, sweep_hi, 401)
    s11_wide = 20 * np.log10(np.abs(grid.model_S(1, 1, wide)))
    res_freq_mhz = float(wide[np.argmin(s11_wide)] / 1e6)

    # Gradient shaping: when resonance is far from the ISM band, Powell
    # sees a flat 0 dB floor and wanders. Add a distance-to-band penalty
    # (0.05 dB per MHz outside 915–928) so there is always a direction
    # home. Inside the band, penalty = 0 and cost reduces to worst_in_ism.
    f_lo_mhz = F_BAND_LO / 1e6
    f_hi_mhz = F_BAND_HI / 1e6
    if res_freq_mhz < f_lo_mhz:
        distance = f_lo_mhz - res_freq_mhz
    elif res_freq_mhz > f_hi_mhz:
        distance = res_freq_mhz - f_hi_mhz
    else:
        distance = 0.0
    cost = worst_in_ism + 0.05 * distance

    return cost, s11_915_dB, res_freq_mhz


# ================================================================
# STEP 1: SCOUT SWEEP — find where the untuned design resonates
# ================================================================
print("=" * 60)
print("STEP 1: Scout sweep (300 MHz – 2 GHz)")
print("=" * 60)

SCOUT_BAR_L = 25.0  # meandered: ~half the straight-bar length for same f

scout = em.Simulation("PIFAScout", loglevel="INFO")
set_best_solver(scout)
scout.mw.set_frequency_range(300e6, 2e9, 51)
scout.mw.set_resolution(0.4)

build_full_model(
    scout,
    L_mm=100.0, W_mm=80.0,
    gnd_height_mm=45.0,
    bar_length_mm=SCOUT_BAR_L, bar_width_mm=4.0,
    feed_offset_mm=6.0,
)

data = scout.mw.run_sweep(parallel=True)
grid = data.scalar.grid
freqs = grid.freq
s11 = grid.S(1, 1)
s11_dB_scout = 20 * np.log10(np.abs(s11))

scout_res_idx = int(np.argmin(s11_dB_scout))
scout_res_hz = float(freqs[scout_res_idx])
print(f"\nScout resonance: {np.min(s11_dB_scout):.2f} dB @ {scout_res_hz/1e6:.0f} MHz")

np.savetxt(
    outdir / "scout_s11.csv",
    np.column_stack([freqs / 1e6, s11_dB_scout]),
    delimiter=",", header="freq_MHz,S11_dB", comments=""
)
plot_sp(np.array(freqs), s11)
scout.clean()

# ================================================================
# STEP 1b: SEED — rescale bar_length so f_scout → band centre
# Simple proportionality: resonant length ∝ 1/frequency. We aim for
# the *band centre* (921.5 MHz), not 915 MHz, so the initial design
# lands inside the band rather than at its low edge.
# ================================================================
seed_bar_L = SCOUT_BAR_L * (scout_res_hz / F_CENTER)
# Clamp well above the lower opt bound: if scout finds a spurious very-low
# resonance, do NOT let the seed collapse to 10 mm (which is almost certainly
# outside the real tuning range for a 3-leg meander on FR-4).
seed_bar_L = float(np.clip(seed_bar_L, 18.0, 38.0))
print(f"Seeding bar_length_mm = {seed_bar_L:.2f}  "
      f"(scaled from {SCOUT_BAR_L:.1f} targeting band centre {F_CENTER/1e6:.1f} MHz)")

# ================================================================
# STEP 2: STAGE-1 OPTIMISATION — coarse mesh, wide bounds
# Cost = max(S11_dB) across AU915 915–928 MHz.
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Stage-1 optimisation (coarse mesh, res=0.4)")
print("Cost = worst S11 across ISM AU915 915–928 MHz")
print("=" * 60)

sim = em.Simulation("PIFAOpt1", loglevel="INFO")
set_best_solver(sim)

# Changes vs first run:
# - bar_length ceiling raised 60 → 70 mm. Previous run pinned at 60 mm.
#   Hard max from board geom (L=80, margin=3) is 74, so 70 leaves headroom
#   for the validity check.
# - bar_width upper 10 → 12 mm for a bit more impedance latitude.
# - feed_offset initial 4 → 8 mm, upper 12 → 20 mm. Previous run left
#   feed_offset frozen at 4 mm; at 4/60 ≈ 6.7 % of bar length from the
#   short, Z_in ≈ 5 Ω → flat -1.4 dB reflection floor. 8 mm is a better
#   seed and 20 mm ceiling lets Powell walk up if it wants to.
# - stub_gap_mm added as a 4th parameter. Directly affects radiation
#   resistance and shifts resonance; exactly the lever missing last time.
sim.opt.add_param('bar_length_mm',  seed_bar_L, (15.0, 55.0))
sim.opt.add_param('bar_width_mm',           4.0, ( 1.5, 10.0))
sim.opt.add_param('feed_offset_mm',         6.0, ( 1.5, 20.0))
sim.opt.add_param('stub_gap_mm',            5.0, ( 3.0, 12.0))
sim.opt.add_param('gnd_height_mm',         45.0, (30.0, 55.0))
sim.opt.method = 'Powell'

for bar_L_opt, bar_W_opt, feed_off_opt, stub_gap_opt, gnd_h_opt in sim.opt.run(max_iter=60):
    print(
        f"\n  Trying: bar_L={bar_L_opt:.2f}mm, "
        f"bar_W={bar_W_opt:.2f}mm, feed_off={feed_off_opt:.2f}mm, "
        f"stub_gap={stub_gap_opt:.2f}mm, gnd_h={gnd_h_opt:.2f}mm"
    )
    try:
        cost, s11_915, res_mhz = evaluate(
            sim, bar_L_opt, bar_W_opt, feed_off_opt, stub_gap_opt, gnd_h_opt,
            resolution=0.4,
        )
        print(
            f"  worst S11 in ISM: {cost:+.2f} dB | "
            f"S11@915: {s11_915:+.2f} dB | "
            f"resonance: {res_mhz:.0f} MHz"
        )
        sim.opt.update(cost)
    except Exception as e:
        print(f"  ERROR: {e}")
        sim.opt.update(FAIL_COST)

stage1_solution, stage1_cost = sim.opt.best
s1_bL = float(stage1_solution['bar_length_mm'])
s1_bW = float(stage1_solution['bar_width_mm'])
s1_fo = float(stage1_solution['feed_offset_mm'])
s1_sg = float(stage1_solution['stub_gap_mm'])
s1_gh = float(stage1_solution['gnd_height_mm'])

print("\nStage-1 best:")
print(f"  bar_length  = {s1_bL:.2f} mm")
print(f"  bar_width   = {s1_bW:.2f} mm")
print(f"  feed_offset = {s1_fo:.2f} mm")
print(f"  stub_gap    = {s1_sg:.2f} mm")
print(f"  gnd_height  = {s1_gh:.2f} mm")
print(f"  worst S11 in ISM = {stage1_cost:+.2f} dB")

# ================================================================
# STEP 2b: STAGE-2 POLISH — fine mesh, ±10 % bounds around stage-1
# ================================================================
print("\n" + "=" * 60)
print("STEP 2b: Stage-2 polish (fine mesh, res=0.25)")
print("=" * 60)

# Fresh Simulation for stage 2. sim.reset(all=True) on the stage-1
# object does NOT clear sim.opt's parameter registry, so re-using it
# would stack a second set of params on top and opt.run would yield
# 6-tuples. Cleaner to start a new sim.
sim.clean()
sim = em.Simulation("PIFAOpt2", loglevel="INFO")
set_best_solver(sim)


def _band(v, frac=0.10, floor=None, ceil=None):
    lo = v * (1.0 - frac)
    hi = v * (1.0 + frac)
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    return (lo, hi)


sim.opt.add_param('bar_length_mm',  s1_bL, _band(s1_bL, 0.10, 15.0, 55.0))
sim.opt.add_param('bar_width_mm',   s1_bW, _band(s1_bW, 0.15,  1.5, 10.0))
sim.opt.add_param('feed_offset_mm', s1_fo, _band(s1_fo, 0.25,  1.5, 20.0))
sim.opt.add_param('stub_gap_mm',    s1_sg, _band(s1_sg, 0.20,  3.0, 12.0))
sim.opt.add_param('gnd_height_mm',  s1_gh, _band(s1_gh, 0.15, 30.0, 55.0))
sim.opt.method = 'Powell'

for bar_L_opt, bar_W_opt, feed_off_opt, stub_gap_opt, gnd_h_opt in sim.opt.run(max_iter=35):
    print(
        f"\n  [polish] bar_L={bar_L_opt:.2f}mm, "
        f"bar_W={bar_W_opt:.2f}mm, feed_off={feed_off_opt:.2f}mm, "
        f"stub_gap={stub_gap_opt:.2f}mm, gnd_h={gnd_h_opt:.2f}mm"
    )
    try:
        cost, s11_915, res_mhz = evaluate(
            sim, bar_L_opt, bar_W_opt, feed_off_opt, stub_gap_opt, gnd_h_opt,
            resolution=0.25,
        )
        print(
            f"  worst S11 in ISM: {cost:+.2f} dB | "
            f"S11@915: {s11_915:+.2f} dB | "
            f"resonance: {res_mhz:.0f} MHz"
        )
        sim.opt.update(cost)
    except Exception as e:
        print(f"  ERROR: {e}")
        sim.opt.update(FAIL_COST)

# ================================================================
# STEP 3: FINAL SIMULATION with polished parameters
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Final simulation with polished parameters")
print("=" * 60)

solution, best_s11 = sim.opt.best
best_bL = float(solution['bar_length_mm'])
best_bW = float(solution['bar_width_mm'])
best_fo = float(solution['feed_offset_mm'])
best_sg = float(solution['stub_gap_mm'])
best_gh = float(solution['gnd_height_mm'])

print("Best parameters from optimiser:")
print(f"  bar_length:   {best_bL:.2f} mm")
print(f"  bar_width:    {best_bW:.2f} mm")
print(f"  feed_offset:  {best_fo:.2f} mm")
print(f"  stub_gap:     {best_sg:.2f} mm")
print(f"  gnd_height:   {best_gh:.2f} mm")
print(f"  Stage-2 worst-S11 in ISM: {best_s11:+.2f} dB")

# Fresh sim for the final verification run — same reason as stage 2.
sim.clean()
sim = em.Simulation("PIFAFinal", loglevel="INFO")
set_best_solver(sim)

# Final verification sweep: wider frequency range at the polish-stage
# mesh resolution so the reported S11 matches what the optimiser saw.
sim.mw.set_frequency_range(700e6, 1300e6, 121)
sim.mw.set_resolution(0.25)

meta = build_full_model(
    sim,
    gnd_height_mm=best_gh,
    bar_length_mm=best_bL,
    bar_width_mm=best_bW,
    feed_offset_mm=best_fo,
    stub_gap_mm=best_sg,
)

data = sim.mw.run_sweep(parallel=True)

grid = data.scalar.grid
freqs = grid.freq
s11 = grid.S(1, 1)

freqs_flat = np.array(freqs).flatten()
s11_flat = np.array(s11).flatten()

# Manual Touchstone export
with open(outdir / "optimised.s1p", "w") as f:
    f.write("! EMerge Printed PIFA - Optimised for AU915 ISM (915-928 MHz)\n")
    f.write(
        f"! bar_length={best_bL:.2f}mm, bar_width={best_bW:.2f}mm, "
        f"feed_offset={best_fo:.2f}mm, stub_gap={best_sg:.2f}mm, "
        f"gnd_height={best_gh:.2f}mm, meander legs=3\n"
    )
    f.write("# MHZ S RI R 50.0\n")
    for i in range(len(freqs_flat)):
        f.write(
            f"{freqs_flat[i]/1e6:.6f} "
            f"{np.real(s11_flat[i]):.8f} {np.imag(s11_flat[i]):.8f}\n"
        )
print(f"Touchstone exported to {outdir / 'optimised.s1p'}")

s11_dB = 20 * np.log10(np.abs(s11_flat))
np.savetxt(
    outdir / "optimised_s11.csv",
    np.column_stack([
        freqs_flat / 1e6,
        np.real(s11_flat), np.imag(s11_flat),
        np.abs(s11_flat), s11_dB,
    ]),
    delimiter=",",
    header="freq_MHz,Re_S11,Im_S11,Mag_S11,S11_dB",
    comments=""
)
print(f"CSV exported to {outdir / 'optimised_s11.csv'}")

idx_915 = int(np.argmin(np.abs(freqs_flat - 915e6)))
idx_best = int(np.argmin(np.abs(s11_flat)))

# ISM-band summary from the verification sweep
ism_eval_freqs = np.linspace(F_BAND_LO, F_BAND_HI, 14)  # 915..928, 1 MHz steps
s11_ism_verify = grid.model_S(1, 1, ism_eval_freqs)
s11_ism_dB = 20 * np.log10(np.abs(s11_ism_verify))
ism_worst = float(np.max(s11_ism_dB))
ism_worst_mhz = float(ism_eval_freqs[int(np.argmax(s11_ism_dB))] / 1e6)
ism_best = float(np.min(s11_ism_dB))
ism_best_mhz = float(ism_eval_freqs[int(np.argmin(s11_ism_dB))] / 1e6)

# -10 dB bandwidth (contiguous span around the best match)
below_10 = s11_dB <= -10.0
if np.any(below_10):
    below_idx = np.where(below_10)[0]
    bw_lo_mhz = freqs_flat[below_idx[0]] / 1e6
    bw_hi_mhz = freqs_flat[below_idx[-1]] / 1e6
    bw_10 = bw_hi_mhz - bw_lo_mhz
else:
    bw_lo_mhz = bw_hi_mhz = bw_10 = float('nan')

print("\nFinal Results:")
print(f"  S11 @ 915 MHz:     {s11_dB[idx_915]:+.2f} dB")
print(
    f"  Best match:        {20*np.log10(np.abs(s11_flat[idx_best])):+.2f} dB "
    f"@ {freqs_flat[idx_best]/1e6:.1f} MHz"
)
print(f"  -10 dB bandwidth:  {bw_lo_mhz:.0f} – {bw_hi_mhz:.0f} MHz ({bw_10:.0f} MHz)")
print("\n  ISM band AU915 915–928 MHz:")
print(f"    worst S11:       {ism_worst:+.2f} dB @ {ism_worst_mhz:.0f} MHz")
print(f"    best S11:        {ism_best:+.2f} dB @ {ism_best_mhz:.0f} MHz")
status = "PASS" if ism_worst <= -10.0 else "FAIL"
print(f"    spec (<= -10 dB across band): {status}")
print(
    f"\n  Optimal params: bar_length={best_bL:.2f}mm, bar_width={best_bW:.2f}mm, "
    f"feed_offset={best_fo:.2f}mm, stub_gap={best_sg:.2f}mm, "
    f"gnd_height={best_gh:.2f}mm (3-leg meander, 100x80 board)"
)

# ================================================================
# PLOTS
# ================================================================
freq_dense = np.linspace(700e6, 1300e6, 1001)
try:
    s11_dense = grid.model_S(1, 1, freq_dense)
except Exception:
    s11_dense = s11_flat
    freq_dense = freqs_flat

plot_sp(freq_dense, s11_dense)
plot_vswr(freq_dense, s11_dense)

# Far-field at 915 MHz
try:
    abc_sel = meta["air"].boundary()
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

    sim.display.populate()
    ant_cx = meta["L"] / 2
    ant_cy = meta["W"] / 2
    ff3d = field_915.farfield_3d(abc_sel, origin=(ant_cx, ant_cy, 0))
    surf = ff3d.surfplot('normE', rmax=meta["L"] * 3, isotropic=True)
    sim.display.add_surf(*surf.xyzf)
    sim.display.show()
except Exception as e:
    print(f"Far-field plotting error: {e}")
    print("S-parameter plots completed successfully. Far-field skipped.")
