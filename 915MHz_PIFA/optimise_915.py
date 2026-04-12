# optimise_915.py
# ================================================================
# Printed Shorted-Patch PIFA Optimiser for AU915 (915–928 MHz)
# ================================================================
# Topology: single rectangular patch on top copper, full bottom
# ground, and a plated-through via fence at the patch's closed
# edge acting as the short. Feed is a vertical lumped-port plate
# offset along the patch length. PCB-realistic: top copper,
# bottom copper, and plated vias only. 100×80 mm FR-4 board.
#
# Free optimisation parameters (4):
#   patch_length_mm   — sets resonance (λ/4 on FR-4 loaded patch)
#   patch_width_mm    — sets Q / bandwidth
#   short_span_mm     — Y extent of via fence: PIFA/IFA character + Z_in
#   feed_offset_mm    — fine tunes 50 Ω match
#
# Cost = max(S11_dB) across 915–928 MHz + distance-to-band penalty
# (so Powell always has a gradient when resonance drifts outside
# the ISM band).
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
F_TARGET = 915e6
F_CENTER = 0.5 * (F_BAND_LO + F_BAND_HI)    # 921.5 MHz

FAIL_COST = 0.0
outdir = Path("out")
outdir.mkdir(exist_ok=True)


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


# ----------------------------------------------------------------
# Shared model builder (inlines the geometry so we control the
# Simulation object lifecycle and can reset it between opt calls).
# ----------------------------------------------------------------
def build_full_model(
    sim,
    L_mm=100.0,
    W_mm=80.0,
    patch_length_mm=45.0,
    patch_width_mm=25.0,
    patch_margin_x_mm=5.0,
    patch_margin_y_mm=5.0,
    short_span_mm=12.0,
    n_short_vias=6,
    via_pad_mm=0.6,
    feed_offset_mm=6.0,
    feed_trace_w_mm=1.5,
    f0=915e6,
    margin_lambda=0.25,
    reset_model=True,
):
    if reset_model:
        sim.reset(all=True)

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

    # --------------------------------------------------------
    # Patch placement + validity
    # --------------------------------------------------------
    patch_x0 = patch_margin_x
    patch_x1 = patch_x0 + patch_length
    patch_y1 = W - patch_margin_y
    patch_y0 = patch_y1 - patch_width
    patch_y_center = 0.5 * (patch_y0 + patch_y1)

    if patch_x1 > L - patch_margin_x:
        raise ValueError(f"patch_length {patch_length_mm:.1f}mm overflows board X")
    if patch_y0 < 0:
        raise ValueError(f"patch overflows board Y (patch_y0={patch_y0*1e3:.1f}mm)")
    if short_span > patch_width:
        raise ValueError(
            f"short_span {short_span_mm:.1f}mm > patch_width {patch_width_mm:.1f}mm"
        )
    if feed_offset <= 0 or feed_offset >= patch_length:
        raise ValueError(
            f"feed_offset {feed_offset_mm:.1f}mm outside (0, patch_length)"
        )
    if feed_trace_w >= short_span:
        raise ValueError("feed_trace_w must be smaller than short_span")
    if n_short_vias < 1:
        raise ValueError("n_short_vias must be >= 1")
    if via_pad > short_span:
        raise ValueError(f"via_pad {via_pad_mm:.2f}mm > short_span {short_span_mm:.1f}mm")
    if via_pad >= patch_length:
        raise ValueError("via_pad must be smaller than patch_length")

    short_y0 = patch_y_center - short_span / 2.0

    # --------------------------------------------------------
    # Substrate
    # --------------------------------------------------------
    substrate = em.geo.Box(L, W, pcb_thickness, position=(0, 0, -pcb_thickness))
    substrate.set_material(em.Material(er=4.4, tand=0.02, color="#2d8c2d", opacity=0.6))

    # --------------------------------------------------------
    # Bottom ground plane (full board)
    # --------------------------------------------------------
    bottom_gnd = em.geo.XYPlate(L, W, position=(0, 0, -pcb_thickness))
    bottom_gnd.set_material(em.lib.PEC)

    # --------------------------------------------------------
    # Top copper: patch + plated-through via fence acting as
    # the short. Each via is a square-cross-section Box
    # approximating a circular plated hole; it extends from
    # just below the bottom ground (z = -pcb_thickness -
    # copper_thick) up through the top of the patch
    # (z = copper_thick), giving face-to-face weld contact
    # with both the bottom-ground XYPlate and the patch Box.
    # Each via is united with the patch so GMSH treats the
    # whole short fence as one PEC entity. Via pitch sits
    # well below λ/10 at 915 MHz so the fence behaves like a
    # continuous short wall. PCB-manufacturable as plated
    # through-holes on a 2-layer board.
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Feed port plate: vertical plate at x = patch_x0 + feed_offset
    # --------------------------------------------------------
    feed_x = patch_x0 + feed_offset
    port_face = em.geo.Plate(
        np.array([feed_x, patch_y_center - feed_trace_w / 2.0, -pcb_thickness]),
        np.array([0.0, feed_trace_w, 0.0]),
        np.array([0.0, 0.0, pcb_thickness])
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
    # Commit + mesh + BCs
    # --------------------------------------------------------
    sim.commit_geometry()
    sim.generate_mesh()

    sim.mw.bc.AbsorbingBoundary(air.boundary(), order=2, abctype='B')
    sim.mw.bc.LumpedPort(
        port_face, 1,
        width=feed_trace_w, height=pcb_thickness,
        direction=em.ZAX, Z0=50.0
    )

    return {
        "air": air, "port_face": port_face,
        "patch": patch,
        "L": L, "W": W,
        "patch_length": patch_length, "patch_width": patch_width,
        "short_span": short_span, "feed_offset": feed_offset,
    }


# ----------------------------------------------------------------
# Fresh-sim evaluation
# ----------------------------------------------------------------
# CRITICAL: every evaluation builds a BRAND-NEW em.Simulation object.
# A previous implementation reused a single Simulation across Powell
# iterations and the Stage-2c 1D polish. When one iteration hit a GMSH
# mesh error (easy to do when patch_length lands on a value where the
# via fence, patch box and port plate fragment badly), the sim entered
# an undefined state and every subsequent build returned ghost data
# from the last-good sweep or raised "list index out of range". The
# optimiser then happily picked a corrupted sample as its best result
# and the final fresh-sim run revealed a totally different answer.
#
# Fresh sim per eval is strictly slower but is the only way to get
# results you can trust. We also use a WIDE sweep (500–1200 MHz) so
# resonances outside our expected band can't silently clip against
# the sweep edges and be reported as fake edge-frequency dips.
def eval_fresh(pL_mm, pW_mm, sW_mm, fo_mm,
               sweep_lo=500e6, sweep_hi=1200e6, n_pts=71,
               resolution=0.25, tag="PIFAEval"):
    """
    Build one antenna geometry in a fresh Simulation and return
    (resonance_mhz, worst_ism_dB, s11_915_dB, freqs_hz, s11_complex).
    On any failure (mesh error, solver error) returns None so the
    caller can discard the sample cleanly.
    """
    sim = em.Simulation(tag, loglevel="ERROR")
    try:
        set_best_solver(sim)
        sim.mw.set_frequency_range(sweep_lo, sweep_hi, n_pts)
        sim.mw.set_resolution(resolution)
        build_full_model(
            sim,
            patch_length_mm=pL_mm,
            patch_width_mm=pW_mm,
            short_span_mm=sW_mm,
            feed_offset_mm=fo_mm,
            reset_model=False,  # fresh sim, nothing to reset
        )
        data = sim.mw.run_sweep(parallel=True)
        grid = data.scalar.grid

        f_sweep = np.array(grid.freq).flatten()
        s11_sweep = np.array(grid.S(1, 1)).flatten()
        s11_sweep_dB = 20.0 * np.log10(np.abs(s11_sweep))

        res_idx = int(np.argmin(s11_sweep_dB))
        res_freq_mhz = float(f_sweep[res_idx] / 1e6)

        # Clip detection: if the argmin is on the very edge of the
        # sweep, the true resonance is outside our window and the
        # reading is meaningless.
        clipped = (res_idx == 0) or (res_idx == len(f_sweep) - 1)

        ism_dense = np.linspace(F_BAND_LO, F_BAND_HI, 27)
        s11_ism_dB = np.interp(ism_dense, f_sweep, s11_sweep_dB)
        worst_in_ism = float(np.max(s11_ism_dB))
        s11_915_dB = float(np.interp(F_TARGET, f_sweep, s11_sweep_dB))

        return {
            "res_mhz": res_freq_mhz,
            "worst_ism_dB": worst_in_ism,
            "s11_915_dB": s11_915_dB,
            "clipped": clipped,
            "freqs": f_sweep,
            "s11": s11_sweep,
        }
    except Exception as e:
        print(f"    eval_fresh FAIL: {e}")
        return None
    finally:
        try:
            sim.clean()
        except Exception:
            pass


# ================================================================
# Fixed secondary parameters + pW candidate set.
# ================================================================
# short_span is held fixed — it primarily sets Z_in, which the feed-
# offset scan cleans up. patch_width is CHOSEN by a pW scan (Step 0)
# because it controls Q and therefore -10 dB bandwidth: the AU915
# ISM band is 13 MHz wide and a narrow patch's dip isn't wide enough
# to cover it. We want the widest pW that still fits the board and
# can place its resonance in band.
SW_FIXED = 12.0
FO_DEFAULT = 6.0
PW_CANDIDATES = [18.0, 22.0, 26.0, 30.0]
# Set by Step 0.
PW_FIXED = None  # type: ignore


def _pick_pL_for_band(scan_pL, scan_res, target_lo=F_BAND_LO / 1e6,
                      target_hi=F_BAND_HI / 1e6):
    """
    From a (pL, resonance_mhz) scan, return the pL that lands inside
    [target_lo, target_hi]. If nothing lands in-band, linearly
    interpolate on the monotone segment to estimate the pL that
    would hit the band centre.

    scan_pL and scan_res must be parallel sequences of the SAME
    monotone-cleaned data (resonance strictly decreasing in pL).
    """
    target_c = 0.5 * (target_lo + target_hi)
    arr = sorted(zip(scan_pL, scan_res), key=lambda t: t[0])
    pLs = [t[0] for t in arr]
    rms = [t[1] for t in arr]
    # In-band candidates — pick closest to band centre
    in_band = [(pL, rm) for pL, rm in arr if target_lo <= rm <= target_hi]
    if in_band:
        return min(in_band, key=lambda t: abs(t[1] - target_c))[0]
    # Straddle? Linear interp on adjacent pair that brackets target_c
    for i in range(len(arr) - 1):
        r0, r1 = rms[i], rms[i + 1]
        lo, hi = (r0, r1) if r0 < r1 else (r1, r0)
        if lo <= target_c <= hi:
            # Linear in pL for interpolation
            frac = (target_c - r0) / (r1 - r0)
            return pLs[i] + frac * (pLs[i + 1] - pLs[i])
    return None


def _enforce_monotone(scan):
    """
    Given a list of dicts with keys pL and res_mhz, keep only the
    largest monotonically-decreasing (in res_mhz with increasing pL)
    subset starting from the end with the HIGHEST resonance. This
    filters out the ghost-data / corrupted outliers we saw when a
    single sim object was reused across builds — real PIFA physics
    is strictly monotone.
    """
    if not scan:
        return []
    scan = sorted(scan, key=lambda d: d["pL"])
    clean = [scan[0]]
    for d in scan[1:]:
        if d["res_mhz"] < clean[-1]["res_mhz"]:
            clean.append(d)
        else:
            print(f"    [monotone filter] dropping pL={d['pL']:.2f}mm "
                  f"res={d['res_mhz']:.0f} MHz "
                  f"(prev {clean[-1]['res_mhz']:.0f} MHz at "
                  f"pL={clean[-1]['pL']:.2f}mm — not monotone)")
    return clean


# ================================================================
# STEP 0: pW SCAN (bandwidth selection)
# ================================================================
# For each candidate patch width, do a coarse pL probe to find the
# patch length that places resonance in the ISM band, then evaluate
# at that point. The pW with the best (most negative) worst-ISM wins
# — because the −10 dB bandwidth of the narrow patch (pW≈18) isn't
# wide enough to cover 915–928, whereas a wider radiating element
# lowers Q and flattens the dip.
print("=" * 60)
print("STEP 0: Patch-width scan (bandwidth selection)")
print("=" * 60)

COARSE_PL_PROBE = np.linspace(26.0, 42.0, 9)  # 2 mm steps

pw_results = []
for pw in PW_CANDIDATES:
    print(f"\n-- pW = {pw:.1f} mm --")
    probe = []
    for pL_try in COARSE_PL_PROBE:
        print(f"  pL={pL_try:.1f}mm ...", flush=True)
        r = eval_fresh(float(pL_try), pw, SW_FIXED, FO_DEFAULT,
                       tag="PIFA_PW")
        if r is None:
            print("    → failed, discarded")
            continue
        if r["clipped"]:
            print(f"    → clipped ({r['res_mhz']:.0f} MHz), discarded")
            continue
        probe.append({
            "pL": float(pL_try),
            "res_mhz": r["res_mhz"],
            "worst_ism_dB": r["worst_ism_dB"],
        })
        print(f"    → res={r['res_mhz']:.0f} MHz, "
              f"worst ISM={r['worst_ism_dB']:+.2f} dB")

    probe = _enforce_monotone(probe)
    if not probe:
        print("  (no valid probe points)")
        continue

    est_pL_pw = _pick_pL_for_band(
        [d["pL"] for d in probe],
        [d["res_mhz"] for d in probe],
    )
    if est_pL_pw is None:
        print("  (probe did not straddle ISM band)")
        continue

    # Evaluate exactly at the estimated in-band pL for this pW
    print(f"  → probing est pL={est_pL_pw:.2f}mm")
    r_est = eval_fresh(est_pL_pw, pw, SW_FIXED, FO_DEFAULT, tag="PIFA_PW_est")
    if r_est is None or r_est["clipped"]:
        print("    (eval at estimated pL failed)")
        continue
    print(f"    → res={r_est['res_mhz']:.0f} MHz, "
          f"worst ISM={r_est['worst_ism_dB']:+.2f} dB, "
          f"S11@915={r_est['s11_915_dB']:+.2f} dB")

    pw_results.append({
        "pW": pw,
        "pL": est_pL_pw,
        "res_mhz": r_est["res_mhz"],
        "worst_ism_dB": r_est["worst_ism_dB"],
        "s11_915_dB": r_est["s11_915_dB"],
    })

if not pw_results:
    raise SystemExit("pW scan produced no usable candidates.")

# Pick the pW with the best worst-ISM across the band. That's the
# correct proxy for the ISM spec — deeper dip at band centre is
# irrelevant if the edges aren't under −10 dB.
best_pw_pt = min(pw_results, key=lambda d: d["worst_ism_dB"])
PW_FIXED = best_pw_pt["pW"]
est_pL = best_pw_pt["pL"]

print("\npW scan summary:")
for d in pw_results:
    marker = "  <-- picked" if d["pW"] == PW_FIXED else ""
    print(
        f"  pW={d['pW']:.1f}mm  pL={d['pL']:.2f}mm  "
        f"res={d['res_mhz']:.0f} MHz  worst-ISM={d['worst_ism_dB']:+.2f} dB"
        f"{marker}"
    )
print(f"\n  Selected pW={PW_FIXED:.1f}mm, starting pL={est_pL:.2f}mm")

# ================================================================
# STEP 1: (skipped — pW scan already produced a pL estimate)
# ================================================================

# ================================================================
# STEP 2: pL FINE SCAN around the estimate
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Patch-length fine scan")
print("=" * 60)

fine_pL = np.linspace(est_pL - 1.0, est_pL + 1.0, 11)  # 0.2 mm steps
fine_scan = []
for pL_try in fine_pL:
    print(f"  pL={pL_try:.3f}mm ...", flush=True)
    r = eval_fresh(float(pL_try), PW_FIXED, SW_FIXED, FO_DEFAULT,
                   tag="PIFAScan2")
    if r is None:
        print("    → failed, discarded")
        continue
    if r["clipped"]:
        print("    → clipped, discarded")
        continue
    fine_scan.append({
        "pL": float(pL_try),
        "res_mhz": r["res_mhz"],
        "worst_ism_dB": r["worst_ism_dB"],
        "s11_915_dB": r["s11_915_dB"],
    })
    print(f"    → res={r['res_mhz']:.0f} MHz, "
          f"worst ISM={r['worst_ism_dB']:+.2f} dB, "
          f"S11@915={r['s11_915_dB']:+.2f} dB")

fine_scan = _enforce_monotone(fine_scan)
if not fine_scan:
    raise SystemExit("Fine pL scan produced no valid points.")

# Prefer a point that's actually in-band. If multiple, pick the one
# with the deepest S11 across the band (best match).
in_band_pts = [d for d in fine_scan
               if F_BAND_LO / 1e6 <= d["res_mhz"] <= F_BAND_HI / 1e6]
if in_band_pts:
    best_fine = min(in_band_pts, key=lambda d: d["worst_ism_dB"])
    print(f"\n  {len(in_band_pts)} fine-scan points landed in-band. "
          f"Picking pL={best_fine['pL']:.3f}mm with worst-ISM "
          f"{best_fine['worst_ism_dB']:+.2f} dB.")
else:
    # Closest-to-centre if nothing landed inside
    best_fine = min(fine_scan, key=lambda d: abs(d["res_mhz"] - F_CENTER / 1e6))
    print(f"\n  WARNING: no fine-scan point landed in-band. "
          f"Closest is pL={best_fine['pL']:.3f}mm at "
          f"{best_fine['res_mhz']:.0f} MHz.")

best_pL_scan = best_fine["pL"]

# ================================================================
# STEP 3: feed_offset scan for matching
# ================================================================
# With pL fixed, sweep feed_offset to minimise S11 across 915–928.
# Short feed offset → high impedance, large offset → low impedance.
# The optimum is at the 50 Ω point on the patch.
print("\n" + "=" * 60)
print("STEP 3: Feed-offset scan (matching)")
print("=" * 60)

fo_candidates = np.linspace(2.0, 18.0, 17)  # 1.0 mm steps
fo_scan = []
for fo_try in fo_candidates:
    print(f"  fo={fo_try:.2f}mm ...", flush=True)
    r = eval_fresh(best_pL_scan, PW_FIXED, SW_FIXED, float(fo_try),
                   tag="PIFAFoScan")
    if r is None:
        print("    → failed, discarded")
        continue
    fo_scan.append({
        "fo": float(fo_try),
        "res_mhz": r["res_mhz"],
        "worst_ism_dB": r["worst_ism_dB"],
        "s11_915_dB": r["s11_915_dB"],
    })
    print(f"    → res={r['res_mhz']:.0f} MHz, "
          f"worst ISM={r['worst_ism_dB']:+.2f} dB, "
          f"S11@915={r['s11_915_dB']:+.2f} dB")

if not fo_scan:
    raise SystemExit("Feed-offset scan produced no valid points.")

best_fo_pt = min(fo_scan, key=lambda d: d["worst_ism_dB"])
best_fo_scan = best_fo_pt["fo"]
print(f"\n  Best feed_offset = {best_fo_scan:.2f} mm "
      f"(worst-ISM {best_fo_pt['worst_ism_dB']:+.2f} dB, "
      f"res {best_fo_pt['res_mhz']:.0f} MHz)")

# If changing fo moved the resonance outside the band (it shouldn't
# much, but fo does load the patch slightly), do one more tiny pL
# nudge to recentre.
if not (F_BAND_LO / 1e6 <= best_fo_pt["res_mhz"] <= F_BAND_HI / 1e6):
    print("\n  Feed-offset change pushed resonance out of band — "
          "nudging pL to recentre.")
    # Use the slope from the fine pL scan (≈ MHz per mm) to estimate
    # the correction.
    if len(fine_scan) >= 2:
        slope = ((fine_scan[-1]["res_mhz"] - fine_scan[0]["res_mhz"]) /
                 (fine_scan[-1]["pL"] - fine_scan[0]["pL"]))
        df = (F_CENTER / 1e6) - best_fo_pt["res_mhz"]
        dpL = df / slope if slope != 0 else 0.0
        best_pL_scan = best_pL_scan + dpL
        print(f"  Nudged pL → {best_pL_scan:.3f} mm (slope "
              f"{slope:+.1f} MHz/mm, Δf={df:+.0f} MHz)")

pW_fixed = PW_FIXED
sW_fixed = SW_FIXED
polish_pL = best_pL_scan
polish_res = best_fo_pt["res_mhz"]
polish_worst = best_fo_pt["worst_ism_dB"]

# ================================================================
# STEP 4: FINAL SIMULATION with scan-picked parameters
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: Final simulation")
print("=" * 60)

best_pL = polish_pL
best_pW = pW_fixed
best_sW = sW_fixed
best_fo = best_fo_scan
best_cost = polish_worst

print("Best parameters from scan:")
print(f"  patch_length: {best_pL:.2f} mm")
print(f"  patch_width:  {best_pW:.2f} mm")
print(f"  short_span:   {best_sW:.2f} mm")
print(f"  feed_offset:  {best_fo:.2f} mm")
print(f"  scan worst-ISM: {best_cost:+.2f} dB")

sim = em.Simulation("PIFAFinal", loglevel="INFO")
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
freqs = grid.freq
s11 = grid.S(1, 1)
freqs_flat = np.array(freqs).flatten()
s11_flat = np.array(s11).flatten()

with open(outdir / "optimised.s1p", "w") as f:
    f.write("! EMerge Printed Shorted-Patch PIFA - Optimised for AU915 (915-928 MHz)\n")
    f.write(
        f"! patch_length={best_pL:.2f}mm, patch_width={best_pW:.2f}mm, "
        f"short_span={best_sW:.2f}mm, feed_offset={best_fo:.2f}mm\n"
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
final_res_mhz = float(freqs_flat[idx_best] / 1e6)

ism_eval_freqs = np.linspace(F_BAND_LO, F_BAND_HI, 14)
s11_ism_verify = grid.model_S(1, 1, ism_eval_freqs)
s11_ism_dB = 20 * np.log10(np.abs(s11_ism_verify))
ism_worst = float(np.max(s11_ism_dB))
ism_worst_mhz = float(ism_eval_freqs[int(np.argmax(s11_ism_dB))] / 1e6)
ism_best = float(np.min(s11_ism_dB))
ism_best_mhz = float(ism_eval_freqs[int(np.argmin(s11_ism_dB))] / 1e6)

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
in_band = (F_BAND_LO / 1e6) <= final_res_mhz <= (F_BAND_HI / 1e6)
matched = ism_worst <= -10.0
status = "PASS" if (in_band and matched) else "FAIL"
print(f"    resonance in band (915–928): {'YES' if in_band else 'NO'} "
      f"({final_res_mhz:.0f} MHz)")
print(f"    spec (<= -10 dB across band + resonance in band): {status}")
if not in_band:
    print(
        f"\n  *** RESONANCE OUT OF BAND at {final_res_mhz:.0f} MHz. "
        f"Do NOT trust these parameters — the cost function's distance "
        f"penalty made the optimiser converge on the wrong side of the "
        f"band. Widen the calibration grid or Stage-1 bounds. ***"
    )
print(
    f"\n  Optimal params: patch_length={best_pL:.2f}mm, patch_width={best_pW:.2f}mm, "
    f"short_span={best_sW:.2f}mm, feed_offset={best_fo:.2f}mm "
    f"(shorted-patch PIFA, 100x80 FR-4)"
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
