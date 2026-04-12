# kicad_export.py
# ================================================================
# Emit a KiCad footprint (.kicad_mod) for the optimised shorted-patch
# PIFA so you can drop a single file into a KiCad library and place
# the antenna like any other component.
#
# The footprint contains:
#   - Patch copper on F.Cu, split into TWO rectangular SMD pads that
#     leave a small clearance around the feed via (KiCad cannot cut
#     holes in SMD pads, so we flank the feed via instead). Both sub
#     pads use pad number "1".
#   - N plated-through short-fence vias along the closed edge, also
#     pad "1" (same net as the patch → GND).
#   - One plated-through feed via, pad "2" (your RF signal net).
#   - Silkscreen + F.Fab outlines for visibility.
#
# Net assignment in KiCad after placement:
#   pad 1 → GND (connects patch + all short-fence vias to your
#                bottom-layer ground pour)
#   pad 2 → your RF feed net (route a 50 Ω trace to this pad)
#
# Layout constraint the footprint itself cannot enforce:
#   Keep the bottom copper pour UNDER the patch unbroken — no splits,
#   no traces, no silkscreen cutouts. The antenna's only ground
#   reference is that continuous pour.
#
# Origin convention: (0, 0) is the BOTTOM-LEFT corner of the patch
# on the shorted edge, so placing the footprint at (x0, y0) puts the
# closed (shorted) edge at board X = x0. Suggested placement on the
# 100 × 80 mm FR-4 board is origin at (5, 27) which centres the
# 26 mm patch on the top half of the board with a 5 mm margin from
# the left edge.
#
# Units in KiCad footprints are millimetres — no scaling needed.
# ================================================================

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class PifaParams:
    # Final optimised values from optimise_915.py (AU915 @ ~925 MHz).
    patch_length_mm: float = 34.42
    patch_width_mm: float = 26.00
    short_span_mm: float = 12.00
    feed_offset_mm: float = 7.00
    n_short_vias: int = 6
    via_pad_mm: float = 0.60       # via annular ring diameter (top copper)
    via_drill_mm: float = 0.30     # finished hole
    feed_pad_mm: float = 0.80      # feed via pad — slightly larger
    feed_drill_mm: float = 0.30
    feed_clearance_mm: float = 0.25  # minimum copper clearance to feed via
    name: str = "PIFA_915_AU915"


def export_kicad_footprint(params: PifaParams, out_path: Path):
    pL = params.patch_length_mm
    pW = params.patch_width_mm
    sS = params.short_span_mm
    fo = params.feed_offset_mm
    nv = params.n_short_vias
    vp = params.via_pad_mm
    vd = params.via_drill_mm
    fp = params.feed_pad_mm
    fd = params.feed_drill_mm
    fc = params.feed_clearance_mm

    # ----------------------------------------------------------------
    # Short-fence via centres (Y) — matches geometry.py exactly
    # ----------------------------------------------------------------
    short_y0 = pW / 2.0 - sS / 2.0
    if nv == 1:
        via_ys = [pW / 2.0]
    else:
        via_ys = np.linspace(
            short_y0 + vp / 2.0,
            short_y0 + sS - vp / 2.0,
            nv,
        ).tolist()
    via_x = vp / 2.0  # via centre sits via_pad/2 from the short edge

    # ----------------------------------------------------------------
    # Feed via
    # ----------------------------------------------------------------
    feed_x = fo
    feed_y = pW / 2.0

    # ----------------------------------------------------------------
    # Patch split: two rectangular sub-pads flanking the feed via.
    # Required clearance on each side of the feed via centre:
    #   feed_pad_radius + clearance
    # ----------------------------------------------------------------
    feed_half = fp / 2.0 + fc
    left_x1 = max(0.0, feed_x - feed_half)
    right_x0 = min(pL, feed_x + feed_half)
    left_width = left_x1
    right_width = pL - right_x0
    left_center = (left_width / 2.0, pW / 2.0)
    right_center = (right_x0 + right_width / 2.0, pW / 2.0)

    # ----------------------------------------------------------------
    # Compose the s-expression
    # ----------------------------------------------------------------
    L = []
    add = L.append
    add(f'(footprint "{params.name}"')
    add('\t(version 20240108)')
    add('\t(generator "emerge_pifa_export")')
    add('\t(layer "F.Cu")')
    descr = (
        f"Shorted-patch PIFA for AU915 (915-928 MHz) on 1.6 mm FR-4. "
        f"Patch {pL:.2f} x {pW:.2f} mm, feed offset {fo:.2f} mm, "
        f"{nv} plated through-vias along the shorted edge. "
        f"Simulated resonance ~925 MHz, best match -35 dB. "
        f"Pad 1 = GND (patch + short fence). Pad 2 = RF feed."
    )
    add(f'\t(descr "{descr}")')
    add('\t(tags "antenna PIFA 915MHz LoRa AU915 Wi-Fi-HaLow shorted-patch")')
    add('\t(attr through_hole)')

    # Text fields
    add(f'\t(fp_text reference "REF**" (at {pL/2:.3f} -2.0) (layer "F.SilkS")')
    add('\t\t(effects (font (size 1 1) (thickness 0.15))))')
    add(f'\t(fp_text value "{params.name}" (at {pL/2:.3f} {pW+2.0:.3f}) (layer "F.Fab")')
    add('\t\t(effects (font (size 1 1) (thickness 0.15))))')
    add(f'\t(fp_text user "SHORT EDGE" (at -1.8 {pW/2:.3f} 90) (layer "F.Fab")')
    add('\t\t(effects (font (size 0.8 0.8) (thickness 0.12))))')

    # Silkscreen patch outline
    add(f'\t(fp_rect (start 0 0) (end {pL:.3f} {pW:.3f})')
    add('\t\t(stroke (width 0.12) (type solid)) (fill none) (layer "F.SilkS"))')

    # F.Fab outline + thick bar on the shorted edge
    add(f'\t(fp_rect (start 0 0) (end {pL:.3f} {pW:.3f})')
    add('\t\t(stroke (width 0.1) (type solid)) (fill none) (layer "F.Fab"))')
    add(f'\t(fp_line (start 0 0) (end 0 {pW:.3f})')
    add('\t\t(stroke (width 0.3) (type solid)) (layer "F.Fab"))')

    # Courtyard
    add(f'\t(fp_rect (start -0.5 -0.5) (end {pL+0.5:.3f} {pW+0.5:.3f})')
    add('\t\t(stroke (width 0.05) (type solid)) (fill none) (layer "F.CrtYd"))')

    # Patch sub-pads (pad "1")
    if left_width > 0.1:
        add(f'\t(pad "1" smd rect (at {left_center[0]:.3f} {left_center[1]:.3f})')
        add(f'\t\t(size {left_width:.3f} {pW:.3f}) (layers "F.Cu" "F.Mask"))')
    if right_width > 0.1:
        add(f'\t(pad "1" smd rect (at {right_center[0]:.3f} {right_center[1]:.3f})')
        add(f'\t\t(size {right_width:.3f} {pW:.3f}) (layers "F.Cu" "F.Mask"))')

    # Short-fence vias (pad "1" — same net as the patch)
    for vy in via_ys:
        add(f'\t(pad "1" thru_hole circle (at {via_x:.3f} {vy:.3f})')
        add(f'\t\t(size {vp:.3f} {vp:.3f}) (drill {vd:.3f}) (layers "*.Cu" "*.Mask"))')

    # Feed via (pad "2")
    add(f'\t(pad "2" thru_hole circle (at {feed_x:.3f} {feed_y:.3f})')
    add(f'\t\t(size {fp:.3f} {fp:.3f}) (drill {fd:.3f}) (layers "*.Cu" "*.Mask"))')

    add(')')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")

    print(f"Wrote KiCad footprint → {out_path}")
    print()
    print("  Geometry:")
    print(f"    patch          : {pL:.2f} x {pW:.2f} mm")
    print(f"    feed offset    : {fo:.2f} mm from shorted edge")
    print(f"    short fence    : {nv} vias, pitch "
          f"{(via_ys[-1]-via_ys[0])/(nv-1) if nv > 1 else 0:.2f} mm, "
          f"pad {vp} / drill {vd} mm")
    print(f"    feed via       : pad {fp} / drill {fd} mm")
    print(f"    patch sub-pads : "
          f"left={left_width:.2f}x{pW:.2f}, right={right_width:.2f}x{pW:.2f} "
          f"(gap {right_x0-left_x1:.2f} mm around feed via)")
    print()
    print("  Nets to assign after placing the footprint in KiCad:")
    print("    pad 1 → GND       (patch + short-fence vias)")
    print("    pad 2 → RF feed   (50 Ω trace from radio/SMA)")
    print()
    print("  Board constraints the footprint cannot enforce:")
    print("    - Bottom copper pour MUST be continuous under the patch")
    print("      (no splits, traces, or silk cutouts).")
    print("    - Do NOT populate other parts inside the courtyard.")
    print("    - Suggested placement on 100 x 80 mm board: origin at")
    print("      (5, 27) mm — leaves 5 mm margin from the left/top")
    print("      edges and centres the radiator on the upper board.")
    print()
    print("  To import in KiCad:")
    print("    1. Preferences → Manage Footprint Libraries → Add Library")
    print("       → point at a folder containing this .kicad_mod file")
    print("       (or: copy the file into an existing .pretty folder).")
    print("    2. In the PCB editor: Place → Add Footprint → select")
    print(f"       '{params.name}'.")
    print("    3. Assign pad 1 to GND and pad 2 to your RF net.")
    print("    4. Flood the back copper as a GND zone across the whole")
    print("       board — the footprint does not include it.")


if __name__ == "__main__":
    p = PifaParams()
    out = Path("out") / f"{p.name}.kicad_mod"
    export_kicad_footprint(p, out)
