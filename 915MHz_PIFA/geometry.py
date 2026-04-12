# geometry.py
# PCB-realistic Shorted-Patch PIFA for 915 MHz (AU915 ISM band).
# 2-layer FR-4, no inner layers, all conductors on top or bottom copper
# plus plated-through vias — this is directly Gerber-generatable.
#
# Topology (X = board length, Y = board width, Z = through substrate):
#
#     +-----------------------------------------+  y = W
#     |  +==========patch=========+             |   ^ top edge of board
#     |  * * * *                  |             |   | fence of plated
#     |  (short                   x  feed       |   | through vias along
#     |   via fence)             (port)         |   | closed edge of patch.
#     |                                         |
#     |          full bottom ground             |
#     |          (on z = -pcb_thickness)        |
#     +-----------------------------------------+  y = 0
#     x = 0                                     x = L
#
# Layer stackup:
#   - Bottom copper (z = -pcb_thickness): full-board ground plane,
#     the antenna's only ground reference. No top ground plane.
#   - Top copper (z = 0 .. copper_thick): a single rectangular patch.
#   - Short fence: a row of N plated-through vias at the patch's
#     closed edge, each a ~0.6 mm square (approximating a circular
#     plated hole). Vias are united with the patch into a single PEC
#     entity. Via pitch is well below λ/10 at 915 MHz so the fence
#     behaves electromagnetically like a continuous short wall. This
#     is how real LoRa / HaLow PIFA products build their shorts.
#   - Feed: a vertical lumped-port plate through the substrate at
#     (patch_x0 + feed_offset, patch_y_centre). Represents a coax /
#     SMA feed whose centre pin comes up through a plated via with a
#     clearance antipad in the bottom ground. The lumped port models
#     the voltage source between the via (patch side) and the bottom
#     ground plane, which is the correct abstraction of the physical
#     SMA feed; the via + antipad need to appear on the Gerbers but do
#     not alter the EM result.

import numpy as np
import emerge as em


def build_pifa_antenna(
    L_mm=100.0,
    W_mm=80.0,
    patch_length_mm=50.0,       # X extent of patch (toward λ/4 in FR-4)
    patch_width_mm=25.0,        # Y extent of patch
    patch_margin_x_mm=5.0,      # X offset of patch's closed end from left edge
    patch_margin_y_mm=5.0,      # Y clearance from top edge of board to patch
    short_span_mm=12.0,         # Y extent of the via fence (<= patch_width)
    n_short_vias=6,             # plated vias along the short fence
    via_pad_mm=0.6,             # square approximation of via copper annulus
    feed_offset_mm=6.0,         # X distance from short fence to feed plate
    feed_trace_w_mm=1.5,
    loglevel="INFO",
):
    """
    Build a printed shorted-patch PIFA on FR-4.

    Returns:
        model, dict of geometry objects + metadata
    """
    mm = 1e-3

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

    model = em.Simulation("PIFA_915", loglevel=loglevel)

    # --------------------------------------------------------
    # Patch placement: along the top edge of the board
    # --------------------------------------------------------
    patch_x0 = patch_margin_x
    patch_x1 = patch_x0 + patch_length
    patch_y1 = W - patch_margin_y
    patch_y0 = patch_y1 - patch_width
    patch_y_center = 0.5 * (patch_y0 + patch_y1)

    # Validity guards
    if patch_x1 > L - patch_margin_x:
        raise ValueError("Patch extends past right edge of board.")
    if patch_y0 < 0:
        raise ValueError("Patch extends past bottom edge of board.")
    if short_span > patch_width:
        raise ValueError("short_span cannot exceed patch_width.")
    if feed_offset <= 0 or feed_offset >= patch_length:
        raise ValueError("feed_offset must be within (0, patch_length).")
    if feed_trace_w >= short_span:
        raise ValueError("feed_trace_w must be smaller than short_span for a sane layout.")
    if n_short_vias < 1:
        raise ValueError("n_short_vias must be >= 1.")
    if via_pad > short_span:
        raise ValueError("via_pad cannot exceed short_span.")
    if via_pad >= patch_length:
        raise ValueError("via_pad must be smaller than patch_length.")

    short_y0 = patch_y_center - short_span / 2.0

    # --------------------------------------------------------
    # Substrate (FR-4)
    # --------------------------------------------------------
    substrate = em.geo.Box(L, W, pcb_thickness, position=(0, 0, -pcb_thickness))
    substrate.set_material(em.Material(er=4.4, tand=0.02, color="#2d8c2d", opacity=0.6))

    # --------------------------------------------------------
    # Bottom ground plane (full board, antenna's only ground)
    # --------------------------------------------------------
    bottom_gnd = em.geo.XYPlate(L, W, position=(0, 0, -pcb_thickness))
    bottom_gnd.set_material(em.lib.PEC)

    # --------------------------------------------------------
    # Top copper: single rectangular patch + plated-through
    # via fence at the patch's closed edge. Each via is a
    # square-cross-section Box (approximating a circular plated
    # hole) that extends from just below the bottom ground
    # plane (z = -pcb_thickness - copper_thick) up through the
    # top of the patch (z = copper_thick), so it makes
    # face-to-face contact with both the bottom-ground XYPlate
    # and the patch Box. Every via is united with the patch so
    # GMSH welds the whole short fence into one PEC entity.
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
    vias = []
    patch = patch_box
    for y_c in via_centres_y:
        via = em.geo.Box(
            via_pad, via_pad, via_height,
            position=(patch_x0, float(y_c) - via_pad / 2.0,
                      -pcb_thickness - copper_thick),
        )
        vias.append(via)
        patch = em.geo.unite(patch, via)

    patch.set_material(em.lib.MET_COPPER)

    # --------------------------------------------------------
    # Feed port plate: second vertical plate through the
    # substrate at x = patch_x0 + feed_offset. Driven as the
    # lumped port.
    # --------------------------------------------------------
    feed_x = patch_x0 + feed_offset
    port_face = em.geo.Plate(
        np.array([feed_x, patch_y_center - feed_trace_w / 2.0, -pcb_thickness]),
        np.array([0.0, feed_trace_w, 0.0]),
        np.array([0.0, 0.0, pcb_thickness])
    )

    return model, {
        "substrate": substrate,
        "bottom_gnd": bottom_gnd,
        "patch": patch,
        "vias": vias,
        "port_face": port_face,
        "meta": {
            "L": L, "W": W,
            "patch_length": patch_length,
            "patch_width": patch_width,
            "patch_x0": patch_x0, "patch_y0": patch_y0,
            "patch_y_center": patch_y_center,
            "short_span": short_span,
            "n_short_vias": n_short_vias,
            "via_pad": via_pad,
            "feed_offset": feed_offset,
            "feed_trace_w": feed_trace_w,
            "feed_w": feed_trace_w,  # alias used by simulation.py
            "pcb_thickness": pcb_thickness,
            "copper_thick": copper_thick,
        },
    }
