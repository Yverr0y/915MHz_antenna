# geometry.py
# Printed Planar Inverted-F Antenna (PIFA) for 915 MHz on FR-4.
#
# Layout (top view, X = board length, Y = board width):
#
#     +-----------------------------------------+  y = W
#     |                                         |
#     |  [=====================================]|  <- radiating bar
#     |  |     |                                |
#     |  |short| feed                           |  <- stubs
#     +--O=====O================================+  y = gnd_height
#     |       (slot)                            |
#     |     top ground plane                    |
#     |                                         |
#     +-----------------------------------------+  y = 0
#
# Copper:
#   - Bottom copper: full-board ground plane (PEC plate)
#   - Top copper:    partial ground plane + F-shaped antenna trace,
#                    united into a single copper piece so the short
#                    stub is electrically continuous with the ground.
#                    A clearance slot around the feed stub isolates it
#                    from the top ground plane.
#
# Feed: vertical lumped-port plate inside the substrate, from the
# bottom ground up to the feed-stub tip (same pattern as the meander).

import numpy as np
import emerge as em


def build_pifa_antenna(
    L_mm=80.0,
    W_mm=50.0,
    gnd_height_mm=25.0,
    bar_length_mm=40.0,        # X extent of the radiating bar (~λ_g/4)
    bar_width_mm=4.0,          # Y extent of the radiating bar
    bar_margin_x_mm=3.0,       # X offset of the bar's closed end from left edge
    stub_gap_mm=3.0,           # Y gap between top ground plane and radiating bar
    feed_offset_mm=4.0,        # X distance from short stub centre to feed stub centre
    feed_penetration_mm=5.0,   # how far the feed stub extends into the ground slot
    trace_w_mm=1.0,
    feed_clearance_mm=0.5,     # clearance around feed stub inside the ground slot
    n_meander_legs=1,          # 1 = straight bar; 2 = U-fold meander; N = N-leg serpentine
    meander_gap_mm=2.0,        # Y gap between adjacent meander legs
    loglevel="INFO",
):
    """
    Build a printed PIFA geometry on FR-4.

    Returns:
        model
        dict of geometry objects and metadata
    """
    mm = 1e-3
    join = 0.05 * mm

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

    model = em.Simulation("PIFA_915", loglevel=loglevel)

    # --------------------------------------------------------
    # Derived positions
    # --------------------------------------------------------
    bar_x0 = bar_margin_x
    bar_y0 = gnd_height + stub_gap       # bottom edge of radiating bar
    bar_x1 = bar_x0 + bar_length
    # Total Y extent of the (possibly meandered) bar
    bar_y1 = bar_y0 + n_meander_legs * bar_width + (n_meander_legs - 1) * meander_gap
    if bar_x1 > L - bar_margin_x:
        raise ValueError("Radiating bar extends past the right edge of the board.")
    if bar_y1 > W:
        raise ValueError("Radiating bar extends past the top edge of the board.")

    # Short stub flush with closed end of bar; feed stub offset to the right.
    short_x_centre = bar_x0 + trace_w / 2.0
    feed_x_centre = short_x_centre + feed_offset
    if feed_x_centre + trace_w / 2.0 > bar_x1:
        raise ValueError("Feed offset pushes the feed stub past the open end of the bar.")

    feed_y_bottom = gnd_height - feed_penetration
    if feed_y_bottom < 0:
        raise ValueError("Feed penetration extends below the ground plane.")

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
    # Top ground plane with feed clearance slot
    # --------------------------------------------------------
    gnd_full = em.geo.Box(L, gnd_height, copper_thick, position=(0, 0, 0))

    slot_left = feed_x_centre - trace_w / 2.0 - feed_clearance
    slot_y_bot = feed_y_bottom - feed_clearance
    slot_width = trace_w + 2 * feed_clearance
    slot_h = gnd_height - slot_y_bot
    if slot_left < 0 or slot_y_bot < 0:
        raise ValueError("Feed slot extends beyond the ground plane boundary.")

    slot = em.geo.Box(
        slot_width, slot_h, copper_thick,
        position=(slot_left, slot_y_bot, 0)
    )
    gnd = em.geo.subtract(gnd_full, slot)

    # --------------------------------------------------------
    # F-shape antenna trace: radiating bar + short stub + feed stub
    # All on the top copper layer. Small overlaps (`ov`) at the
    # junctions give boolean unite a clean weld.
    # --------------------------------------------------------
    ov = min(join, 0.25 * trace_w)

    # Meandered radiating element: N horizontal legs stacked in Y,
    # connected at alternating X ends by short Y-direction jumpers.
    # Leg 0 is the bottom leg (closest to the ground plane); the short
    # and feed stubs attach to leg 0. With n_meander_legs=1 this reduces
    # to the original straight bar.
    meander_parts = []
    for i in range(n_meander_legs):
        leg_y = bar_y0 + i * (bar_width + meander_gap)
        leg = em.geo.Box(
            bar_length, bar_width, copper_thick,
            position=(bar_x0, leg_y, 0)
        )
        meander_parts.append(leg)

    for i in range(n_meander_legs - 1):
        # Alternate the end the connector sits on: even i -> right end,
        # odd i -> left end, producing a serpentine path.
        if i % 2 == 0:
            conn_x = bar_x1 - trace_w
        else:
            conn_x = bar_x0
        conn_y_bot = bar_y0 + i * (bar_width + meander_gap) + bar_width - ov
        conn = em.geo.Box(
            trace_w, meander_gap + 2 * ov, copper_thick,
            position=(conn_x, conn_y_bot, 0)
        )
        meander_parts.append(conn)

    bar = em.geo.unite(*meander_parts) if len(meander_parts) > 1 else meander_parts[0]

    # Short stub: overlaps the ground plane at the bottom (y < gnd_height)
    # and the radiating bar at the top (y > bar_y0). This electrically
    # ties the stub to both without leaving a touching-face gap.
    short_stub = em.geo.Box(
        trace_w, stub_gap + 2 * ov, copper_thick,
        position=(short_x_centre - trace_w / 2.0, gnd_height - ov, 0)
    )

    # Feed stub: from feed_y_bottom (inside the slot) up into the bar
    feed_stub_len = (bar_y0 - feed_y_bottom) + ov
    feed_stub = em.geo.Box(
        trace_w, feed_stub_len, copper_thick,
        position=(feed_x_centre - trace_w / 2.0, feed_y_bottom, 0)
    )

    # Unite the ground plane (with slot) and all three antenna parts into
    # one top-copper piece. The short stub welds into the ground plane,
    # giving the characteristic PIFA DC short.
    top_copper = em.geo.unite(gnd, bar, short_stub, feed_stub)
    top_copper.set_material(em.lib.MET_COPPER)

    # --------------------------------------------------------
    # Port face: vertical plate at the feed stub bottom, spanning the
    # substrate thickness from the bottom ground up to the top copper.
    # --------------------------------------------------------
    port_face = em.geo.Plate(
        np.array([feed_x_centre - trace_w / 2.0, feed_y_bottom, -pcb_thickness]),
        np.array([trace_w, 0, 0]),
        np.array([0, 0, pcb_thickness])
    )

    return model, {
        "substrate": substrate,
        "bottom_gnd": bottom_gnd,
        "top_copper": top_copper,
        "port_face": port_face,
        "meta": {
            "L": L, "W": W,
            "gnd_height": gnd_height,
            "bar_length": bar_length,
            "bar_width": bar_width,
            "n_meander_legs": n_meander_legs,
            "meander_gap": meander_gap,
            "bar_x0": bar_x0, "bar_y0": bar_y0, "bar_y1": bar_y1,
            "trace_w": trace_w,
            "feed_w": trace_w,  # alias used by simulation.py
            "short_x_centre": short_x_centre,
            "feed_x_centre": feed_x_centre,
            "feed_y_bottom": feed_y_bottom,
            "stub_gap": stub_gap,
            "feed_offset": feed_offset,
            "pcb_thickness": pcb_thickness,
            "copper_thick": copper_thick,
        },
    }
