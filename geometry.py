# geometry.py
import emerge as em

def build_meander_antenna(
    N=4,
    L_mm=50.0,
    W_mm=60.0,
    trace_w_mm=0.5,
    gnd_height_mm=40.0,
    feed_gap_mm=1.0,
    feed_penetration_mm=10.0,
    feed_clearance_mm=0.5,
    meander_offset_x_mm=2.0,
    meander_offset_y_mm=2.0,
    port_gap_mm=0.5,
    loglevel="INFO"
):
    """
    Builds geometry only.
    No ports, no air box, no frequency.

    The feed line now has a small gap (port_gap) cut into it where
    the lumped port surface will be placed. This ensures the port
    face is a real meshed surface in the geometry.

    Returns:
        model
        dict of geometry objects and metadata
    """

    mm  = 1e-3
    join = 0.05 * mm

    # Convert to meters
    L = L_mm * mm
    W = W_mm * mm

    trace_w = trace_w_mm * mm
    gnd_height = gnd_height_mm * mm
    feed_gap = feed_gap_mm * mm
    feed_penetration = feed_penetration_mm * mm
    feed_clearance = feed_clearance_mm * mm
    offset_x = meander_offset_x_mm * mm
    offset_y = meander_offset_y_mm * mm
    port_gap = port_gap_mm * mm

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------
    model = em.Simulation("MeanderAntenna", loglevel=loglevel)

    pcb_thickness = 1.6 * mm
    copper_thick  = 0.035 * mm

    # --------------------------------------------------------
    # Substrate
    # --------------------------------------------------------
    substrate = em.geo.Box(L, W, pcb_thickness, position=(0, 0, -pcb_thickness))
    substrate.set_material(em.Material(er=4.4, tand=0.02, color="#2d8c2d", opacity=0.6))

    # --------------------------------------------------------
    # Meander region (Y)
    # --------------------------------------------------------
    meander_y_bottom = gnd_height + feed_gap
    meander_y_top    = W - offset_y
    meander_avail_h  = meander_y_top - meander_y_bottom
    meander_arm_length = meander_avail_h - trace_w
    if meander_arm_length <= 0:
        raise ValueError("Meander height too small for given trace width and offsets.")

    # --------------------------------------------------------
    # Meander region (X)
    # --------------------------------------------------------
    n_legs = 2 * N
    if n_legs < 2:
        raise ValueError("N must be >= 1.")

    meander_x_start = max(offset_x, feed_clearance)
    meander_x_end   = L - offset_x
    meander_avail_w = meander_x_end - meander_x_start
    if meander_avail_w <= 0:
        raise ValueError("Meander X available width is <= 0.")

    # Compute trace_gap so the meander fills available width
    trace_gap = (meander_avail_w - n_legs * trace_w) / (n_legs - 1)
    if trace_gap <= 0:
        raise ValueError(
            f"Computed trace_gap is {trace_gap/mm:.3f} mm (<=0). "
            "Reduce N, increase L, or reduce offsets/trace width."
        )
    leg_pitch = trace_w + trace_gap

    # --------------------------------------------------------
    # Feed position
    # --------------------------------------------------------
    feed_x_centre = meander_x_start + trace_w / 2
    feed_y_bottom = gnd_height - feed_penetration
    feed_y_top    = meander_y_bottom
    feed_len      = feed_y_top - feed_y_bottom
    if feed_len <= 0:
        raise ValueError("Feed length <= 0.")

    # Port gap location: centred at the top edge of the ground plane
    # The port surface will sit in a small gap in the feed line at y = gnd_height
    port_y_centre = gnd_height
    port_y_bottom = port_y_centre - port_gap / 2.0
    port_y_top    = port_y_centre + port_gap / 2.0

    # --------------------------------------------------------
    # Ground plane + slot
    # --------------------------------------------------------
    gnd_full = em.geo.Box(L, gnd_height, copper_thick, position=(0, 0, 0))

    slot_left  = feed_x_centre - trace_w / 2 - feed_clearance
    slot_width = trace_w + 2 * feed_clearance
    slot_y_bot = feed_y_bottom - feed_clearance
    slot_h     = gnd_height - slot_y_bot
    if slot_left < 0:
        raise ValueError("Feed slot extends beyond left PCB edge.")
    if slot_y_bot < 0:
        raise ValueError("Feed slot extends below y=0.")

    slot = em.geo.Box(
        slot_width, slot_h, copper_thick,
        position=(slot_left, slot_y_bot, 0)
    )

    gnd = em.geo.subtract(gnd_full, slot)
    gnd.set_material(em.lib.MET_COPPER)

    # --------------------------------------------------------
    # Antenna copper parts
    # --------------------------------------------------------
    antenna_parts = []

    # Feed line (single piece from inside GND up to meander)
    feed = em.geo.Box(
        trace_w, feed_len, copper_thick,
        position=(feed_x_centre - trace_w / 2, feed_y_bottom, 0)
    )
    antenna_parts.append(feed)

    # Port surface (a 2D face embedded in the geometry at the gap)
    # This is the actual surface the LumpedPort BC will be applied to.
    # It spans the trace width (x) and the port gap (y), at z = copper_thick/2
    #
    # Build as an XYPolygon rectangle, then place it in 3D using a
    # coordinate system at the centre of the port gap.
    # Port surface: a VERTICAL plate connecting the feed to the ground plane.
    # Based on the patch antenna example: the port spans from the ground plane
    # (z = 0, top of GND copper) down through the substrate (z = -pcb_thickness).
    # It is located at the bottom end of the feed, at the GND top edge.
    #
    # The port is a Plate defined by:
    #   origin: bottom-left corner of the port rectangle
    #   u: width vector (along X, spanning trace_w)
    #   v: height vector (along Z, spanning the substrate thickness)
    import numpy as np
    port_face = em.geo.Plate(
        np.array([feed_x_centre - trace_w / 2.0, feed_y_bottom, -pcb_thickness]),  # origin (bottom corner)
        np.array([trace_w, 0, 0]),                                                  # u: width along X
        np.array([0, 0, pcb_thickness])                                             # v: height along Z
    )

    # Legs
    x0 = meander_x_start
    ov = min(join, 0.25 * trace_w)

    for k in range(n_legs):
        xk = x0 + k * leg_pitch
        leg = em.geo.Box(
            trace_w,
            meander_arm_length + ov,
            copper_thick,
            position=(xk, meander_y_bottom, 0)
        )
        antenna_parts.append(leg)

    # Connectors
    for k in range(n_legs - 1):
        x_left = x0 + k * leg_pitch
        conn_w = 2 * trace_w + trace_gap

        if (k % 2) == 0:
            # TOP connector
            top_conn = em.geo.Box(
                conn_w,
                trace_w + ov,
                copper_thick,
                position=(x_left, meander_y_top - trace_w - ov, 0)
            )
            antenna_parts.append(top_conn)
        else:
            # BOTTOM connector
            bot_conn = em.geo.Box(
                conn_w,
                trace_w + ov,
                copper_thick,
                position=(x_left, meander_y_bottom, 0)
            )
            antenna_parts.append(bot_conn)

    antenna = em.geo.unite(*antenna_parts)
    antenna.set_material(em.lib.MET_COPPER)

    return model, {
        "substrate": substrate,
        "ground": gnd,
        "antenna": antenna,
        "port_face": port_face,
        "meta": {
            "L": L, "W": W,
            "gnd_height": gnd_height,
            "meander_y_bottom": meander_y_bottom,
            "trace_w": trace_w,
            "trace_gap": trace_gap,
            "copper_thick": copper_thick,
            "feed_clearance": feed_clearance,
            "feed_x_centre": feed_x_centre,
            "feed_y_bottom": feed_y_bottom,
            "feed_y_top": feed_y_top,
            "port_gap": port_gap,
            "port_y_centre": port_y_centre,
        }
    }
