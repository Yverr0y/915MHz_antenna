# simulation.py
# Simulation helpers for EMerge microwave FEM.

import emerge as em

C0 = 299_792_458.0


def add_airbox_and_open_boundary(
    model,
    geo,
    f0=915e6,
    margin_lambda=0.25,
    order=2,
    abctype="B",
):
    """
    Adds an air box around the model and applies an absorbing
    boundary on its outer faces.
    """
    meta = geo["meta"]
    L = meta["L"]
    W = meta["W"]
    pcb_thick = 1.6e-3  # Match geometry.py

    lam = C0 / f0
    m = margin_lambda * lam

    # Air box encloses the full PCB + margin on all sides
    air = em.geo.Box(
        L + 2 * m,
        W + 2 * m,
        pcb_thick + 2 * m,
        position=(-m, -m, -pcb_thick - m),
    )

    return air


def apply_boundary_conditions(model, geo, air, f0=915e6, order=2, abctype="B"):
    """
    Apply absorbing BC on airbox exterior and lumped port on the feed.
    Call this AFTER commit_geometry() and generate_mesh().
    """
    meta = geo["meta"]

    # Absorbing boundary on the outside of the airbox
    model.mw.bc.AbsorbingBoundary(
        air.boundary(),
        order=order,
        abctype=abctype
    )

    # Lumped port on the embedded port face
    port_face = geo["port_face"]
    trace_w = meta["trace_w"]
    pcb_thickness = 1.6e-3

    model.mw.bc.LumpedPort(
        port_face,
        1,
        width=trace_w,
        height=pcb_thickness,
        direction=em.ZAX,
        Z0=50.0
    )
