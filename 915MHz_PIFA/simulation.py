# simulation.py
# Simulation helpers for the PIFA 915 MHz model.

import emerge as em

C0 = 299_792_458.0


def add_airbox_and_open_boundary(
    model,
    geo,
    f0=915e6,
    margin_lambda=0.25,
):
    """
    Build an air box around the PIFA with a quarter-wave margin on all
    sides. Returns the air box so the caller can attach an absorbing BC
    to its outer faces.
    """
    meta = geo["meta"]
    L = meta["L"]
    W = meta["W"]
    pcb_thickness = meta["pcb_thickness"]

    lam = C0 / f0
    m = margin_lambda * lam

    air = em.geo.Box(
        L + 2 * m,
        W + 2 * m,
        pcb_thickness + 2 * m,
        position=(-m, -m, -pcb_thickness - m),
    )
    return air


def apply_boundary_conditions(model, geo, air, order=2, abctype="B"):
    """
    Apply the absorbing boundary on the air-box exterior and the lumped
    port on the vertical feed plate. Call AFTER commit_geometry() and
    generate_mesh().
    """
    meta = geo["meta"]

    model.mw.bc.AbsorbingBoundary(
        air.boundary(),
        order=order,
        abctype=abctype,
    )

    model.mw.bc.LumpedPort(
        geo["port_face"],
        1,
        width=meta["feed_w"],
        height=meta["pcb_thickness"],
        direction=em.ZAX,
        Z0=50.0,
    )
