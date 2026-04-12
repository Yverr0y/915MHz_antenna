# main.py - View printed shorted-patch PIFA geometry only (no simulation)
from geometry import build_pifa_antenna

if __name__ == "__main__":
    model, geo = build_pifa_antenna(
        L_mm=100.0,
        W_mm=80.0,
        patch_length_mm=50.0,
        patch_width_mm=25.0,
        short_span_mm=12.0,
        feed_offset_mm=6.0,
    )
    model.commit_geometry()
    model.mw.set_frequency(915e6)
    model.generate_mesh()
    model.view()
