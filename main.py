# main.py - View geometry only (no simulation)
from geometry import build_meander_antenna

if __name__ == "__main__":
    model, geo = build_meander_antenna(N=4, L_mm=50, W_mm=60)
    model.commit_geometry()
    model.mw.set_frequency(915e6)
    model.generate_mesh()
    model.view()
