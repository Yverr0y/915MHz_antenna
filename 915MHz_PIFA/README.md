# PIFA 915 MHz Test Board

Shorted-patch PIFA for the AU915 ISM band (915-928 MHz) on a 2-layer 1.6 mm FR-4 PCB, simulated in EMerge.

![PCB 3D View](report/pcb_3d.png)

## Design

- **Patch**: 34.42 x 26.00 mm on F.Cu, shorted to GND via 6 plated-through vias
- **Feed**: Via offset 7.00 mm from the shorted edge
- **Board**: 100 x 80 mm, full B.Cu ground plane
- **Matching**: Pi network (L1, C1, C2) between SMA and antenna feed
- **Connector**: SMA edge-mount (Amphenol 132289)

## Simulation Results

Verified 2026-04-12 with EMerge FEM, 700-1300 MHz sweep.

| Metric | Value |
|--------|-------|
| Resonance | 925.0 MHz (-33.23 dB) |
| -10 dB BW | 920 - 930 MHz (10 MHz) |
| S11 at 915 MHz | -7.09 dB |
| S11 at 926 MHz | -44.96 dB |

![S11](report/s11_plot.png)

![VSWR](report/vswr_plot.png)

![Smith Chart](report/smith_chart.png)

## Schematic

![Schematic](report/schematic.png)

## Reproducing

```bash
cd 915MHz_PIFA
python verify.py
```
