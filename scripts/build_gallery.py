"""Build assets/gallery.svg: a self-contained, auto-rotating image gallery.

GitHub renders README images through <img>, which blocks scripts and external
resources, so the slides are embedded as base64 and cross-faded with CSS.
Source images live in assets/gallery/ (pre-resized JPEGs).
"""
import base64
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIDES = [  # (file, caption, optional second caption line)
    ("chemical-reviews-cover.jpg", "Chemical Reviews cover",
     "Made for Benjamin Elling (lead author). I helped, but was not the main author."),
    ("captisol.jpg", "Captisol", None),
    ("dielsAlder.jpg", "Diels-Alder reaction", None),
    ("hemoglobin.jpg", "Hemoglobin", None),
    ("mpro.jpg", "Mpro", None),
    ("myoglobin.jpg", "Myoglobin", None),
    ("OCT.jpg", "Octreotide", None),
    ("solvation.jpg", "Solvation", None),
]
W, H = 820, 520
IMG_BOX = (740, 410)
HOLD, FADE = 4.0, 1.0  # seconds


def jpeg_size(data):
    i = 2
    while i < len(data):
        marker = data[i + 1]
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + length
    raise ValueError("no SOF marker")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def build():
    n = len(SLIDES)
    total = n * HOLD
    p = lambda t: f"{t / total * 100:.3f}%"
    css = (
        f".s{{opacity:0;animation:cycle {total}s infinite}}"
        f"@keyframes cycle{{0%{{opacity:0}}{p(FADE)}{{opacity:1}}{p(HOLD)}{{opacity:1}}"
        f"{p(HOLD + FADE)}{{opacity:0}}100%{{opacity:0}}}}"
        "text{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;fill:#e6edf3}"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="Rotating gallery of molecular graphics">',
        f"<style>{css}</style>",
        f'<rect width="{W}" height="{H}" rx="14" fill="#0d1117"/>',
    ]
    for i, (fname, cap, sub) in enumerate(SLIDES):
        data = (ROOT / "assets/gallery" / fname).read_bytes()
        iw, ih = jpeg_size(data)
        scale = min(IMG_BOX[0] / iw, IMG_BOX[1] / ih)
        w, h = iw * scale, ih * scale
        x, y = (W - w) / 2, 16 + (IMG_BOX[1] - h) / 2
        b64 = base64.b64encode(data).decode()
        out.append(f'<g class="s" style="animation-delay:{i * HOLD}s">')
        out.append(f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                   f'href="data:image/jpeg;base64,{b64}"/>')
        out.append(f'<text x="{W / 2}" y="{H - (50 if sub else 30)}" text-anchor="middle" '
                   f'font-size="17" font-weight="600">{esc(cap)}</text>')
        if sub:
            out.append(f'<text x="{W / 2}" y="{H - 24}" text-anchor="middle" font-size="14" '
                       f'style="fill:#9da7b3">{esc(sub)}</text>')
        out.append("</g>")
    out.append("</svg>")
    (ROOT / "assets/gallery.svg").write_text("\n".join(out))


if __name__ == "__main__":
    build()
