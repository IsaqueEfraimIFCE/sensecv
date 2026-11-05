"""Generate one Kaggle kernel folder per variant from the shared trainer."""
import json
import shutil
from pathlib import Path

BASE = Path(r"C:\Users\Isaque\Desktop\sense_run4_experiments")
FONTE = BASE / "kernel_var" / "train_variant.py"
VARIANTES = {
    "fps5": ("sensecv-mnv4-fps5", "SenseCV MobileNetV4 5 FPS"),
    "dino098": ("sensecv-mnv4-dino098", "SenseCV MobileNetV4 DINOv2 0.98"),
    "dia_validacao": ("sensecv-mnv4-diaval", "SenseCV MobileNetV4 held-out day validation"),
}
codigo = FONTE.read_text(encoding="utf-8")
assert 'VARIANTE = "fps5"  # PACKAGER REPLACES THIS LINE' in codigo

for variante, (slug, titulo) in VARIANTES.items():
    destino = BASE / "kernels" / slug
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "train_variant.py").write_text(
        codigo.replace('VARIANTE = "fps5"  # PACKAGER REPLACES THIS LINE',
                       f'VARIANTE = "{variante}"'), encoding="utf-8")
    (destino / "kernel-metadata.json").write_text(json.dumps({
        "id": f"anamariabarrosnonato/{slug}",
        "title": titulo,
        "code_file": "train_variant.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["anamariabarrosnonato/sensecv-run4-sense"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2), encoding="utf-8")
    print("prepared", destino)
