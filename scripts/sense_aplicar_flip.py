"""Gera treino_efetivo/ (66.236 linhas) a partir de treino/ (33.118 linhas).

Reproduz augment() de scripts/mobilenetv1_two_head_all_frames.py: cada linha de
treino e duplicada com a imagem espelhada horizontalmente e os rotulos de desvio
left <-> right trocados. O rotulo de obstaculo nao muda. validacao e teste nunca
sao aumentados.

    python aplicar_flip.py            # ao lado de treino/

Resultado esperado (= train_effective do run4 MobileNetV1):
    66.236 linhas | free_path 32.102 / obstacle 34.134 | left 14.054 / right 14.054 / none 38.128
"""
import csv
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parent
ORIGEM = RAIZ / "treino"
DESTINO = RAIZ / "treino_efetivo"
IMAGENS = "images_x_y_224"
TROCA = {"0": "1", "1": "0", "2": "2"}  # left <-> right, none igual
QUALIDADE_JPEG = 95


def main():
    linhas = [l.split() for l in (ORIGEM / "labels.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    proveniencia = list(csv.DictReader((ORIGEM / "proveniencia.csv").open(encoding="utf-8", newline="")))
    destino_imgs = DESTINO / IMAGENS
    destino_imgs.mkdir(parents=True, exist_ok=True)

    saida, prov_saida = [], []
    for n, (nome, obstaculo, desvio) in enumerate(linhas, start=1):
        shutil.copyfile(ORIGEM / IMAGENS / nome, destino_imgs / f"{n}.jpg")
        saida.append(f"{n}.jpg {obstaculo} {desvio}")
        registro = dict(proveniencia[n - 1], arquivo=f"{n}.jpg", flip=0)
        prov_saida.append(registro)

    deslocamento = len(linhas)
    for n, (nome, obstaculo, desvio) in enumerate(linhas, start=1):
        indice = deslocamento + n
        with Image.open(ORIGEM / IMAGENS / nome) as im:
            espelhada = im.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            espelhada.save(destino_imgs / f"{indice}.jpg", "JPEG", quality=QUALIDADE_JPEG)
        saida.append(f"{indice}.jpg {obstaculo} {TROCA[desvio]}")
        registro = dict(proveniencia[n - 1], arquivo=f"{indice}.jpg", flip=1)
        registro["deviation_label"] = {"left": "right", "right": "left", "none": "none"}[registro["deviation_label"]]
        prov_saida.append(registro)

    (DESTINO / "labels.txt").write_text("\n".join(saida) + "\n", encoding="utf-8")
    with (DESTINO / "proveniencia.csv").open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(prov_saida[0].keys()))
        escritor.writeheader()
        escritor.writerows(prov_saida)

    obstaculos = Counter({"0": 0, "1": 0})
    desvios = Counter({"0": 0, "1": 0, "2": 0})
    for linha in saida:
        _, o, d = linha.split()
        obstaculos[o] += 1
        desvios[d] += 1
    print(f"{DESTINO}: {len(saida)} linhas")
    print(f"  free_path={obstaculos['0']} obstacle={obstaculos['1']}")
    print(f"  left={desvios['0']} right={desvios['1']} none={desvios['2']}")
    esperado = (len(saida) == 66236 and obstaculos["0"] == 32102 and obstaculos["1"] == 34134
                and desvios["0"] == 14054 and desvios["1"] == 14054 and desvios["2"] == 38128)
    print("confere com train_effective do run4" if esperado else "ATENCAO: nao confere com train_effective")


if __name__ == "__main__":
    main()
