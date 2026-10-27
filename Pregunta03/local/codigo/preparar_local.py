# -*- coding: utf-8 -*-
"""preparar_local.py — Ejecuta en orden los 4 pasos de preparación local (Pregunta 3, CCT-20).

    paso1_particiones.py   -> JSON + particiones cis/trans
    paso2_verificar.py     -> anti-fuga + Tabla 1
    paso3_exportar_yolo.py -> imágenes reducidas + etiquetas YOLO
    paso4_comprimir.py     -> yolo_data_cct20.zip

Ejemplo (Windows):
    python codigo/preparar_local.py --resize 640
    python codigo/preparar_local.py --pasos 3 4      # solo algunos pasos (usa lo guardado por los anteriores)
Cada paso también se puede ejecutar solo: python codigo/paso3_exportar_yolo.py
"""
import argparse, sys, time
import cct_comun as C
import paso1_particiones, paso2_verificar, paso3_exportar_yolo, paso4_comprimir

def main():
    ap = C.agregar_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--pasos", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4], help="pasos a ejecutar")
    cfg = C.resolver(ap.parse_args())
    C.mostrar_rutas(cfg)
    t0 = time.time(); data = ann = None
    if 1 in cfg.pasos:
        print("=== PASO 1/4: leer JSON, indexar imágenes y armar particiones ===")
        data, ann, _ = paso1_particiones.ejecutar(cfg)
    if 2 in cfg.pasos:
        print("\n=== PASO 2/4: verificaciones anti-fuga y estadísticas ===")
        paso2_verificar.ejecutar(cfg, data, ann)
    if 3 in cfg.pasos:
        print("\n=== PASO 3/4: exportar a YOLO con imágenes reducidas ===")
        paso3_exportar_yolo.ejecutar(cfg, data, ann)
    if 4 in cfg.pasos:
        print("\n=== PASO 4/4: comprimir ===")
        paso4_comprimir.ejecutar(cfg)
    print(f"\nListo en {(time.time()-t0)/60:.1f} min. Resultados en {cfg.work}")

if __name__ == "__main__":
    sys.exit(main())
