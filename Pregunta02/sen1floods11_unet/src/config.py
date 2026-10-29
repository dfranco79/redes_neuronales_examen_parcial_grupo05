"""Hiperparámetros y opciones del experimento (los valores por defecto son los que se usaron en el informe)."""
import argparse
from dataclasses import dataclass, asdict, fields


@dataclass
class Config:
    data_dir: str = "data/sen1floods11"      # carpeta del dataset (misma estructura que el bucket v1.1)
    out_dir: str = "outputs"                 # checkpoints, métricas, figuras e informe
    seed: int = 42
    crop: int = 256               # recorte aleatorio para entrenar, a resolución nativa (10 m)
    batch_size: int = 8
    eval_batch: int = 4           # en evaluación se usa el chip completo de 512x512
    epochs: int = 40              # épocas de los modelos con chips dorados
    weak_epochs: int = 10         # épocas de los modelos dorados + débiles (~4.4k chips por época)
    weak_pretrain_epochs: int = 8 # épocas de pre-entrenamiento solo con débiles
    lr: float = 1e-3
    ft_lr: float = 3e-4
    weight_decay: float = 1e-4
    pos_weight: float = 3.0
    base_ch: int = 32
    depth: int = 4
    max_weak: int = 0             # 0 = todos los chips débiles fuera de Bolivia; >0 = submuestreo
    use_s2_weak: bool = True      # usar también las etiquetas débiles de Sentinel-2 (S2IndexLabelWeak)
    loader_workers: int = 2       # procesos que leen los chips débiles del disco (Colab tiene 2 CPU)
    cloud_max: float = 0.05       # fracción máxima de nubes para considerar un chip S2 como despejado
    run_loss_ablation: bool = True
    run_weak_ablation: bool = True
    run_s2_bonus: bool = True
    download: bool = True
    force_retrain: bool = False
    amp: bool = True              # precisión mixta en GPU
    workers: int = 16             # hilos de descarga
    quick: bool = False           # prueba rápida: pocas épocas y modelo pequeño

    def finalize(self):
        if self.quick:
            self.epochs, self.weak_epochs, self.weak_pretrain_epochs = 2, 1, 1
            self.base_ch, self.crop = 8, 128
        return self

    def as_dict(self):
        return asdict(self)


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Sen1Floods11 - U-Net desde cero (Pregunta 2)")
    for f in fields(Config):
        if f.type in (bool, "bool"):
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes", "si", "sí"),
                           default=f.default, metavar="{true,false}")
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type) if isinstance(f.type, str) else f.type
            p.add_argument(f"--{f.name}", type=typ, default=f.default)
    return Config(**vars(p.parse_args(argv))).finalize()
