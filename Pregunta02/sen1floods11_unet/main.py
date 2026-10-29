"""Pregunta 2: mapeo de inundaciones con Sentinel-1 (Sen1Floods11) y una U-Net propia.

Todo el proyecto se corre con:

    python main.py

Paso 0         Descarga los datos de gs://sen1floods11 (v1.1): los 446 chips dorados de los 11 eventos con su imagen
               Sentinel-2, los chips débiles que no son de Bolivia con etiqueta Otsu (S1) y por índice espectral (S2),
               y la partición oficial. Revisa el inventario del dataset y mide el ruido de las etiquetas débiles.
Entregable 1   U-Net propia (src/model.py) entrenada solo con chips dorados y pérdida BCE+Dice.
Entregable 2a  Solo dorados vs. dorados + débiles S1 vs. dorados + débiles S2 vs. pre-entrenamiento con débiles + ajuste.
Entregable 2b  Pérdida BCE vs. Dice vs. BCE+Dice.
Entregable 3   Evaluación en test (eventos vistos): IoU, F1, precisión, recall y matriz de confusión.
Entregable 4   Bolivia (evento no visto): brecha con IC bootstrap y datos para revisar las hipótesis.
Entregable 5   Tipos de error característicos del SAR, con ejemplos.
Entregable 6   (Bono) Fusión SAR + Sentinel-2 en chips sin nubes.
Al final       Escribe outputs/results/*.json|csv, outputs/figures, outputs/analisis.md, README.md y
               outputs/informe/informe_p2.tex (plantilla NeurIPS).

Si la ejecución se interrumpe, al volver a correrla se cargan los checkpoints que ya existen.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src import analysis as A
from src.config import parse_args
from src.data import FloodDataset, Paths, build_map, load_splits, read_label
from src.download import download_dataset
from src.losses import BCEDice, MaskedBCE, MaskedDice
from src.metrics import bootstrap_iou, collect, evaluate_chips, sweep_threshold
from src.model import UNet, count_params
from src.report import write_reports
from src.train import load_model, load_or_train, set_seed


def main(cfg):
    t_start = time.time()
    out = Path(cfg.out_dir)
    fig_dir, res_dir = out / "figures", out / "results"
    for d in (fig_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    print(f"Dispositivo: {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    if device.type == "cpu" and not cfg.quick:
        print("Aviso: sin GPU el experimento completo tarda muchas horas.")

    R = dict(config=cfg.as_dict(), device=str(device), models={}, groups={}, figures={})

    # --- Paso 0: datos ---
    if cfg.download:
        download_dataset(cfg, need_s2=True)
    P = Paths(cfg.data_dir)
    s1m, lblm, s2m = build_map(P.s1), build_map(P.lbl), build_map(P.s2)
    S = load_splits(P)
    allsp = [c for ids in S.values() for c in ids]
    assert all(c in s1m and c in lblm for c in allsp), "Faltan archivos de los splits oficiales"
    assert not set(S["bolivia"]) & set(S["train"] + S["val"] + S["test"]), "Bolivia aparece en otro split"
    assert not any(c.startswith("Bolivia") for c in S["train"] + S["val"]), "Hay chips de Bolivia en entrenamiento"
    weak_ids = P.weak_list.read_text().split() if P.weak_list.exists() else []
    w1, wl, wl2 = build_map(P.s1_weak), build_map(P.lbl_weak), build_map(P.s2_weak)
    weak_ids = [c for c in weak_ids if c in w1 and not c.startswith("Bolivia")]
    weak_s1 = [c for c in weak_ids if c in wl]
    weak_s2 = [c for c in weak_ids if c in wl2] if cfg.use_s2_weak else []
    R["data"] = {k: len(v) for k, v in S.items()} | dict(weak=len(weak_ids), weak_s1=len(weak_s1),
                                                        weak_s2=len(weak_s2), s2=len(s2m))
    R["bucket"] = json.loads(P.inventory.read_text()) if P.inventory.exists() else {}
    print("Datos:", R["data"])

    # inventario: lo que dice el enunciado vs. lo que hay en los datos
    inv_rows, ev_table = A.inventory(P, S, s1m, lblm, s2m, build_map(P.otsu_hand), w1, wl, wl2, weak_ids)
    meta0 = A.load_metadata(P.meta)
    ev_table["orbita"] = ev_table.evento.map(lambda e: meta0.get(e, {}).get("orbit", "?"))
    ev_table["fecha_s1"] = ev_table.evento.map(lambda e: meta0.get(e, {}).get("s1_date", "?"))
    R["inventario"], R["eventos"] = inv_rows, ev_table.to_dict("records")
    R["figures"]["eventos"] = A.fig_events(ev_table, fig_dir / "chips_por_evento.png")
    pd.DataFrame(inv_rows).to_csv(res_dir / "inventario_dataset.csv", index=False)
    ev_table.to_csv(res_dir / "chips_por_evento.csv", index=False)
    for r in inv_rows:
        print(f"  {r['Estado']} {r['Característica']}: {r['Encontrado']}")
    train_events = {c.rsplit('_', 1)[0] for c in S["train"]}
    R["verificacion"] = dict(
        bolivia_train=sum(c.startswith("Bolivia") for c in S["train"]),
        bolivia_val=sum(c.startswith("Bolivia") for c in S["val"]),
        bolivia_weak=sum(c.startswith("Bolivia") for c in weak_ids),
        bolivia_eval=len(S["bolivia"]),
        test_events_seen=all(c.rsplit('_', 1)[0] in train_events for c in S["test"]),
        seg_libs=A.forbidden_imports(Path(__file__).resolve().parent / "src"))

    def loader(ds, train=False, bs=None, workers=0):
        return DataLoader(ds, batch_size=bs or (cfg.batch_size if train else cfg.eval_batch), shuffle=train,
                          drop_last=train and len(ds) > cfg.batch_size, num_workers=workers,
                          persistent_workers=workers > 0, pin_memory=device.type == "cuda")
    train_ds = FloodDataset(S["train"], s1m, lblm, crop=cfg.crop, augment=True, name="train")
    val_ds, test_ds, bol_ds = (FloodDataset(S[k], s1m, lblm, name=k) for k in ("val", "test", "bolivia"))
    train_dl, val_dl, test_dl, bol_dl = loader(train_ds, True), loader(val_ds), loader(test_ds), loader(bol_ds)
    R["n_params"] = count_params(UNet(2, 1, cfg.base_ch, cfg.depth))

    # --- Paso 0: ruido de las etiquetas débiles ---
    noise = A.weak_label_noise(P, S["train"] + S["val"] + S["test"])
    if noise:
        R["weak_noise"] = dict(overall=noise["overall"], per_event=noise["per_event"].to_dict("records"))
        noise["per_event"].round(4).to_csv(res_dir / "ruido_etiquetas_debiles.csv", index=False)
        print("Ruido Otsu vs. dorado:", {k: round(v, 4) for k, v in noise["overall"].items()})

    histories, collected = {}, {}

    def report(name, model, group, tag, vdl=val_dl, tdl=test_dl, bdl=bol_dl):
        vchips = collect(model, vdl, device)
        th, sweep = sweep_threshold(vchips)
        rv, _ = evaluate_chips(vchips, th)
        tchips, bchips = collect(model, tdl, device), collect(model, bdl, device)
        rt, pt = evaluate_chips(tchips, th)
        rb, pb = evaluate_chips(bchips, th)
        R["models"][name] = dict(tag=tag, group=group, threshold=th, val=rv, test=rt, bolivia=rb,
                                 test_ci=bootstrap_iou(pt), bolivia_ci=bootstrap_iou(pb))
        R["groups"].setdefault(group, []).append(name)
        collected[name] = dict(sweep=sweep, test=tchips, bol=bchips, per_test=pt, per_bol=pb)
        print(f"  {name}: th={th:.2f} | test IoU={rt['iou']:.4f} (chips {rt['mean_iou_chips']:.4f}) "
              f"F1={rt['f1']:.4f} | Bolivia IoU={rb['iou']:.4f}")

    def keep_light(name):  # libera memoria de los modelos que no se analizan en detalle
        collected[name].pop("test", None); collected[name].pop("bol", None)

    # --- Entregable 1: modelo base ---
    print("\n=== Entregable 1: U-Net propia, modelo base (solo dorados, BCE+Dice) ===")
    m, histories["Solo dorados · BCE+Dice"] = load_or_train(cfg, "gold_bcedice", train_dl, val_dl,
                                                            BCEDice(cfg.pos_weight), device)
    report("Solo dorados · BCE+Dice", m, "base", "gold_bcedice")
    models = {"Solo dorados · BCE+Dice": "gold_bcedice"}
    del m

    # --- Entregable 2b: función de pérdida ---
    if cfg.run_loss_ablation:
        print("\n=== Entregable 2b: pérdida BCE vs. Dice (BCE+Dice es el modelo base) ===")
        for name, tag, crit in [("Solo dorados · BCE", "gold_bce", MaskedBCE(cfg.pos_weight)),
                                ("Solo dorados · Dice", "gold_dice", MaskedDice())]:
            m, histories[name] = load_or_train(cfg, tag, train_dl, val_dl, crit, device)
            report(name, m, "loss", tag); keep_light(name); models[name] = tag
            del m
        R["groups"]["loss"] = ["Solo dorados · BCE", "Solo dorados · Dice", "Solo dorados · BCE+Dice"]

    # --- Entregable 2a: etiquetas débiles ---
    if cfg.run_weak_ablation and weak_s1:
        print(f"\n=== Entregable 2a: dorados vs. dorados + débiles ({len(weak_s1):,} débiles S1"
              f"{f', {len(weak_s2):,} débiles S2' if weak_s2 else ''}) ===")
        W = cfg.loader_workers
        wfrac = lambda ids, m: float(np.mean([((l := read_label(m[c])) == 1).sum() / max((l != -1).sum(), 1) for c in ids]))  # noqa: E731
        R["data"]["weak_water_frac"] = wfrac(weak_s1, wl)
        weak_ds1 = FloodDataset(weak_s1, w1, wl, crop=cfg.crop, augment=True, preload=False)
        arms = [("Dorados + débiles S1 (Otsu)", "goldweak_s1", weak_ds1)]
        if weak_s2:
            R["data"]["weak_s2_water_frac"] = wfrac(weak_s2, wl2)
            arms.append(("Dorados + débiles S2 (índice)", "goldweak_s2",
                         FloodDataset(weak_s2, w1, wl2, crop=cfg.crop, augment=True, preload=False)))
        names = ["Solo dorados · BCE+Dice"]
        for name, tag, wds in arms:
            m, histories[name] = load_or_train(cfg, tag, loader(ConcatDataset([train_ds, wds]), True, workers=W),
                                               val_dl, BCEDice(cfg.pos_weight), device, epochs=cfg.weak_epochs)
            report(name, m, "weak", tag); keep_light(name); models[name] = tag; names.append(name)
            del m
        name = "Pre-entreno débiles S1 → ajuste dorados"
        mp, histories["Pre-entreno (solo débiles S1)"] = load_or_train(
            cfg, "weak_s1_pretrain", loader(weak_ds1, True, workers=W), val_dl, BCEDice(cfg.pos_weight), device,
            epochs=cfg.weak_pretrain_epochs)
        m, histories[name] = load_or_train(cfg, "weak_s1_pretrain_ft", train_dl, val_dl, BCEDice(cfg.pos_weight), device,
                                           lr=cfg.ft_lr, init_state=mp.state_dict(),
                                           extra_sig=dict(pretrain_chips=len(weak_ds1), pretrain_epochs=cfg.weak_pretrain_epochs))
        report(name, m, "weak", "weak_s1_pretrain_ft"); keep_light(name); models[name] = "weak_s1_pretrain_ft"
        names.append(name)
        del m, mp, arms, weak_ds1
        gc.collect()
        R["groups"]["weak"] = names

    # --- Entregable 3: modelo final (se elige con la validación) ---
    final = max(models, key=lambda n: R["models"][n]["val"]["iou"])
    R["final_model"] = final
    print(f"\n=== Entregables 3, 4 y 5: modelo final (mayor IoU en validación dorada) = {final} ===")
    fm = load_model(cfg, models[final], device)
    if "test" not in collected[final]:
        collected[final]["test"], collected[final]["bol"] = collect(fm, test_dl, device), collect(fm, bol_dl, device)
    C, th = collected[final], R["models"][final]["threshold"]

    # --- Entregable 4: Bolivia y composición de los conjuntos ---
    meta = meta0
    R["composition"] = [A.composition(test_ds, C["per_test"], "test"), A.composition(bol_ds, C["per_bol"], "bolivia")]
    ev = A.per_event_table(C["per_test"])
    ev["orbita"] = ev.event.map(lambda e: meta.get(e, {}).get("orbit", "?"))
    R["per_event_test"] = ev.to_dict("records")
    R["bolivia_meta"] = meta.get("Bolivia", {})
    R["per_chip_bolivia"] = C["per_bol"][["cid", "iou", "water_frac"]].to_dict("records")
    ev.round(4).to_csv(res_dir / "iou_por_evento_test.csv", index=False)

    # --- Entregable 5: errores del SAR ---
    tax, breakdown, totals = A.error_taxonomy(C["test"], th)
    cases = A.select_cases(tax)
    R["error_breakdown"], R["error_totals"] = breakdown, totals
    titles = {"A": "A) Comisión: superficie oscura lejos del agua (sombra de radar / superficie lisa)",
              "B": "B) Omisión: agua con retrodispersión alta (doble rebote / vegetación inundada)",
              "C": "C) Confusión en el borde agua–suelo húmedo",
              "D": "D) Omisión: agua con VV intermedio (rugosidad por viento / turbidez)"}
    by_id = {c["cid"]: c for c in C["test"]}
    R["cases"] = {}
    for k, cid in cases.items():
        R["cases"][k] = dict(title=titles[k], **A.case_stats(by_id[cid], th, s2m.get(cid), key=k))
        R["figures"][f"caso_{k}"] = A.fig_case(by_id[cid], th, titles[k], fig_dir / f"caso_{k}_{cid}.png", s2m.get(cid))
    tax.to_csv(res_dir / "taxonomia_errores_por_chip.csv", index=False)

    # --- figuras ---
    R["figures"]["curvas"] = A.fig_curves(histories, fig_dir / "curvas_entrenamiento.png")
    R["figures"]["umbral"] = A.fig_sweep(C["sweep"], th, fig_dir / "barrido_umbral.png")
    R["figures"]["confusion"] = A.fig_confusion(R["models"][final]["test"], th, "Matriz de confusión — test", fig_dir / "matriz_confusion_test.png")
    R["figures"]["confusion_bol"] = A.fig_confusion(R["models"][final]["bolivia"], th, "Matriz de confusión — Bolivia", fig_dir / "matriz_confusion_bolivia.png")
    R["figures"]["ejemplos_test"] = A.fig_examples(C["test"], th, fig_dir / "ejemplos_test.png", n=4, title="Test (eventos vistos)")
    R["figures"]["ejemplos_bolivia"] = A.fig_examples(C["bol"], th, fig_dir / "ejemplos_bolivia.png", n=3, title="Bolivia (evento no visto)")
    R["figures"]["taxonomia"] = A.fig_breakdown(breakdown, fig_dir / "taxonomia_errores.png")
    tab = lambda names: pd.DataFrame([dict(modelo=n, **{k: R["models"][n]["test"][k] for k in ("iou", "mean_iou_chips", "f1")}) for n in names])  # noqa: E731
    if "loss" in R["groups"]:
        R["figures"]["ablacion_perdida"] = A.fig_bars(tab(R["groups"]["loss"]), fig_dir / "ablacion_perdida.png", "Ablación 2b — pérdida (test)")
    if "weak" in R["groups"]:
        R["figures"]["ablacion_debiles"] = A.fig_bars(tab(R["groups"]["weak"]), fig_dir / "ablacion_debiles.png", "Ablación 2a — etiquetas débiles (test)")
    R["figures"]["brecha"] = A.fig_gap([dict(modelo=n, test=v["test"]["iou"], bolivia=v["bolivia"]["iou"]) for n, v in R["models"].items()],
                                       fig_dir / "brecha_bolivia.png")
    del fm
    collected.clear()
    gc.collect()

    # --- Entregable 6 (bono): SAR + Sentinel-2 ---
    if cfg.run_s2_bonus and s2m:
        print("\n=== Entregable 6 (bono): SAR + Sentinel-2 en chips sin nubes ===")
        cf = {c: A.cloud_fraction(s2m[c]) for c in allsp if c in s2m}
        clear = {k: [c for c in v if c in cf and cf[c] < cfg.cloud_max] for k, v in S.items()}
        R["s2"] = dict(cloud_max=cfg.cloud_max, clear={k: len(v) for k, v in clear.items()},
                       total={k: len(v) for k, v in S.items()})
        print("  chips sin nubes:", R["s2"]["clear"])
        if min(len(clear["train"]), len(clear["val"]), len(clear["test"])) > 0:
            for name, tag, s2map, ch in [("Sin nubes · SAR (2 canales)", "clear_sar", None, 2),
                                         ("Sin nubes · SAR+S2 (15 canales)", "clear_sar_s2", s2m, 15)]:
                dl = {k: loader(FloodDataset(clear[k] or S[k][:1], s1m, lblm, s2m=s2map, crop=cfg.crop if k == "train" else None,
                                             augment=k == "train", name=f"{tag}-{k}"), k == "train")
                      for k in ("train", "val", "test", "bolivia")}
                m, histories[name] = load_or_train(cfg, tag, dl["train"], dl["val"], BCEDice(cfg.pos_weight), device, in_ch=ch)
                report(name, m, "s2", tag, dl["val"], dl["test"], dl["bolivia"])
                del m, dl
                gc.collect(); torch.cuda.empty_cache()
            if not clear["bolivia"]:
                R["s2"]["bolivia_note"] = "Ningún chip de Bolivia quedó libre de nubes, así que la métrica de Bolivia del bono no es válida."

    # --- guardar resultados e informe ---
    R["minutes"] = (time.time() - t_start) / 60
    rows = [dict(modelo=n, grupo=v["group"], umbral=v["threshold"], **{f"{s}_{k}": v[s][k] for s in ("val", "test", "bolivia")
            for k in ("iou", "mean_iou_chips", "f1", "precision", "recall", "omission", "commission")}) for n, v in R["models"].items()]
    pd.DataFrame(rows).round(4).to_csv(res_dir / "resumen_modelos.csv", index=False)
    (res_dir / "resultados.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float))
    write_reports(R, out)
    print(f"\nTerminado en {R['minutes']:.1f} min. Resultados en {out.resolve()}")


if __name__ == "__main__":
    main(parse_args())
