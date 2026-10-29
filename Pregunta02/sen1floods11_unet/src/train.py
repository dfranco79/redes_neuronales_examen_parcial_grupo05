"""Entrenamiento: se queda con la época de mayor IoU en la validación dorada y guarda checkpoints reutilizables."""
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .metrics import collect, evaluate_chips
from .model import UNet


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_one_epoch(model, loader, opt, crit, device, scaler, use_amp):
    model.train()
    tot = n = 0
    for img, lbl, valid, _ in loader:
        img, lbl, valid = img.to(device, non_blocking=True), lbl.to(device), valid.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(img)
        loss = crit(logits, lbl, valid)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tot += loss.item() * img.size(0)
        n += img.size(0)
    return tot / max(n, 1)


def load_or_train(cfg, tag, train_loader, val_loader, crit, device, in_ch=2, epochs=None, lr=None,
                  init_state=None, extra_sig=None, log=print):
    """Entrena el modelo o, si ya hay un checkpoint con la misma configuración, lo carga. Devuelve (modelo, historial).

    La firma guarda lo que cambia el resultado (arquitectura, épocas, lr, pérdida, nº de chips...). Si el checkpoint
    tiene otra firma (por ejemplo, de una prueba rápida), se vuelve a entrenar.
    """
    epochs, lr = epochs or cfg.epochs, lr or cfg.lr
    sig = dict(in_ch=in_ch, base_ch=cfg.base_ch, depth=cfg.depth, epochs=epochs, lr=lr, crop=cfg.crop,
               batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, pos_weight=cfg.pos_weight, seed=cfg.seed,
               loss=type(crit).__name__, n_train=len(train_loader.dataset), n_val=len(val_loader.dataset),
               init=init_state is not None, **(extra_sig or {}))
    path = Path(cfg.out_dir) / "checkpoints" / f"{tag}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = UNet(in_ch, 1, cfg.base_ch, cfg.depth).to(device)
    if path.exists() and not cfg.force_retrain:
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("sig") == sig:
            model.load_state_dict(ck["state"])
            log(f"{tag}: checkpoint cargado (mejor IoU val={ck['best_iou']:.4f})")
            return model, ck["history"]
        log(f"{tag}: el checkpoint guardado tiene otra configuración, se vuelve a entrenar")
    if init_state is not None:
        model.load_state_dict(init_state)
    set_seed(cfg.seed)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, hist = -1.0, deepcopy(model.state_dict()), []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, crit, device, scaler, use_amp)
        vm, _ = evaluate_chips(collect(model, val_loader, device), 0.5)
        sch.step()
        hist.append(dict(epoch=ep, train_loss=float(tr), val_iou=float(vm["iou"]), val_f1=float(vm["f1"])))
        star = vm["iou"] > best
        if star:
            best, best_state = float(vm["iou"]), deepcopy(model.state_dict())
        log(f"[{tag}] {ep:02d}/{epochs} loss={tr:.4f} | val IoU={vm['iou']:.4f} F1={vm['f1']:.4f} "
            f"P={vm['precision']:.3f} R={vm['recall']:.3f} | {time.time() - t0:.1f}s{' *' if star else ''}")
    model.load_state_dict(best_state)
    torch.save(dict(state=best_state, best_iou=best, history=hist, sig=sig,
                    config=dict(in_channels=in_ch, base_ch=cfg.base_ch, depth=cfg.depth)), path)
    return model, hist


def load_model(cfg, tag, device, in_ch=2):
    path = Path(cfg.out_dir) / "checkpoints" / f"{tag}.pt"
    model = UNet(in_ch, 1, cfg.base_ch, cfg.depth).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state"])
    return model.eval()
