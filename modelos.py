"""
Modelos 2.5D por estudio: un extractor 2D procesa cada triplete de cortes y un pooling por
atención (aprendizaje de múltiples instancias, MIL) combina los cortes en 12 salidas sigmoides.
Así una etiqueta por estudio (débil) entrena sin saber en qué corte está el hallazgo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import configuracion as C


class PoolingAtencion(nn.Module):
    """Pondera los cortes según su relevancia aprendida; devuelve vector del estudio y pesos."""
    def __init__(self, dim: int, oculto: int = 128):
        super().__init__()
        self.puntaje = nn.Sequential(nn.Linear(dim, oculto), nn.Tanh(), nn.Linear(oculto, 1))

    def forward(self, f):                      # f: (B, D, dim)
        pesos = torch.softmax(self.puntaje(f).squeeze(-1), dim=1)
        return (pesos.unsqueeze(-1) * f).sum(1), pesos


def bloque(entrada: int, salida: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(entrada, salida, 3, padding=1, bias=False), nn.BatchNorm2d(salida), nn.ReLU(inplace=True),
                         nn.Conv2d(salida, salida, 3, padding=1, bias=False), nn.BatchNorm2d(salida), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class CNNDesdeCero(nn.Module):
    """CNN 2.5D propia: 5 bloques conv (32->512 canales), GAP, atención entre cortes, 12 salidas."""
    def __init__(self, canales=(32, 64, 128, 256, 512), abandono: float = 0.3):
        super().__init__()
        capas, entrada = [], 3
        for c in canales:
            capas.append(bloque(entrada, c)); entrada = c
        self.extractor = nn.Sequential(*capas)
        self.capa_gradcam = self.extractor[-1]            # último bloque convolucional
        self.atencion = PoolingAtencion(entrada)
        self.cabeza = nn.Sequential(nn.Dropout(abandono), nn.Linear(entrada, C.N_ETIQUETAS))

    def caracteristicas(self, x):                # x: (B, D, 3, H, W)
        b, d = x.shape[:2]
        mapa = self.extractor(x.flatten(0, 1))
        return F.adaptive_avg_pool2d(mapa, 1).flatten(1).view(b, d, -1)

    def forward(self, x):
        estudio, pesos = self.atencion(self.caracteristicas(x))
        return self.cabeza(estudio), pesos


class ModeloTransferencia(CNNDesdeCero):
    """Backbone preentrenado en ImageNet (timm: resnet50, efficientnet_b0, ...) + misma cabeza MIL."""
    def __init__(self, nombre: str = "resnet50", preentrenado: bool = True, abandono: float = 0.3):
        nn.Module.__init__(self)
        import timm
        self.extractor = timm.create_model(nombre, pretrained=preentrenado, num_classes=0, global_pool="")
        dim = self.extractor.num_features
        self.capa_gradcam = self.extractor.layer4 if hasattr(self.extractor, "layer4") else self.extractor.conv_head
        self.atencion = PoolingAtencion(dim)
        self.cabeza = nn.Sequential(nn.Dropout(abandono), nn.Linear(dim, C.N_ETIQUETAS))


def crear_modelo(arquitectura: str, preentrenado: bool = True) -> nn.Module:
    return CNNDesdeCero() if arquitectura == "cero" else ModeloTransferencia(arquitectura, preentrenado)


# ------------------------------------------------------------------ pérdidas
def perdida_bce(logits, y, peso, pos_weight=None):
    """BCE con etiquetas suaves; 'peso' = confianza por celda (0 ignora la celda)."""
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (l * peso).sum() / peso.sum().clamp(min=1)


def perdida_focal(logits, y, peso, pos_weight=None, gamma: float = C.GAMMA_FOCAL):
    """Focal loss (Lin et al. 2017) con etiquetas suaves: baja el peso de ejemplos fáciles."""
    p = torch.sigmoid(logits)
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    pt = y * p + (1 - y) * (1 - p)
    return ((1 - pt) ** gamma * l * peso).sum() / peso.sum().clamp(min=1)
