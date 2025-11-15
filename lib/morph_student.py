import numpy as np
from .morph_core import indices_delaunay  # os alunos podem reutilizar

# ------------------------- Funções a implementar pelos estudantes -------------------------

def pontos_medios(pA, pB):
    """
    Retorna os pontos médios (N,2) entre pA e pB.
    """
    return (pA + pB) / 2.0

def indices_pontos_medios(pA, pB):
    """
    Calcula a triangulação de Delaunay nos pontos médios e retorna (M,3) int.
    Dica: use pontos_medios + indices_delaunay().
    """
    return indices_delaunay(pontos_medios(pA, pB))

# Interpoladoras
def linear(t, a=1.0, b=0.0):
    """
    Interpolação linear: a*t + b (espera-se mapear t em [0,1]).
    """
    return a * t + b

def sigmoide(t, k):
    """
    Sigmoide centrada em 0.5, normalizada para [0,1].
    k controla a "inclinação": maior k => transição mais rápida no meio.
    """

    # x eh a entrada da sigmoide
    x = k * (t - 0.5)
    s = 1.0 / (1.0 + np.exp(-x))

    # s(0) e s(1) devem ser zero e 1
    sMin = 1.0 / (1.0 + np.exp(k * 0.5))
    sMax = 1.0 / (1.0 + np.exp(-k * 0.5))

    # pra nao ter divisao por zero
    if np.abs(sMax - sMin) < 1e-9:
        return linear(t)
    
    return (s - sMin) / (sMax - sMin)

def dummy(t):
    """
    Função 'dummy' que pode ser usada como exemplo de função constante.
    """
    if isinstance(t, np.ndarray):
        return 0.5 * np.ones_like(t)
    return 0.5

# Geometria / warping por triângulos
def _det3(a, b, c):
    """
    Determinante 2D para área assinada (auxiliar das baricêntricas).
    """
    return a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])

def _transf_baricentrica(pt, tri):
    """
    pt: (x,y)
    tri: (3,2) com vértices v1,v2,v3
    Retorna (w1,w2,w3); espera-se w1+w2+w3=1 quando pt está no plano do tri.
    """
    v1, v2, v3 = tri
    detT = _det3(v1, v2, v3)

    # nao pode ter divisao por zero
    if np.abs(detT) < 1e-9:
        return -1.0, -1.0, -1.0
    
    w1 = _det3(pt, v2, v3) / detT
    w2 = _det3(v1, pt, v3) / detT
    w3 = _det3(v1, v2, pt) / detT
    
    return w1, w2, w3

def _check_bari(w1, w2, w3, eps=1e-6):
    """
    Testa inclusão de ponto no triângulo usando baricêntricas (com tolerância).
    """
    return (w1 >= -eps) and (w2 >= -eps) and (w3 >= -eps)

def _tri_bbox(tri, W, H):
    """
    Retorna bounding box inteiro (xmin,xmax,ymin,ymax), recortado ao domínio [0..W-1],[0..H-1].
    """
    xMin = np.floor(np.min(tri[:, 0])).astype(int)
    xMax = np.ceil(np.max(tri[:, 0])).astype(int)

    yMin = np.floor(np.min(tri[:, 1])).astype(int)
    yMax = np.ceil(np.max(tri[:, 1])).astype(int)
    
    
    xMin = max(0, xMin)
    yMin = max(0, yMin)

    xMax = min(W - 1, xMax)
    yMax = min(H - 1, yMax)
    
    return xMin, xMax, yMin, yMax

def _amostra_bilinear(img_float, x, y):
    """
    Amostragem bilinear em (x,y) com clamp nas bordas.
    img_float: (H,W,3) float32 [0,1] — retorna vetor (3,).
    """
    H, W = img_float.shape[:2]

    # pra que x e y fiquem dentro dos limites
    x = np.clip(x, 0, W - 1 - 1e-6)
    y = np.clip(y, 0, H - 1 - 1e-6)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    dc = x - x0
    dl = y - y0

    a1 = img_float[y0, x0] # cima-esquerda
    a2 = img_float[y0, x1] # cima-direita
    a3 = img_float[y1, x0] # baixo-esquerda
    a4 = img_float[y1, x1] # baixo-direita

    return (1 - dc) * (1 - dl) * a1 + dc * (1 - dl) * a2 + (1 - dc) * dl * a3 + dc * dl * a4

def gera_frame(A, B, pA, pB, triangles, alfa, beta):
    """
    Gera um frame intermediário por morphing com warping por triângulos.
    - A,B: imagens (H,W,3) float32 em [0,1]
    - pA,pB: (N,2) pontos correspondentes
    - triangles: (M,3) índices de triângulos
    - alfa: controla geometria (0=A, 1=B)
    - beta:  controla mistura de cores (0=A, 1=B)
    Retorna (H,W,3) float32 em [0,1].
    """
    H, W = A.shape[:2]
    frame = np.zeros_like(A)

    pT = (1.0 - alfa) * pA + alfa * pB

    for indices in triangles:
        triA = pA[indices]
        triB = pB[indices]
        triT = pT[indices]

        xMin, xMax, yMin, yMax = _tri_bbox(triT, W, H)

        for x in range(xMin, xMax):
            for y in range(xMin, xMax):
                pt = (x, y)
                w1, w2, w3 = _transf_baricentrica(pt, triT)
                if _check_bari(w1, w2, w3):
                    # mapeia (x, y) de volta pra A e B usando os pesos
                    xA = w1 * triA[0, 0] + w2 * triA[1, 0] + w3 * triA[2, 0]
                    yA = w1 * triA[0, 1] + w2 * triA[1, 1] + w3 * triA[2, 1]
                    
                    xB = w1 * triB[0, 0] + w2 * triB[1, 0] + w3 * triB[2, 0]
                    yB = w1 * triB[0, 1] + w2 * triB[1, 1] + w3 * triB[2, 1]
                    
                    # combina as cores
                    colorA = _amostra_bilinear(A, xA, yA)
                    colorB = _amostra_bilinear(B, xB, yB)
                    final_color = (1.0 - beta) * colorA + beta * colorB
                    frame[y, x] = final_color
    
    return frame
