"""
mzi_tutorial.py
===============
Учебный модуль к методичке "Интерферометр Маха–Цендера: от физики к геометрии".

Конфигурация: симметричный MZI (два направленных ответвителя 50:50 + два плеча).
Платформа: SOI strip 220 x 500 nm, TE-мода, рабочая длина волны ~1550 нм.

Все длины — в микрометрах (мкм), длины волн — в мкм, потери — в 1/мкм.

Запуск:  python mzi_tutorial.py   -> построит все 6 рисунков в папку images/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # рисуем в файл, без окна
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch

# ----------------------------------------------------------------------
# Константы платформы (SOI strip 220 x 500 нм, TE, ~1550 нм)
# Это типовые "учебные" значения. В реальном дизайне их берут из mode solver.
# ----------------------------------------------------------------------
N_EFF      = 2.45      # эффективный индекс моды
N_G        = 4.20      # групповой индекс (учитывает дисперсию)
ALPHA_PROP = 4.6e-5    # потери распространения, 1/мкм  (~2 дБ/см)
LAMBDA0    = 1.55      # рабочая длина волны, мкм
DN_DT      = 1.86e-4   # термооптический коэффициент кремния, 1/К
KAPPA_C    = 0.05      # сила связи направленного ответвителя, 1/мкм

# Линейная дисперсия n_eff, согласованная с n_g:
#   n_g = n_eff - lambda * dn_eff/dlambda  =>  dn_eff/dlambda = (n_eff - n_g)/lambda0
DNEFF_DLAMBDA = (N_EFF - N_G) / LAMBDA0   # 1/мкм


# ----------------------------------------------------------------------
# Уровень 1: геометрия -> базовые величины
# ----------------------------------------------------------------------
def n_eff_of_lambda(lam):
    """Эффективный индекс с учётом (линейной) дисперсии."""
    return N_EFF + DNEFF_DLAMBDA * (lam - LAMBDA0)


def arm_phase(lam, L):
    """Фаза, набранная за один проход по плечу длиной L: phi = 2*pi*n_eff*L/lambda."""
    return 2.0 * np.pi * n_eff_of_lambda(lam) * L / lam


def arm_amplitude(L, alpha=ALPHA_PROP):
    """Множитель a = exp(-alpha*L/2): доля АМПЛИТУДЫ, дожившая до конца плеча."""
    return np.exp(-alpha * L / 2.0)


def splitter_cross_power(Lc, kappa_c=KAPPA_C):
    """Доля МОЩНОСТИ, перешедшая в соседний волновод в ответвителе длиной Lc.
    Стандартная модель направленного ответвителя: P_cross = sin^2(kappa_c * Lc).
    50:50 достигается при kappa_c*Lc = pi/4."""
    return np.sin(kappa_c * Lc) ** 2


def delta_phase(lam, dL, phi_tune=0.0):
    """Разность фаз между плечами:
        Delta phi = 2*pi*n_eff*dL/lambda + phi_tune
    dL = L_long - L_short (геометрический разбаланс), phi_tune — внешняя подстройка
    (термооптика / электрооптика)."""
    return 2.0 * np.pi * n_eff_of_lambda(lam) * dL / lam + phi_tune


# ----------------------------------------------------------------------
# Уровень 2: спектр пропускания (два выхода)
# ----------------------------------------------------------------------
def mzi_transmission(lam, dL, split=0.5, imbalance_dB=0.0, phi_tune=0.0,
                     L_short=50.0):
    """
    Пропускание симметричного MZI с двумя одинаковыми ответвителями.
    Свет подан в верхний входной порт. Возвращает (T_bar, T_cross):
      bar   — выход на той же стороне, что и вход;
      cross — выход на противоположной стороне.

    Матрица одного ответвителя C = [[t, -i*k], [-i*k, t]],  t^2 + k^2 = 1,
    где k^2 = split (доля мощности в cross). Между ними — два плеча с фазами
    phi1, phi2 и амплитудами a1, a2. Полная матрица M = C * P * C даёт:

        T_cross = k^2 t^2 (a1^2 + a2^2 + 2 a1 a2 cos(dphi))
        T_bar   = t^4 a1^2 + k^4 a2^2 - 2 t^2 k^2 a1 a2 cos(dphi)

    dphi = phi1 - phi2 — разность фаз между плечами.
    """
    k2 = split
    t2 = 1.0 - split
    L_long = L_short + dL
    a1 = arm_amplitude(L_short)
    a2 = arm_amplitude(L_long) * 10.0 ** (-imbalance_dB / 20.0)
    dphi = delta_phase(lam, dL, phi_tune)
    cosd = np.cos(dphi)
    T_cross = k2 * t2 * (a1 ** 2 + a2 ** 2 + 2.0 * a1 * a2 * cosd)
    T_bar = t2 ** 2 * a1 ** 2 + k2 ** 2 * a2 ** 2 - 2.0 * t2 * k2 * a1 * a2 * cosd
    return T_bar, T_cross


# ----------------------------------------------------------------------
# Уровень 3: метрики (FOMs)
# ----------------------------------------------------------------------
def FSR(dL, lam=LAMBDA0):
    """Free Spectral Range — период спектра: FSR = lambda^2 / (n_g * dL).
    Та же форма, что у кольца, но вместо длины оборота L стоит разбаланс плеч dL."""
    return lam ** 2 / (N_G * dL)


def _arm_amps(dL, imbalance_dB, L_short=50.0):
    a1 = arm_amplitude(L_short)
    a2 = arm_amplitude(L_short + dL) * 10.0 ** (-imbalance_dB / 20.0)
    return a1, a2


def extinction_ratio_cross_dB(dL, imbalance_dB, L_short=50.0):
    """Глубина гашения на cross-выходе: ER = 20*log10|(a1+a2)/(a1-a2)|.
    Бесконечна при равных амплитудах плеч (a1 = a2) — это "критическое" условие MZI."""
    a1, a2 = _arm_amps(dL, imbalance_dB, L_short)
    denom = abs(a1 - a2)
    if denom < 1e-12:
        return np.inf
    return 20.0 * np.log10((a1 + a2) / denom)


def insertion_loss_cross_dB(dL, split=0.5, imbalance_dB=0.0, L_short=50.0):
    """Вносимые потери на пике cross-выхода: IL = -10*log10(T_cross_max).
    T_cross_max = k^2 t^2 (a1 + a2)^2."""
    a1, a2 = _arm_amps(dL, imbalance_dB, L_short)
    T_max = split * (1.0 - split) * (a1 + a2) ** 2
    return -10.0 * np.log10(T_max)


def finesse():
    """Финесс двухлучевого интерферометра. Форма линии — синус (cos^2),
    поэтому финесс фиксирован и равен ~2 (в отличие от резонатора)."""
    return 2.0


# ----------------------------------------------------------------------
# Уровень 4: термооптическая подстройка
# ----------------------------------------------------------------------
def thermo_phase(L_heater, dT, lam=LAMBDA0):
    """Набег фазы от нагрева плеча: dphi = (2*pi/lambda) * dn/dT * dT * L_heater."""
    return 2.0 * np.pi / lam * DN_DT * dT * L_heater


def heater_length_for_pi(dT, lam=LAMBDA0):
    """Длина нагревателя, дающая сдвиг фазы pi (полное переключение) при перегреве dT."""
    return lam / (2.0 * DN_DT * dT)


def Lc_for_5050(kappa_c=KAPPA_C):
    """Длина направленного ответвителя для деления 50:50: kappa_c*Lc = pi/4."""
    return (np.pi / 4.0) / kappa_c


# ----------------------------------------------------------------------
# Уровень 5: обратная задача (inverse design)
# ----------------------------------------------------------------------
def design_for_FSR_ER(FSR_target, ER_target, lam=LAMBDA0):
    """
    Подобрать разбаланс плеч dL под заданный FSR и допуск на разбаланс амплитуд
    под заданный ER. Возвращает словарь с параметрами.
    """
    dL = lam ** 2 / (N_G * FSR_target)                  # шаг 1: dL из FSR
    Lc = Lc_for_5050()                                  # шаг 2: ответвитель 50:50
    # шаг 3: из ER -> макс. допустимый разбаланс амплитуд r = |a1-a2|/(a1+a2)
    r = 10.0 ** (-ER_target / 20.0)
    # перевод в допуск по потерям одного плеча, дБ:
    #   a2/a1 = (1-r)/(1+r);  dB = -20*log10(a2/a1)
    imbalance_dB = -20.0 * np.log10((1.0 - r) / (1.0 + r))
    # шаг 4: длина нагревателя для полного переключения (pi) при перегреве 30 К
    L_heater = heater_length_for_pi(dT=30.0)
    return {"dL": dL, "Lc_5050": Lc, "max_imbalance_dB": imbalance_dB,
            "L_heater_pi": L_heater,
            "fab_ok": imbalance_dB > 0.02}  # ~0.02 дБ — реалистичный технологический допуск


# ----------------------------------------------------------------------
# Вспомогательное: форма линии резонатора (для контраста в бонусе)
# ----------------------------------------------------------------------
def _ring_dip(lam, dL_equiv, t=0.86, lam0=LAMBDA0):
    """Грубая модель провала all-pass кольца с тем же FSR, что у MZI с разбалансом dL_equiv.
    Нужна только для рисунка-контраста (узкий лоренциан против широкого синуса)."""
    A = 0.86
    phi = 2.0 * np.pi * n_eff_of_lambda(lam) * dL_equiv / lam
    num = t ** 2 - 2.0 * t * A * np.cos(phi) + A ** 2
    den = 1.0 - 2.0 * t * A * np.cos(phi) + (t * A) ** 2
    return num / den


# ======================================================================
#                         Р И С У Н К И
# ======================================================================
plt.rcParams.update({"font.size": 12, "figure.dpi": 130,
                     "axes.grid": True, "grid.alpha": 0.3})

COL_BAR   = "#2b6cb0"
COL_CROSS = "#c53030"
COL_OK    = "#2f855a"
COL_WG    = "#2d3748"


def _sbend(ax, x0, y0, x1, y1, color=COL_WG, lw=7):
    """Гладкий S-образный изгиб между точками через косинусную интерполяцию."""
    xs = np.linspace(x0, x1, 60)
    ys = y0 + (y1 - y0) * (1 - np.cos(np.pi * (xs - x0) / (x1 - x0))) / 2
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")


def fig1_schematic(path="images/fig1_schematic.png"):
    """Рисунок к уровню 1: устройство симметричного MZI."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.set_aspect("equal"); ax.axis("off")

    y_top, y_bot = 0.62, -0.62        # разнос плеч
    y_in = 0.18                       # сближение в ответвителях
    xc1a, xc1b = -2.4, -1.7           # первый ответвитель
    xc2a, xc2b = 1.7, 2.4             # второй ответвитель

    # --- входные стабы ---
    ax.plot([-3.4, xc1a], [y_in, y_in], color=COL_WG, lw=7, solid_capstyle="round")
    ax.plot([-3.4, xc1a], [-y_in, -y_in], color=COL_WG, lw=7, solid_capstyle="round")
    # --- первый ответвитель (две близкие линии + эллипс) ---
    ax.plot([xc1a, xc1b], [y_in, y_in], color=COL_WG, lw=7, solid_capstyle="round")
    ax.plot([xc1a, xc1b], [-y_in, -y_in], color=COL_WG, lw=7, solid_capstyle="round")
    ax.add_patch(Ellipse(((xc1a + xc1b) / 2, 0), 1.0, 0.78, fill=False,
                 lw=1.6, ls="--", color="gray"))
    ax.text((xc1a + xc1b) / 2, -0.62, "50:50", ha="center", fontsize=11, color="gray")

    # --- S-изгибы к плечам ---
    _sbend(ax, xc1b, y_in, xc1b + 0.6, y_top)
    _sbend(ax, xc1b, -y_in, xc1b + 0.6, y_bot)
    _sbend(ax, xc2a - 0.6, y_top, xc2a, y_in)
    _sbend(ax, xc2a - 0.6, y_bot, xc2a, -y_in)

    # --- верхнее плечо: длиннее на dL (петля вверх) + нагреватель ---
    xa, xb = xc1b + 0.6, xc2a - 0.6
    bump_x0, bump_x1 = -0.55, 0.55
    ax.plot([xa, bump_x0], [y_top, y_top], color=COL_WG, lw=7, solid_capstyle="round")
    ax.plot([bump_x1, xb], [y_top, y_top], color=COL_WG, lw=7, solid_capstyle="round")
    # петля удлинения
    bx = np.linspace(bump_x0, bump_x1, 80)
    by = y_top + 0.42 * np.sin(np.pi * (bx - bump_x0) / (bump_x1 - bump_x0))
    ax.plot(bx, by, color=COL_WG, lw=7, solid_capstyle="round")
    ax.annotate(r"+$\Delta L$", (0.0, y_top + 0.5), ha="center", fontsize=14)
    # нагреватель
    ax.add_patch(Rectangle((-0.32, y_top - 0.10), 0.64, 0.20, color="#dd6b20", alpha=0.9))
    ax.text(0.0, y_top - 0.30, "нагреватель", ha="center", fontsize=10, color="#9c4221")

    # --- нижнее плечо (опорное) ---
    ax.plot([xa, xb], [y_bot, y_bot], color=COL_WG, lw=7, solid_capstyle="round")

    # --- второй ответвитель ---
    ax.plot([xc2a, xc2b], [y_in, y_in], color=COL_WG, lw=7, solid_capstyle="round")
    ax.plot([xc2a, xc2b], [-y_in, -y_in], color=COL_WG, lw=7, solid_capstyle="round")
    ax.add_patch(Ellipse(((xc2a + xc2b) / 2, 0), 1.0, 0.78, fill=False,
                 lw=1.6, ls="--", color="gray"))
    ax.text((xc2a + xc2b) / 2, -0.62, "50:50", ha="center", fontsize=11, color="gray")

    # --- выходные стабы ---
    ax.plot([xc2b, 3.4], [y_in, y_in], color=COL_BAR, lw=7, solid_capstyle="round")
    ax.plot([xc2b, 3.4], [-y_in, -y_in], color=COL_CROSS, lw=7, solid_capstyle="round")

    # --- подписи портов ---
    ax.annotate("", xy=(-3.35, y_in), xytext=(-3.9, y_in),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
    ax.text(-3.95, y_in, r"$b_{in}$", ha="right", va="center", fontsize=14)
    ax.text(3.5, y_in, "bar",  ha="left", va="center", fontsize=12, color=COL_BAR)
    ax.text(3.5, -y_in, "cross", ha="left", va="center", fontsize=12, color=COL_CROSS)
    ax.text(-1.05, y_bot - 0.22, "опорное плечо", ha="center", va="top", fontsize=10,
            color=COL_WG)

    ax.set_xlim(-4.4, 4.4); ax.set_ylim(-1.25, 1.45)
    ax.set_title("Уровень 1. Симметричный интерферометр Маха–Цендера", fontsize=13)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig2_transmission(path="images/fig2_transmission.png"):
    """Рисунок к уровню 2: спектр пропускания обоих выходов, FSR."""
    dL = 100.0                         # разбаланс плеч 100 мкм
    lam = np.linspace(1.530, 1.570, 4000)
    T_bar, T_cross = mzi_transmission(lam, dL)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(lam * 1000, T_cross, lw=1.8, color=COL_CROSS, label="cross")
    ax.plot(lam * 1000, T_bar, lw=1.8, color=COL_BAR, label="bar")
    ax.set_xlabel("Длина волны, нм")
    ax.set_ylabel(r"Пропускание $T$")
    ax.set_ylim(-0.03, 1.12)
    ax.legend(loc="upper right", fontsize=10)

    # отметим FSR между двумя соседними максимумами cross
    idx = np.where((T_cross[1:-1] > T_cross[:-2]) & (T_cross[1:-1] > T_cross[2:]))[0] + 1
    if len(idx) >= 2:
        l1, l2 = lam[idx[0]] * 1000, lam[idx[1]] * 1000
        ax.annotate("", xy=(l2, 1.05), xytext=(l1, 1.05),
                    arrowprops=dict(arrowstyle="<->", color=COL_OK, lw=1.5))
        ax.text((l1 + l2) / 2, 1.07, f"FSR ≈ {l2 - l1:.1f} нм",
                ha="center", color=COL_OK, fontsize=12)

    ax.set_title(f"Уровень 2. Спектр пропускания (ΔL = {dL:.0f} мкм)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig3_metrics(path="images/fig3_metrics.png"):
    """Рисунок к уровню 3: ER, IL, точка квадратуры + три режима разбаланса плеч."""
    dL = 100.0
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    # --- левая панель: один период, ER / IL / квадратура ---
    lam = np.linspace(1.550, 1.550 + FSR(dL) * 1.05, 4000)
    _, T_cross = mzi_transmission(lam, dL, imbalance_dB=0.8)   # лёгкий разбаланс -> конечный ER
    axL.plot(lam * 1000, T_cross, lw=1.8, color=COL_CROSS)
    Tmax, Tmin = T_cross.max(), T_cross.min()
    lam_max = lam[np.argmax(T_cross)] * 1000
    # ER
    axL.annotate("", xy=(lam_max, Tmax), xytext=(lam_max, Tmin),
                 arrowprops=dict(arrowstyle="<->", color=COL_OK, lw=1.5))
    axL.text(lam_max + 0.3, (Tmax + Tmin) / 2,
             f"ER ≈ {extinction_ratio_cross_dB(dL, 0.8):.0f} дБ",
             color=COL_OK, fontsize=11, va="center")
    # IL
    axL.axhline(Tmax, ls="--", color="gray", lw=1.1)
    axL.text(lam[5] * 1000, Tmax + 0.015,
             f"IL ≈ {insertion_loss_cross_dB(dL, imbalance_dB=0.8):.2f} дБ",
             color="gray", fontsize=10)
    # квадратура (T = середина)
    half = (Tmax + Tmin) / 2
    j = np.argmin(np.abs(T_cross[:len(T_cross)//2] - half))
    axL.plot(lam[j] * 1000, T_cross[j], "o", color="black", ms=7)
    axL.annotate("квадратура\n(самый линейный участок)", (lam[j] * 1000, T_cross[j]),
                 (lam[j] * 1000 - 1.0, 0.25), fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", lw=1))
    axL.set_xlabel("Длина волны, нм"); axL.set_ylabel(r"$T_{cross}$")
    axL.set_ylim(-0.03, 1.12); axL.set_title("Один период: ER, IL, квадратура", fontsize=11)

    # --- правая панель: три режима разбаланса плеч ---
    lam2 = np.linspace(1.550, 1.550 + FSR(dL) * 2.05, 4000)
    regimes = [(0.0,  "balanced (a1=a2): max ER", COL_OK),
               (1.5,  "лёгкий разбаланс 1.5 дБ",  "#dd6b20"),
               (4.0,  "сильный разбаланс 4 дБ",   "#805ad5")]
    for imb, label, c in regimes:
        _, Tc = mzi_transmission(lam2, dL, imbalance_dB=imb)
        axR.plot(lam2 * 1000, Tc, lw=1.8, color=c, label=label)
    axR.set_xlabel("Длина волны, нм"); axR.set_ylabel(r"$T_{cross}$")
    axR.set_ylim(-0.03, 1.12); axR.legend(fontsize=9, loc="lower right")
    axR.set_title("Разбаланс плеч поднимает дно провала", fontsize=11)

    fig.suptitle("Уровень 3. Что мы измеряем: FSR, ER, IL", fontsize=13)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig4_geometry(path="images/fig4_geometry.png"):
    """Рисунок к уровню 4: геометрия -> физика. FSR(dL) и делитель Lc."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    # FSR(dL)
    dL = np.linspace(20, 400, 400)
    axL.plot(dL, FSR(dL) * 1000, lw=2, color=COL_CROSS)
    axL.axvline(100, ls="--", color="gray")
    axL.plot(100, FSR(100) * 1000, "o", color=COL_OK, ms=8)
    axL.annotate(f"ΔL=100 мкм →\nFSR≈{FSR(100)*1000:.1f} нм", (100, FSR(100) * 1000),
                 (160, FSR(100) * 1000 + 4), fontsize=10, color=COL_OK)
    axL.set_xlabel("Разбаланс плеч ΔL, мкм"); axL.set_ylabel("FSR, нм")
    axL.set_title("Больше ΔL → уже FSR", fontsize=11)

    # делитель: cross-мощность vs длина ответвителя
    Lc = np.linspace(0, 2 * Lc_for_5050(), 400)
    axR.plot(Lc, splitter_cross_power(Lc), lw=2, color=COL_BAR,
             label=r"$P_{cross}=\sin^2(\kappa_c L_c)$")
    axR.axhline(0.5, ls="--", color="gray")
    axR.axvline(Lc_for_5050(), ls="--", color=COL_OK,
                label=f"50:50 @ Lc≈{Lc_for_5050():.1f} мкм")
    axR.set_xlabel("Длина ответвителя Lc, мкм"); axR.set_ylabel("доля мощности в cross")
    axR.set_ylim(-0.03, 1.05); axR.legend(fontsize=9, loc="upper right")
    axR.set_title("Делитель: длиной Lc задаём 50:50", fontsize=11)

    fig.suptitle("Уровень 4. От геометрии (ΔL, Lc) к физике (FSR, делитель)", fontsize=13)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig5_designmap(path="images/fig5_designmap.png"):
    """Рисунок к уровню 5: карта дизайна — ER в плоскости (dL, разбаланс плеч)."""
    dL = np.linspace(40, 350, 240)
    imb = np.linspace(0.0, 3.0, 240)
    DL, IMB = np.meshgrid(dL, imb)

    ER_map = np.vectorize(lambda d, i: min(extinction_ratio_cross_dB(d, i), 60.0))(DL, IMB)
    FSR_map = FSR(DL) * 1000.0

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    pc = ax.pcolormesh(DL, IMB, ER_map, shading="auto", cmap="viridis")
    cb = fig.colorbar(pc, ax=ax); cb.set_label("ER (cross), дБ")

    # линии постоянного FSR (зависит только от dL -> вертикальные)
    cs = ax.contour(DL, IMB, FSR_map, levels=[4, 6, 8, 10, 14],
                    colors="white", linewidths=1.2)
    ax.clabel(cs, fmt="FSR=%.0f нм", fontsize=9)

    # гребень "balanced arms" (max ER) — аналог критической связи кольца
    ax.axhline(0.0, color=COL_CROSS, lw=2.5, label="balanced arms (max ER)")

    # пример точки дизайна
    d = design_for_FSR_ER(FSR_target=0.006, ER_target=25.0)
    ax.plot(d["dL"], d["max_imbalance_dB"], "*", ms=18, color="yellow",
            markeredgecolor="black",
            label=f"пример: FSR=6нм, ER=25дБ → ΔL={d['dL']:.0f}мкм, ≤{d['max_imbalance_dB']:.2f}дБ")

    ax.set_xlabel("Разбаланс плеч ΔL, мкм"); ax.set_ylabel("разбаланс амплитуд плеч, дБ")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Уровень 5. Карта дизайна: ΔL задаёт FSR, баланс плеч задаёт ER", fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig6_modulator(path="images/fig6_modulator.png"):
    """Бонус: балансный MZI как модулятор/переключатель + контраст с резонатором."""
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.8, 4.3))

    # (A) передаточная характеристика балансного MZI (dL=0) от фазы подстройки
    phi = np.linspace(0, 2 * np.pi, 600)
    _, Tc = mzi_transmission(LAMBDA0, dL=0.0, phi_tune=phi)
    Tb, _ = mzi_transmission(LAMBDA0, dL=0.0, phi_tune=phi)
    axA.plot(phi / np.pi, Tc, lw=2, color=COL_CROSS, label="cross")
    axA.plot(phi / np.pi, Tb, lw=2, color=COL_BAR, label="bar")
    axA.axvline(0.5, ls="--", color="black", lw=1)
    axA.text(0.5, 1.05, "квадратура", ha="center", fontsize=9)
    axA.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5),
                 arrowprops=dict(arrowstyle="<->", color=COL_OK, lw=1.4))
    axA.text(0.5, 0.4, r"$V_\pi$: сдвиг на $\pi$", ha="center", color=COL_OK, fontsize=10)
    axA.set_xlabel(r"фаза подстройки, $\times\pi$"); axA.set_ylabel(r"$T$")
    axA.set_ylim(-0.03, 1.15); axA.legend(fontsize=9, loc="upper right")
    axA.set_title("Передаточная характеристика", fontsize=11)

    # (B) малосигнальная модуляция: квадратура (линейно) vs пик (удвоение частоты)
    t = np.linspace(0, 2, 600)
    dphi_m = 0.5
    for bias, c, lbl in [(np.pi / 2, COL_OK, "bias=квадратура (линейно)"),
                         (0.0, "#dd6b20", "bias=пик (искажение, 2f)")]:
        phase = bias + dphi_m * np.sin(2 * np.pi * t)
        _, out = mzi_transmission(LAMBDA0, dL=0.0, phi_tune=phase)
        axB.plot(t, out, lw=1.8, color=c, label=lbl)
    axB.set_xlabel("время (периоды сигнала)"); axB.set_ylabel(r"$T_{cross}$")
    axB.set_ylim(-0.03, 1.05); axB.legend(fontsize=8, loc="upper right")
    axB.set_title("Где смещать рабочую точку", fontsize=11)

    # (C) контраст формы линии: двухлучевой MZI vs многолучевой резонатор
    dL_c = 100.0
    lam = np.linspace(1.550, 1.550 + 2 * FSR(dL_c), 4000)
    _, Tc_mzi = mzi_transmission(lam, dL_c)
    Tring = _ring_dip(lam, dL_c)
    axC.plot(lam * 1000, Tc_mzi, lw=1.8, color=COL_CROSS, label="MZI (2 луча, F≈2)")
    axC.plot(lam * 1000, Tring, lw=1.8, color=COL_BAR, label="кольцо (много лучей, F≫1)")
    axC.set_xlabel("Длина волны, нм"); axC.set_ylabel(r"$T$")
    axC.set_ylim(-0.03, 1.08); axC.legend(fontsize=8, loc="lower right")
    axC.set_title("Один FSR: широкий синус vs острый провал", fontsize=11)

    fig.suptitle("Бонус. Балансный MZI как модулятор и его отличие от резонатора",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("images", exist_ok=True)
    fig1_schematic()
    fig2_transmission()
    fig3_metrics()
    fig4_geometry()
    fig5_designmap()
    fig6_modulator()
    print("OK: 6 рисунков сохранены в images/")
    # короткая сводка чисел для проверки
    print(f"FSR(dL=100)        = {FSR(100)*1000:.2f} нм")
    print(f"Lc для 50:50       = {Lc_for_5050():.2f} мкм")
    print(f"ER @разбаланс 0.8дБ = {extinction_ratio_cross_dB(100, 0.8):.1f} дБ")
    print(f"L нагревателя на pi = {heater_length_for_pi(30.0):.1f} мкм @dT=30 K")
    print("inverse:", design_for_FSR_ER(0.006, 25.0))
