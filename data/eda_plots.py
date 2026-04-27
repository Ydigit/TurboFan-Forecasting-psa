import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "setting_1", "setting_2", "setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

train = pd.read_csv(Path(__file__).parent / "train.csv")
test  = pd.read_csv(Path(__file__).parent / "test.csv")

train_real = train[train["mask"] == 1]
test_real  = test[test["mask"] == 1]

print(f"Train: {len(train)} linhas ({len(train_real)} reais, {len(train)-len(train_real)} padding)")
print(f"Test : {len(test)} linhas ({len(test_real)} reais, {len(test)-len(test_real)} padding)")

# ── 1. Timeline global: sensor_11 com padding destacado ─────────────────────
print("Plot 1: timeline global...")
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

for ax, df, label, color in [
    (axes[0], train, "TRAIN (80 motores)", "steelblue"),
    (axes[1], test,  "TEST  (20 motores)", "tomato"),
]:
    real = df[df["mask"] == 1]
    pad  = df[df["mask"] == 0]
    ax.scatter(real["t"], real["sensor_11"], s=0.3, color=color, alpha=0.6, label="ciclo real")
    ax.scatter(pad["t"],  pad["sensor_11"],  s=0.3, color="lightgrey", alpha=0.4, label="padding (=0)")
    ax.set_ylabel("sensor_11 (escalado)")
    ax.set_title(f"{label} — timeline global com padding")
    ax.legend(markerscale=6, fontsize=8)
    ax.grid(True, alpha=0.2)

axes[0].set_xlabel("")
axes[1].set_xlabel("timestep global t")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_timeline_global.png", dpi=130)
plt.close()

# ── 2. Distribuicao de ciclos por motor ──────────────────────────────────────
print("Plot 2: ciclos por motor...")
train_cycles = train[train["mask"] == 1].groupby("engine_id").size()
test_cycles  = test[test["mask"] == 1].groupby("engine_id").size()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, cycles, label, color in [
    (axes[0], train_cycles, "TRAIN", "steelblue"),
    (axes[1], test_cycles,  "TEST",  "tomato"),
]:
    ax.hist(cycles, bins=20, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(cycles.mean(), color="black", linestyle="--", linewidth=1.5, label=f"media={cycles.mean():.0f}")
    ax.set_xlabel("numero de ciclos (duracao do voo)")
    ax.set_ylabel("contagem de motores")
    ax.set_title(f"{label} — distribuicao de ciclos por motor")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_ciclos_por_motor.png", dpi=130)
plt.close()

# ── 3. Variacao de cada feature (boxplots) ───────────────────────────────────
print("Plot 3: variacao por feature...")
fig, ax = plt.subplots(figsize=(16, 5))
train_real[FEATURE_COLS].boxplot(ax=ax, sym=".", medianprops={"color": "tomato"})
ax.set_title("Distribuicao das features no treino (ciclos reais, escalados)")
ax.set_ylabel("valor escalado (StandardScaler)")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_feature_variation.png", dpi=130)
plt.close()

# ── 4. Correlacao entre features e target ────────────────────────────────────
print("Plot 4: correlacoes...")
corr_data = train_real[FEATURE_COLS + ["target"]]
corr = corr_data.corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap completo
sns.heatmap(corr, ax=axes[0], cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.4, annot_kws={"size": 7})
axes[0].set_title("Matriz de correlacao (features + target)")

# Correlacao com target ordenada
corr_target = corr["target"].drop("target").sort_values(key=abs, ascending=True)
colors = ["tomato" if v > 0 else "steelblue" for v in corr_target.values]
axes[1].barh(corr_target.index, corr_target.values, color=colors, alpha=0.8)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].axvline( 0.3, color="grey", linestyle="--", linewidth=0.8, label="|r|=0.3")
axes[1].axvline(-0.3, color="grey", linestyle="--", linewidth=0.8)
axes[1].set_title("Correlacao de cada feature com o TARGET (sensor_11 t+1)")
axes[1].set_xlabel("coeficiente de Pearson")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_correlacoes.png", dpi=130)
plt.close()

# ── 5. Evolucao do target ao longo do ciclo de vida (normalizado) ────────────
print("Plot 5: degradacao normalizada...")
fig, ax = plt.subplots(figsize=(12, 5))

rng = np.random.default_rng(42)
sample_engines = rng.choice(train_real["engine_id"].unique(), 15, replace=False)

for eid in sample_engines:
    grp = train_real[train_real["engine_id"] == eid].reset_index(drop=True)
    life_pct = np.linspace(0, 100, len(grp))
    ax.plot(life_pct, grp["target"], alpha=0.4, linewidth=0.9, color="steelblue")

# Media por percentil de vida
all_interp = []
for eid in train_real["engine_id"].unique():
    grp = train_real[train_real["engine_id"] == eid]["target"].values
    interp = np.interp(np.linspace(0, 100, 100), np.linspace(0, 100, len(grp)), grp)
    all_interp.append(interp)
mean_curve = np.mean(all_interp, axis=0)
ax.plot(np.linspace(0, 100, 100), mean_curve, color="tomato", linewidth=2.5, label="media (todos os motores)")

ax.set_xlabel("% vida do motor (0=inicio, 100=falha)")
ax.set_ylabel("target escalado (sensor_11 t+1)")
ax.set_title("Degradacao do sensor_11 ao longo da vida — 15 motores + media")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_degradacao_normalizada.png", dpi=130)
plt.close()

# ── 6. Features com alta correlacao mutua (candidatas a remover) ─────────────
print("Plot 6: features altamente correlacionadas...")
feat_corr = train_real[FEATURE_COLS].corr().abs()
upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
high_corr_pairs = [(c, r, upper.loc[r, c])
                   for c in upper.columns for r in upper.index
                   if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.90]
high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

fig, ax = plt.subplots(figsize=(16, 6))
mask_tri = np.triu(np.ones_like(feat_corr, dtype=bool))
sns.heatmap(feat_corr, ax=ax, cmap="YlOrRd", mask=mask_tri,
            annot=True, fmt=".2f", linewidths=0.3, annot_kws={"size": 7},
            vmin=0, vmax=1)
ax.set_title("Correlacao absoluta entre features (triangulo inferior)\n"
             "Pares >0.90 sao candidatos a remover (redundantes)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_correlacao_entre_features.png", dpi=130)
plt.close()

print(f"\nPares com |r|>0.90 (candidatos a remover):")
for f1, f2, r in high_corr_pairs:
    print(f"  {f1:12s} <-> {f2:12s}  r={r:.3f}")

print(f"\nTodos os plots guardados em {PLOTS_DIR}/")
