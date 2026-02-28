[← Home](Home)

# 積層構造 — Layup Structure

> H3 フェアリング サンドイッチパネルの積層構成。`generate_fairing_dataset.py` / `generate_realistic_fairing.py` で定義。

---

## 1. 概要

| 項目 | 値 |
|------|-----|
| **総層数** | 17層（スキン 8+8 + コア 1） |
| **総厚さ** | 40 mm |
| **積層構成** | [45/0/-45/90]s 準等方性積層 × 2（内・外スキン） |

---

## 3. 断面図

![Layup Structure](images/layup_structure.png)

---

## 4. 層別詳細

### Outer Skin（外板）

| Ply | 角度 | 厚さ | 材料 |
|-----|------|------|------|
| 1 | 45° | 0.125 mm | CFRP T1000G |
| 2 | 0° | 0.125 mm | CFRP T1000G |
| 3 | -45° | 0.125 mm | CFRP T1000G |
| 4 | 90° | 0.125 mm | CFRP T1000G |
| 5 | 90° | 0.125 mm | CFRP T1000G |
| 6 | -45° | 0.125 mm | CFRP T1000G |
| 7 | 0° | 0.125 mm | CFRP T1000G |
| 8 | 45° | 0.125 mm | CFRP T1000G |
| **合計** | | **1.0 mm** | |

### Core（コア）

| 層 | 厚さ | 材料 |
|----|------|------|
| 1 | 38 mm | Al-5052 ハニカム（直交異方性） |

### Inner Skin（内板）

| Ply | 角度 | 厚さ | 材料 |
|-----|------|------|------|
| 1–8 | [45/0/-45/90]s | 0.125 mm/ply | CFRP T1000G |
| **合計** | | **1.0 mm** | |

---

## 5. 関連

| ページ | 内容 |
|--------|------|
| [Architecture](Architecture) | FEM モデル概要 |
| [Node-Features](Node-Features) | 積層角度特徴量 (layup_0, layup_45, layup_minus45, layup_90) |
| [SHM-Context](SHM-Context) | 準等方性積層の注意点 |
