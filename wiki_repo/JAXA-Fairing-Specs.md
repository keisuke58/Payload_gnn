[← Home](Home)

# JAXA Fairing Specs: H3 ロケット フェアリング仕様

> 最終更新: 2026-02-28

## 1. プロジェクト対象: H3 フェアリング

### なぜ H3 か

H3はJAXAの次世代基幹ロケット（今後30年運用）。フェアリングに**CFRP AFP スキン + Al-Honeycomb コア**を採用した**JAXA初の機体**。

**決定的根拠**:

> Epsilon Users Manual p.17: *"The PLF employs the proven track record technology of H-IIA/B PLF. It has **aluminum-skin, aluminum-honeycomb-sandwich structure**."*
>
> KHI ANSWERS (H3): *「H3では**CFRPプリプレグ自動積層スキン/アルミハニカムサンドイッチパネル構造**とします」*

| 観点 | Epsilon (= H-IIA/B 継承) | **H3 (本プロジェクト)** |
|------|-------------------------|----------------------|
| **スキン材** | Al 7075 (0.3-0.6 mm) | **CFRP AFP** (T1000クラス, ~1.2 mm) |
| **コア材** | Al Honeycomb | Al Honeycomb (5052系) |
| **直径** | 2.6 m | **5.2 m** |
| **ガイド波分散** | 等方性 | **異方性**（繊維配向依存） |
| **CTE不整合** | ≈0 (Al同士) | **巨大** (CFRP: -0.3e-6 vs Al: 23e-6 /℃) |
| **SHMニーズ** | 低（成熟40年） | **高**（新材料系、F8事故で顕在化） |

### Epsilon の活用方法

H3 Users Manual は非公開のため、環境荷重データは Epsilon Manual で代替:
- 音響: Epsilon 147 dB ≈ H3 推定 ~148 dB（同クラス）
- 振動・衝撃: 両機とも Notched Bolt 分離、同等レベル
- 熱: 両機ともフェアリング外皮 100-200℃

## 2. フェアリング型式

| 型式 | 直径 | 全長 | 製造者 | 初飛行 |
| :--- | :--- | :--- | :--- | :--- |
| **Type-S** (Short) | 5.2 m | 10.4 m | KHI (川崎重工) | TF1 (2023) |
| **Type-L** (Long) | 5.2 m | 16.4 m | KHI | — |
| **Type-W** (Wide) | **5.4 m** | ~16 m | **Beyond Gravity** (旧RUAG) | F7 (2025/10) |

> Type-S と Type-L は「ロング形態の上部がショート形態になる」設計で、異なる衛星サイズに対応。

## 3. H3 フェアリング構造仕様 (KHI製 S/L型)

| パラメータ | 値 | ソース |
|-----------|-----|--------|
| **構造形式** | CFRP / Al-Honeycomb サンドイッチ | KHI, JAXA |
| **スキン材** | CFRP AFP (Toray T1000クラス) | KHI, 東レ |
| **コア材** | Al 5052 Honeycomb | 推定 (H-IIA継承) |
| **パネル総厚** | **~40 mm** | KHI ANSWERS |
| **スキン厚** | 推定 1.0-1.5 mm (8-12 ply) | 推定 |
| **コア厚** | 推定 ~38 mm | 総厚 − 両面スキン |
| **積層構成** | 準等方性 [45/0/-45/90]s (推定) | 推定 |
| **ノーズ形状** | **オジャイブ** (滑らかな流線形) | KHI |
| **分離機構** | Clamshell (2分割) + Notched Bolt (低衝撃) | JAXA |

### FEM シミュレーションモデル（検証済み — [[FEM-Visualization]]）

```
モデル範囲: 1/6 円筒バレルセクション（60°弧）
  半径: 2600 mm (φ5.2m)
  高さ: 5000 mm（代表セクション）
  外板: S4R Shell — CFRP T1000G [45/0/-45/90]s (1.0mm)
  コア: C3D8R Solid — Al-5052 Honeycomb (38mm)
  内板: S4R Shell — CFRP T1000G [45/0/-45/90]s (1.0mm)
  メッシュ: ~50mm (22,220 nodes)
  BC: z=0 固定 + θ=0°/60° 対称 (Uθ=0)
  荷重: 50 kPa 均一外圧
  バリデーション: 14/14 PASS
```

| パラメータ | モデル値 | H3推定値 | 状態 |
|-----------|--------|----------|------|
| `RADIUS` | 2600 mm | 2600 mm | OK |
| `HEIGHT` | 5000 mm | セクション | OK |
| `CORE_T` | 38 mm | ~38 mm | OK |
| `FACE_T` | 1.0 mm | 1.0-1.5 mm | OK |
| 最大変位 | 2.94 mm (0.11%R) | — | OK |
| 最大応力 | 74.3 MPa | < 800 MPa | OK |

## 4. 製造革新: AFP + OoA

H3 フェアリングの製造には 2 つの革新技術が導入された:

### AFP (Automated Fiber Placement / 自動繊維配置)

- 大型 AFP 装置で CFRP プリプレグテープをハニカムコア上に自動積層
- オジャイブ形状の高精度な曲面成形を実現
- 製造時間短縮 + コスト低減
- H-IIA/B のアルミスキンでは不可能だった流線形状 → 空気抵抗低減 + 搭載容積拡大

### OoA (Out of Autoclave / 脱オートクレーブ)

- オートクレーブ不使用の常圧加温成形
- 縦結合部: ボルト/ナットではなく **CFRP板 + 接着剤** で接合
- 大型オートクレーブ不要 → 設備コスト低減
- 接着接合により軽量化を達成

> **注意**: OoA + 接着接合は軽量化とコスト削減の利点がある一方、F8 事故では PSS の接着界面に製造時の剥離が確認された。接着品質管理の重要性が改めて浮き彫りになった。

## 5. Beyond Gravity 製 Type-W フェアリング

| パラメータ | 値 |
| :--- | :--- |
| **直径** | 5.4 m |
| **構造** | Al-Honeycomb コア + CFRP カバーレイヤー |
| **製造** | 自動化プロセス (オートクレーブ不使用) |
| **共通設計** | Ariane 6, Vulcan Centaur, Terran R と共有 |
| **初飛行** | F7 (2025/10/26, HTV-X1) |
| **付帯品** | **PSS (Payload Support Structure)** も供給 |
| **実績** | 全サイズ合計 **100% 成功率** |

> Beyond Gravity は PSS も供給しているが、F8 で破壊した PSS の製造者がBeyond Gravity か KHI かは公式には未公表。

## 6. 衛星保護機能

| 機能 | 仕様 |
| :--- | :--- |
| **温湿度管理** | 空調ドアによるフェアリング内部環境制御 |
| **防音** | 吸音材 (アコースティックブランケット), 内部 ~135–140 dB に低減 |
| **耐熱** | シリコンフォーム断熱材, **300°C 以上** 対応 |
| **分離** | 低衝撃分離 (Notched Bolt + 非火薬式アクチュエータ) |

## 7. 荷重環境

### 音響荷重（最重要 — SHM主目標）

| パラメータ | 値 |
|-----------|-----|
| **OASPL** | ~148 dB (推定) |
| **RMS圧力** | ~5,000 Pa |
| **帯域** | 100-1000 Hz |
| **発生時期** | リフトオフ直後 |
| **影響** | CFRP/Al界面の剥離主原因 |

### 空力荷重 (Max Q)

| パラメータ | 値 |
|-----------|-----|
| **最大動圧** | 30-40 kPa |
| **発生時期** | 打上後 ~60秒 |

### 分離衝撃

| パラメータ | 値 |
|-----------|-----|
| **衝撃レベル** | ~1,500 G |
| **持続時間** | <10 ms |
| **機構** | Notched Bolt (低衝撃) |

### 熱環境

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| **基準温度** | 25℃ | 打上前常温 |
| **外板表面** | 100-200℃ | 空力加熱ピーク |
| **内板表面** | ~50℃ | 遮蔽 |
| **CTE不整合** | CFRP(-0.3e-6) vs Al(23e-6) /℃ | **H3固有の課題** |

### 荷重の時系列

```
Time ──────────────────────────────────────────→
  0s          30s         60s         120s
  ├──────────┼───────────┼───────────┤
  │ Liftoff  │ Transonic │  Max Q    │ Fairing Sep
  │ ~148dB   │ Buffet    │ 30-40kPa  │ ~1500G
  │ Acoustic │ + Heating │ + 150℃    │ Shock
  └──────────┴───────────┴───────────┘
```

## 8. シミュレーションモデル

```
モデル範囲: 1/6 円筒バレルセクション（60°弧）
  半径: 2600 mm
  高さ: 5000 mm（代表セクション）
  要素: S4RT (Shell) + C3D8RT (Solid)
  メッシュ: ~80mm（Phase 1）
  材料: CFRP T1000クラス + Al-5052 Honeycomb
```

| パラメータ | 現行値 | H3推定値 | 状態 |
|-----------|--------|----------|------|
| `RADIUS` | 2600 mm | 2600 mm | OK |
| `HEIGHT` | 5000 mm | セクション | OK |
| `CORE_T` | 38 mm | ~38 mm | OK |
| `FACE_T` | 1.0 mm | 1.0-1.5 mm | 範囲内 |

## 9. 安全基準

**JAXA-JMR-002** (Payload Safety Standard):
- 安全率 SF ≥ 1.4-1.5
- 損傷許容: "No-Growth" 条件 → **CFRP接着構造に直接適用 → SHMが規格上も必要**

## 10. ソース

| Source | Content |
|--------|---------|
| [JAXA H3 諸元](https://www.rocket.jaxa.jp/rocket/h3/system.html) | フェアリング型式 |
| [JAXA H3 フェアリング](https://www.rocket.jaxa.jp/rocket/h3/faring.html) | CFRP/Al HC構造, オジャイブ形状 |
| [KHI H3](https://www.khi.co.jp/mobility/aero/space/h_3.html) | 製造担当 |
| [KHI ANSWERS](https://answers.khi.co.jp/ja/mobility/20210806j-01/) | パネル厚~40mm, AFP/OoA工法 |
| [SPACE Media](https://spacemedia.jp/spacebis/6234) | 耐熱300°C+, 衛星保護4機能, AFP詳細 |
| [Beyond Gravity H3](https://www.beyondgravity.com/en/news/beyond-gravity-rocket-nose-cone-celebrates-premiere-japans-h3-rocket) | Type-W, PSS供給 |
| [Epsilon Users Manual](JAXA_LIBRARY/EpsilonUsersManual_e.pdf) | 環境荷重データ (代替利用) |
| JAXA-JMR-002E | 安全基準 |
