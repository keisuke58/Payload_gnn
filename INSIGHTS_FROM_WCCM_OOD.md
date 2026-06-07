# WCCM/OOD研究からPayloadへ反映する知見（2026-06-07）

姉妹プロジェクト（段間CFRP, WCCM/Composites B）のOOD・評価研究で得た知見を、Payload(H3フェアリングGNN-SHM)に適用するためのメモ。詳細は WCCM repo `OOD_RESULTS.md`。

## 1. 評価は多指標パネルで総合判断（単一指標禁止）
WCCMで macro-F1 がクラス0(99.97%)に底上げされ、OOD 0.33 でも **defect-only F1≈0**（欠陥分類は完全崩壊）だった。Payloadでも accuracy/F1 単独で判断しない。**必ず併記**:
- **detRec(=1−FNR)**: 見逃さない＝SHM最優先。
- **detFPR**: 誤警報＝運用コスト。
- **AUPRC**: 不均衡・閾値フリー。
- (局在課題なら) defect-only F1 / 重心誤差[mm] / IoU。
→ Payloadの healthy/defect 二値・5class局在も同パネルで。`train_gw.py` は既に rec/auc を出すが、**AUPRC・FPR・FNRを明示**する。

## 2. 「検出は汎化、精密局在は崩壊」
WCCM OOD: 異常検出は部分汎化(detRec 0.6–0.96)だが、正確なクラス/領域は当てられない(exact≈0)。
→ Payloadでも **「デボンドの有無」と「位置の精密推定」を分けて評価・主張**する。検出だけなら sim-to-real が通る可能性。

## 3. OODはアーキ非依存（過投資注意）
WCCM: in-dist 0.55–0.79 と差があるのに OOD は全7アーキ≈0.33（アーキ改善はOODに転移しない）。むしろ**単純GATがOOD検出最良**(高表現MGNはin-distに過適合)。
→ Payloadの **LGSTA(0.86) の優位が未知条件(別欠陥サイズ/位置/温度)に転移するか必ず検証**。転移しないなら、アーキでなくデータ/正規化/DAに投資。

## 4. 正規化スケールの統一とロバスト性検証（sim-to-realの罠）
Payload OGW sim-to-real 失敗の主因の一つが**特徴スケール不整合**(train std3247 vs OGW std1318)。WCCMでは「スケールを変えてもOOD不変＝モデルはスケール頑健」と確認できた。
→ Payloadでも **ドメイン間で特徴を共有スケーラで標準化**し、**スケール掃引でロバスト性を確認**する(`ood_scale.py`相当)。標準化だけで0/3→2/3に改善した実績あり。

## 5. sim-to-realの支配要因は構造ミスマッチ
OGW(平板+オメガストリンガ) vs 学習FEM(湾曲ハニカムフェアリング)＝別構造で転移失敗。
→ **構造整合FEM**（`FEM_OGW_MATCHED_DESIGN.md`）が最優先。正規化/few-shotでは埋まらない。

## 次アクション（Payload）
- (a) LGSTA/SAGEを **多指標パネル**で再評価（in-dist＋可能なOOD軸）。
- (b) OGW sim-to-realに **AUPRC/FNR/FPR** を明記し「検出 vs 局在」を分離報告。
- (c) スケール掃引でロバスト性確認。
- (d) 構造整合FEM生成（Abaqus並列ライセンス復帰後）。
