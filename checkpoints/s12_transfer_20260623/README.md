# S12 転移学習チェックポイント (2026-06-23)

arch: SAGE, hidden=128, layers=4, focal γ=2.0
task: binary 3D欠陥位置推定 (defect/healthy per node)
data faithful: processed_s12_czm_96_binary (15,206 nodes/sample, 機械的荷重)
data proxy:    processed_s12_mixed_400     (15,206 nodes/sample, 熱荷重)

| ファイル | EXP | seed | val_opt_F1 | 備考 |
|---------|-----|------|-----------|------|
| exp1_scratch_seed42.pt  | EXP1 from-scratch (binary) | 42  | 0.9384 | ep232 |
| exp1_scratch_seed123.pt | EXP1 from-scratch (binary) | 123 | 0.8964 | ep52  |
| exp1_scratch_seed7.pt   | EXP1 from-scratch (binary) | 7   | 0.9060 | ep48  |
| exp2a_proxy_seed42.pt   | EXP2a proxy-only  (binary) | 42  | 0.8469 | ep190 |
| exp2a_proxy_seed123.pt  | EXP2a proxy-only  (binary) | 123 | 0.8632 | ep268 |
| exp2a_proxy_seed7.pt    | EXP2a proxy-only  (binary) | 7   | 0.8576 | ep160 |
| exp3_finetune_seed42.pt | EXP3 fine-tune ★  (binary) | 42  | 0.9284 | ep138 |
| exp3_finetune_seed123.pt| EXP3 fine-tune ★  (binary) | 123 | 0.9271 | ep107 |
| exp3_finetune_seed7.pt  | EXP3 fine-tune ★  (binary) | 7   | 0.9362 | ep42  |
| exp4_8class_seed42.pt   | EXP4 8-class      (multi)  | 42  | 0.8774 | ep81  |
| exp4_8class_seed123.pt  | EXP4 8-class      (multi)  | 123 | 0.8921 | ep144 |
| exp4_8class_seed7.pt    | EXP4 8-class      (multi)  | 7   | 0.8851 | ep142 |

★ EXP3 mean=0.931±0.004 = oracle (EXP1 scratch 0.914 を上回る)
EXP4 8-class mean OptF1=0.885±0.006, mean Macro-F1=0.328±0.033
  8クラス: healthy/debonding/fod/impact/delamination/inner_debond/thermal_progression/acoustic_fatigue
  acoustic_fatigue = F1 0.000 (全 seed) — 他クラスと特徴量の重なり

EXP2b zero-shot はチェックポイント = exp2a_proxy と同一 (fine-tune なし直接適用)
