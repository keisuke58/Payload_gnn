#!/bin/bash
# batch_gw_sensor_patterns.sh — 100センサで1回解析し、10/20/30/50 を抽出
#
# 再解析不要。batch_generate_gw_dataset.sh all のエイリアス。
#
# Usage:
#   bash scripts/batch_gw_sensor_patterns.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/batch_generate_gw_dataset.sh all
