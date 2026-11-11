#!/usr/bin/env bash
# Прогон движка на Movie-Dialog корпусе: data/data (83 097 диалогов).
#
#   ./run.sh          — вся цепочка: 1000 → ... → 80000
#   ./run.sh 1        — только первая сессия
#   ./run.sh 2        — до 3000 диалогов
#   ./run.sh 3        — до 6000 диалогов
#   ./run.sh 4        — до 11000 диалогов
#   ./run.sh 5        — до 20000 диалогов
#   ...
#   ./run.sh 11       — до 80000 диалогов
#
# Признаки задаются один раз, в первой сессии.
# Последующие сессии восстанавливают их из CK.cfg.
# --files — абсолютная граница от начала корпуса.
# --repeat 3 — каждый диапазон читается три раза подряд: первый проход это
#   новое (сознание включено, сюрпризов много), второй и третий — тот же
#   материал второй раз (движок его запомнил, сюрпризов почти нет, сознание
#   гаснет). Отложенная часть (каждый 10-й диалог) в проходах одна и та же,
#   поэтому строки сравнимы.
# --window 6 --joint 3,4,5,6 — длинные признаки, строк в памяти в ~1.7 раза
#   больше, чем при окне 4 с признаками 3,4. Нужно ~10 ГБ свободной RAM.

set -euo pipefail
cd "$(dirname "$0")"

BIN=./target/release/mind
RUNS=data/runs
CK="$RUNS/m.bin"
LOG="$RUNS/s.csv"
COMMON=(--no-chat --steps 0 --minutes 0 --ram 15 --repeat 3)

mkdir -p "$RUNS"

if [ ! -x "$BIN" ] || [ "${1:-}" = "" ]; then
  echo "--- сборка ---"
  cargo build --release
fi

session1() {
  echo "--- сессия 1: до 1000 диалогов, задаёт признаки ---"
  "$BIN" "${COMMON[@]}" --files 1000 --window 6 --joint 3,4,5,6 --save "$CK" --log "$LOG"
}

session2() {
  echo "--- сессия 2: до 3000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 3000 --load "$CK" --save "$CK" --log "$LOG"
}

session3() {
  echo "--- сессия 3: до 6000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 6000 --load "$CK" --save "$CK" --log "$LOG"
}

session4() {
  echo "--- сессия 4: до 11000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 11000 --load "$CK" --save "$CK" --log "$LOG"
}

session5() {
  echo "--- сессия 5: до 20000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 20000 --load "$CK" --save "$CK" --log "$LOG"
}

session6() {
  echo "--- сессия 6: до 30000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 30000 --load "$CK" --save "$CK" --log "$LOG"
}

session7() {
  echo "--- сессия 7: до 40000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 40000 --load "$CK" --save "$CK" --log "$LOG"
}

session8() {
  echo "--- сессия 8: до 50000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 50000 --load "$CK" --save "$CK" --log "$LOG"
}

session9() {
  echo "--- сессия 9: до 60000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 60000 --load "$CK" --save "$CK" --log "$LOG"
}

session10() {
  echo "--- сессия 10: до 70000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 70000 --load "$CK" --save "$CK" --log "$LOG"
}

session11() {
  echo "--- сессия 11: до 80000 диалогов ---"
  "$BIN" "${COMMON[@]}" --files 80000 --load "$CK" --save "$CK" --log "$LOG"
}

case "${1:-all}" in
  1) session1 ;;
  2) session2 ;;
  3) session3 ;;
  4) session4 ;;
  5) session5 ;;
  6) session6 ;;
  7) session7 ;;
  8) session8 ;;
  9) session9 ;;
  10) session10 ;;
  11) session11 ;;
  all)
    session1
    session2
    session3
    session4
    session5
    session6
    session7
    session8
    session9
    session10
    session11
    ;;
  *) echo "usage: ./run.sh [1-11]"; exit 1 ;;
esac

echo
echo "=== результат (по проходам) ==="
cat "$LOG"
