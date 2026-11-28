"""Re-export kiến trúc V3 (src/v3/models_xai.py, bus_core.py) — giữ nguyên, đã kiểm thử 25/25 trong V3. Cần torch."""
from ._paths import ensure_v3
ensure_v3()
from models_xai import EXPERIMENTS, MAIN_CONFIGS, build_experiment      # noqa: E402,F401
from bus_core import build_effnet_b3, gradcam                           # noqa: E402,F401
