import os, sys

V3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3')


def ensure_v3():
    """Thêm src/v3 vào sys.path để import bus_core, bus_trust, models_xai... (module V3 giữ nguyên)."""
    if V3_DIR not in sys.path:
        sys.path.insert(0, V3_DIR)
    return V3_DIR
