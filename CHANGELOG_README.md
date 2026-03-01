# Change Log README

## 2026-03-01 — ONNX/C++ Export & Runtime Robustness Update

### Summary
- Added ONNX export workflow via `tools/export_onnx.py`, including a wrapper over `TrajAirNet.inference` with 2 export inputs (`obs_traj`, `route_priors`) and dynamic axes for batch/agent dimensions.
- Added ONNX Runtime C++ inference sample: `cpp/onnx_infer.cpp` and build file `cpp/CMakeLists.txt`.
- Updated C++ sample build requirement to **C++11**.
- Made `model/trajairnet.py` device handling more robust by replacing hard `.cuda()` usage with device-aware `.to(...)` placement.
- Improved `model/utils.py::seq_collate_with_padding` so optional priors (`None` / empty tensor / missing column) are handled safely and batch unpacking is more robust.
- Updated `train.py` for mixed precision support (`autocast` + `GradScaler`) and safer gradient clipping behavior.
- Fixed `tools/export_onnx.py` import path handling so `python tools/export_onnx.py` works when launched outside the repository root.

### Affected Files
- `tools/export_onnx.py`
- `cpp/onnx_infer.cpp`
- `cpp/CMakeLists.txt`
- `model/trajairnet.py`
- `model/utils.py`
- `train.py`
