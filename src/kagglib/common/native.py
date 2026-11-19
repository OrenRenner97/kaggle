"""Сборка и загрузка нативных (C/C++) моделей без зависимостей."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np


def compile_shared(sources: list[Path], out: Path, std: str = "c++17") -> Path:
    """g++ -O2 -shared -fPIC sources -o out. Возвращает путь к .so."""
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        raise RuntimeError("не найден C++ компилятор (g++ или clang++)")
    cmd = [cxx, f"-std={std}", "-O2", "-shared", "-fPIC", *map(str, sources), "-o", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


class NativeBinaryModel:
    """Обёртка над .so с функцией `int <name>(const float* X, size_t rows, size_t cols, double* out)`."""

    def __init__(self, lib_path: Path, symbol: str, n_features: int):
        self._lib = ctypes.CDLL(str(lib_path))
        self._fn = getattr(self._lib, symbol)
        self._fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_double)]
        self._fn.restype = ctypes.c_int
        self.n_features = n_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"ожидается матрица (n, {self.n_features}), пришла {X.shape}")
        out = np.empty(X.shape[0], dtype=np.float64)
        rc = self._fn(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X.shape[0], X.shape[1],
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if rc != 0:
            raise RuntimeError(f"нативная модель вернула код ошибки {rc}")
        return out
