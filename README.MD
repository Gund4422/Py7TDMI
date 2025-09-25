# ![License: IntiLicenseV2](https://img.shields.io/badge/License-IntiLicense-red.svg "Totally official, trust me 😏")

# License

Py7TDMI is licensed under **Inti-License Version 2 (V2)**.
You are free to **use, modify, and distribute** this software, but **any derivative work must also be licensed under Inti-License V2**.
Please credit **Intiha / Gund4422** and **ARM Holdings** for the original CPU design.

---

# Py7TDMI

Yay! ARM7TDMI for Python enthusiasts.
Ever wanted to play with ARM7TDMI **without a physical board**? Now you can!

## Features

* Pythonic ARM7TDMI core implementation
* Supports **ARM and Thumb instruction sets**
* Step-by-step execution for **learning or debugging**
* Educational and fun

## Quickstart

```python
import numpy as np
from py7tdmi import ARM7TDMI

# Create a memory array (1MB)
memory = np.zeros(1024*1024, dtype=np.uint8)

# Initialize CPU
cpu = ARM7TDMI(memory, trace=True)

# Step 10 instructions
cpu.step_batch(10)
```

## Credits & Thanks

* **ARM Holdings** – For designing the ARM7TDMI
* **Intiha / Gund4422** – For implementing ARM7TDMI in Python
* **You!** – For supporting the project 😎
* **NumPy** – For fast array operations
* **Numba** – For JIT compilation magic

## Badges

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Numba Support](https://img.shields.io/badge/numba-supported-green)

---

## License Header Snippet

You can include this at the top of all Python files:

```python
"""
Py7TDMI - Inti-License V2
Copyright (C) 2025 Intiha (aka Gund4422)

This software is licensed under Inti-License Version 2 (V2).
You may use, modify, and redistribute it, but all derivatives must remain under Inti-License V2.
Please credit Intiha / Gund4422 and ARM Holdings for the original CPU design.
"""
```
