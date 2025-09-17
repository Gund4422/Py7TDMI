# License

![License: IntiLicenseV1](https://img.shields.io/badge/License-IntiLicenseV1-red.svg "Totally official, trust me 😏")

---

Py7TDMI is licensed under the **Inti-License V1**.
You are free to use, modify, and distribute this software, but any derivative work must also be licensed under Inti-License v1.

# Py7TDMI

Yay! ARM7TDMI for Python enthusiasts.
Ever wanted to play with ARM7TDMI without a physical board? Now you can!

## Features

* Pythonic ARM7TDMI core implementation
* Supports ARM and Thumb instruction sets
* Step-by-step execution for learning or debugging
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
* **Me!** – For making the ARM7TDMI in Python
* **You!** – For supporting me \:D
* **NumPy** – For fast array operations
* **Numba** – For JIT compilation magic

## Badges

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Numba Support](https://img.shields.io/badge/numba-supported-green)
