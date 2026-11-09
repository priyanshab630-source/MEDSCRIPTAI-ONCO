"""Test defaults: no real docking unless a test asks for it.

Docking is on in the app (DOCKING_ENABLED defaults to 1). In tests it is off, because
a Vina grid on a 4,000-atom receptor inside every SHP2 analysis turns a 90-second suite
into a ten-minute one. The docking tests turn it back on explicitly.
"""

import os

os.environ.setdefault("DOCKING_ENABLED", "0")
