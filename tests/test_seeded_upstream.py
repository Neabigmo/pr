import subprocess
import sys
from pathlib import Path


def test_seeded_upstream_reproduces_direct_script_local_imports(tmp_path: Path):
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 17\n", encoding="utf-8")
    script = tmp_path / "entry.py"
    script.write_text(
        "import random\nimport numpy as np\nimport torch\nfrom helper import VALUE\n"
        "print(VALUE, random.random(), float(np.random.rand()), float(torch.rand(1)))\n",
        encoding="utf-8",
    )
    wrapper = Path(__file__).resolve().parents[1] / "tools" / "run_seeded_upstream.py"
    command = [sys.executable, str(wrapper), "--seed", "123", "--script", str(script), "--"]
    first = subprocess.check_output(command, text=True).strip()
    second = subprocess.check_output(command, text=True).strip()
    assert first == second
    assert first.startswith("17 ")
