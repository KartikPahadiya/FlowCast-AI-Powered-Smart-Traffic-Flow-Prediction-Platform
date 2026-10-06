"""Exercise every dashboard view headlessly and report any exception."""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ["Live prediction", "Historical trends", "Congestion heatmap",
         "Road comparison", "Model performance", "Feature importance",
         "Forecast visualisation", "Prediction confidence",
         "Weather vs traffic", "🔧 Predict (custom)", "📤 Data upload",
         "📄 Reports & insights"]

app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
# hardcode the view instead of asking the sidebar radio
marker = 'view = st.sidebar.radio('
idx = app.index(marker)
end = app.index(')', app.index('"📄 Reports & insights",', idx))

failures = []
for v in VIEWS:
    patched = app[:idx] + f'view = {v!r}' + app[end + 1:]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(patched)
        tmp = f.name
    r = subprocess.run([sys.executable, "-c", f"""
import warnings; warnings.filterwarnings('ignore')
import __main__; __main__.__file__ = {tmp!r}
exec(compile(open({tmp!r}, encoding='utf-8').read(), {tmp!r}, 'exec'))
"""], capture_output=True, text=True, cwd=ROOT, timeout=240)
    tail = (r.stderr or "").strip().splitlines()
    bad = r.returncode != 0 or "Traceback" in (r.stderr or "")
    print(f"{'FAIL' if bad else 'ok  '}  {v}")
    if bad:
        failures.append((v, "\n".join(tail[-12:])))

print()
for v, err in failures:
    print("=" * 20, v, "=" * 20)
    print(err)
sys.exit(1 if failures else 0)
