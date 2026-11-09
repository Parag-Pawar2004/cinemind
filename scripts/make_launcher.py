"""Generates CineMind_Launcher.ipynb (kept as code so the notebook is reproducible)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("""# 🎬 CineMind — one-click launcher

Runs end-to-end with **one terminal command** (no clicking needed):

```bash
pip install jupyter
jupyter nbconvert --to notebook --execute --inplace CineMind_Launcher.ipynb
```

What it does, in order — every step is fail-safe, so the project always comes up:

1. **Unzips** `cinemind*.zip` sitting next to this notebook (skips if already extracted)
2. **Installs** dependencies into the current Python (+ optional local embedding model)
3. **Loads your free API keys** (from `.env`, environment vars, or prompts if run interactively)
4. **Ingests** every free data source available → builds the catalog, embeddings, vibe vectors and the CF model
5. **Evaluates** the recommender (MovieLens ranking metrics + agent query suite)
6. **Smoke-tests** the multi-agent pipeline
7. **Launches the dashboard** in the background and opens your browser

Environment switches: `CINEMIND_LITE=1` (skip the ~200 MB embedding model), `CINEMIND_OFFLINE=1` (no network),
`CINEMIND_REBUILD=1` (force rebuild), `CINEMIND_PORT=8501`."""),

    code("""# 1) Locate or unzip the project
import os, sys, zipfile, subprocess, pathlib, json, time, platform, socket, webbrowser
HERE = pathlib.Path.cwd()

def find_project():
    cands = [HERE, HERE / 'cinemind', *sorted(HERE.glob('cinemind*/')), *sorted(HERE.glob('*/cinemind*/'))]
    for c in cands:
        if (c / 'cinemind' / 'engine.py').exists():
            return c.resolve()
    return None

PROJ = find_project()
if PROJ is None:
    zips = sorted(HERE.glob('cinemind*.zip')) + sorted(HERE.parent.glob('cinemind*.zip'))
    if not zips:
        raise FileNotFoundError('Put cinemind.zip in the same folder as this notebook.')
    print('Unzipping', zips[0].name)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(HERE)
    PROJ = find_project()
os.chdir(PROJ)
sys.path.insert(0, str(PROJ))
# Optional: a .env placed next to the notebook is copied into the project
if (HERE / '.env').exists() and HERE != PROJ and not (PROJ / '.env').exists():
    (PROJ / '.env').write_text((HERE / '.env').read_text())
print('Project root:', PROJ)"""),

    code("""# 2) Install dependencies (idempotent)
def pip(*args):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-800:], r.stderr[-800:])
    return r.returncode == 0

ok = pip('-r', 'requirements.txt')
print('core deps:', 'ok' if ok else 'FAILED (see above)')
LITE = os.environ.get('CINEMIND_LITE') == '1' or os.environ.get('CINEMIND_OFFLINE') == '1'
if not LITE:
    try:
        import sentence_transformers  # noqa: F401
        print('sentence-transformers: already installed')
    except ImportError:
        print('Installing sentence-transformers (CPU, one-time)… set CINEMIND_LITE=1 to skip')
        if platform.system() == 'Linux':
            pip('torch', '--index-url', 'https://download.pytorch.org/whl/cpu')
        print('sentence-transformers:', 'ok' if pip('-r', 'requirements-full.txt') else 'skipped -> TF-IDF/LSA fallback')"""),

    code("""# 3) Free API keys (all optional)
import importlib
import cinemind.config as config
importlib.reload(config)
status = config.key_status()
missing = [k for k, v in status.items() if not v]
if missing and not (PROJ / '.env').exists():
    try:  # works in interactive Jupyter; raises under nbconvert (no stdin)
        from getpass import getpass
        print('Optional free keys - press Enter to skip each:')
        upd = {k: getpass(f'{k} ({config.API_KEYS[k]}): ').strip() for k in missing}
        config.save_env({k: v for k, v in upd.items() if v})
    except Exception:
        print('Non-interactive run: keys not prompted.')
status = config.key_status()
for k, v in status.items():
    print(f"  {'✅' if v else '—'} {k:20s} {'' if v else config.API_KEYS[k]}")
if not any(status.values()):
    print('\\nNo keys: CineMind runs on free keyless sources (TVMaze, Wikidata, MovieLens) + rule-based agents.')
    print('Add keys any time: dashboard → Settings tab, or `python scripts/configure.py`, then rebuild.')"""),

    code("""# 4) Build catalog + index (skips if already built, unless CINEMIND_REBUILD=1)
from cinemind.build import build
OFFLINE = os.environ.get('CINEMIND_OFFLINE') == '1'
need = os.environ.get('CINEMIND_REBUILD') == '1' or not config.CATALOG_FILE.exists()
if need:
    t = time.time()
    try:
        info = build(offline=OFFLINE)
    except Exception as e:
        print('Online build failed -> falling back to offline seed build:', e)
        info = build(offline=True)
    print(json.dumps({k: info[k] for k in ('sources', 'n_titles', 'by_industry', 'embedding_backend', 'cf')}, indent=1))
    print(f'built in {time.time()-t:.0f}s')
else:
    print('Index exists - skipping build. Build log:')
    print(config.BUILD_LOG.read_text() if config.BUILD_LOG.exists() else '(none)')"""),

    code("""# 5) Evaluate
from cinemind.engine import Engine
from cinemind.evaluate import run_all
engine = Engine()
report = run_all(engine, log=lambda *a: None)
rk = report['movielens_ranking']
if isinstance(rk, dict):
    import pandas as pd
    display(pd.DataFrame({k: v for k, v in rk.items() if isinstance(v, dict)}).T)
else:
    print('MovieLens ranking:', rk)
ag = report['agent_suite']
print('Agent filter precision:', ag['mean_filter_precision'], '| diversity:', ag['mean_diversity'])"""),

    code("""# 6) Smoke-test the multi-agent pipeline
from cinemind.agents import Orchestrator
orch = Orchestrator(engine)
for q in ['A dark, twisty revenge K-drama that is a slow burn',
          'Something like Inception but from Bollywood',
          'Epic Telugu mythology action movie']:
    out = orch.run(q)
    print(f'\\n> {q}   [intent: {out["trace"][0]["mode"]}, critic: {out["trace"][2]["mode"]}]')
    for r in out['results'][:5]:
        print(f"   • {r['title']} ({r['year']}, {r['industry']}) — {r['why'][:100]}")
src, res = engine.bridge('Parasite', 'Bollywood', n=5)
print('\\nBridge  Parasite → Bollywood:', [engine.items[c['idx']]['title'] for c in res])"""),

    code("""# 7) Launch the dashboard (detached, survives this notebook finishing)
import urllib.request

def healthy(port):
    try:
        return urllib.request.urlopen(f'http://localhost:{port}/_stcore/health', timeout=1).read() == b'ok'
    except Exception:
        return False

def free_port(start):
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return start

PORT = int(os.environ.get('CINEMIND_PORT', 8501))
if healthy(PORT):
    print(f'Dashboard already running → http://localhost:{PORT}')
else:
    PORT = free_port(PORT)
    log = open(PROJ / 'reports' / 'dashboard.log', 'w')
    kw = dict(cwd=PROJ, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if platform.system() == 'Windows':
        kw['creationflags'] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw['start_new_session'] = True
    proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app/dashboard.py',
                             '--server.port', str(PORT), '--server.headless', 'true'], **kw)
    for _ in range(90):
        if healthy(PORT):
            break
        time.sleep(1)
    print(('✅ Dashboard is live' if healthy(PORT) else '⚠️ Still starting (see reports/dashboard.log)')
          + f' → http://localhost:{PORT}   (pid {proc.pid})')
    if os.environ.get('CINEMIND_NO_BROWSER') != '1':
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except Exception:
            pass
print('Stop it later with:  ' + ('taskkill /F /IM streamlit.exe' if platform.system() == 'Windows'
                                  else "pkill -f 'streamlit run app/dashboard.py'"))"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent.parent / "CineMind_Launcher.ipynb"
nbf.write(nb, out)
print("wrote", out)
