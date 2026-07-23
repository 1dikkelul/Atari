const ALE_WASM_LOCAL = './vendor/ale-wasm/ale.js';
const ALE_WASM_CDN = 'https://cdn.jsdelivr.net/npm/@farama/ale-wasm/+esm';

const DEFAULT_OPPONENTS = [
  { id: 'human', label: 'Human (you) vs Atari AI', kind: 'human' },
  {
    id: 'sb3_v1',
    label: 'SB3 Pong v1 vs Atari AI',
    kind: 'onnx',
    modelPath: './models/sb3_pong_actor.onnx',
    inferenceMode: 'sample',
    temperature: 1.0,
  },
  // Placeholder for future PettingZoo/AgileRL export path.
  { id: 'pz_gen100', label: 'PettingZoo Gen100 (coming soon)', kind: 'onnx', modelPath: './models/pong_champ_gen_100.onnx', enabled: false },
];
const OPPONENTS_MANIFEST_URL = './models/opponents.json';
const MODELS_DIR_URL = './models/';

const LEADERBOARD_KEY = 'ale_wasm_casual_leaderboard_v1';
const DEFAULT_ROM_URL = './roms/pong.bin';

class ALECasualApp {
  constructor() {
    this.canvas = document.getElementById('gameCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.canvas.tabIndex = 0;

    this.nativeCanvas = document.createElement('canvas');
    this.nativeCtx = this.nativeCanvas.getContext('2d');

    this.preCanvas = document.createElement('canvas');
    this.preCanvas.width = 84;
    this.preCanvas.height = 84;
    this.preCtx = this.preCanvas.getContext('2d', { willReadFrequently: true });

    this.rawCanvas = document.createElement('canvas');
    this.rawCtx = this.rawCanvas.getContext('2d');

    this.actionTrendCanvas = document.getElementById('actionTrendCanvas');
    this.actionTrendCtx = this.actionTrendCanvas ? this.actionTrendCanvas.getContext('2d') : null;
    this.actionSnapshotCanvas = document.getElementById('actionSnapshotCanvas');
    this.actionSnapshotCtx = this.actionSnapshotCanvas ? this.actionSnapshotCanvas.getContext('2d') : null;
    this.actionTrend = [];
    this.trendIntervalMs = 1000;
    this.trendWindowSec = 30;
    this.trendMaxPoints = 240;
    this.trendCycleStartSec = 0;
    this.trendBinStartMs = 0;
    this.trendBinCounts = { dominant: 0, total: 0 };
    this.lastChosenModelIndex = null;
    this.currentActionIntent = 'noop';

    this.snapshotPeriodMs = 30000;
    this.snapshotDurationMs = 1000;
    this.snapshotBinMs = 50;
    this.snapshotNextCaptureAtMs = 0;
    this.snapshotCaptureStartMs = null;
    this.snapshotCaptureEvents = [];
    this.snapshotSeries = [];
    this.snapshotTickSeries = [];
    this.lastSnapshotCapturedAtMs = 0;
    this.lastDecisionNonDominant = false;
    this.lastDecisionIntent = null;

    this.frameStack = [];
    this.rawFramePair = [];
    this.currentAIAction = 0;
    this.inferenceBusy = false;

    this.running = false;
    this.lastStepAt = 0;
    this.stepMs = 1000 / 60;
    this.frameCounter = 0;
    this.rewardSum = 0;

    this.playerName = 'Player1';
    this.opponents = DEFAULT_OPPONENTS.slice();
    this.selectedOpponent = this.opponents[0];
    this.matchStart = 0;

    this.keys = {};
    this.invertControls = false;
    this.boundKeyDown = (e) => this.onKeyEvent(e, true);
    this.boundKeyUp = (e) => this.onKeyEvent(e, false);
    window.addEventListener('keydown', this.boundKeyDown, { passive: false });
    window.addEventListener('keyup', this.boundKeyUp, { passive: false });
    document.addEventListener('keydown', this.boundKeyDown, { passive: false });
    document.addEventListener('keyup', this.boundKeyUp, { passive: false });
  }

  normalizeKey(key) {
    if (!key) return '';
    if (key === 'ArrowUp' || key === 'Up') return 'ArrowUp';
    if (key === 'ArrowDown' || key === 'Down') return 'ArrowDown';
    return key.length === 1 ? key.toLowerCase() : key;
  }

  onKeyEvent(event, pressed) {
    const key = this.normalizeKey(event.key);
    if (key === 'ArrowUp' || key === 'ArrowDown' || key === 'w' || key === 's') {
      event.preventDefault();
    }
    this.keys[key] = pressed;
  }

  async init() {
    const status = document.getElementById('engineStatus');
    status.textContent = 'Engine: Loading ALE WASM...';

    let createALEModule;
    let usingLocalBundle = false;
    try {
      createALEModule = await this.loadLocalAleBundle();
      usingLocalBundle = true;
      this.uiLog('Loaded ALE WASM from local bundle.');
    } catch (err) {
      this.uiLog(`Local ALE WASM load failed: ${err.message}`, true);
      this.uiLog('Falling back to CDN import path.', true);
      try {
        ({ default: createALEModule } = await import(ALE_WASM_CDN));
      } catch (cdnErr) {
        this.uiLog(`ALE WASM CDN import failed: ${cdnErr.message}`, true);
        this.uiLog('Check that public/vendor/ale-wasm/ale.js exists and that the server can serve .wasm/.data files.', true);
        throw cdnErr;
      }
    }

    const moduleConfig = usingLocalBundle
      ? {
          // Local UMD build expects ale.wasm and ale.data resolved via locateFile.
          locateFile: (path) => `./vendor/ale-wasm/${path}`,
        }
      : {};

    this.ALE = await createALEModule(moduleConfig);
    this.ale = new this.ALE.ALEInterface();

    // Match common Atari training defaults (no sticky actions) for inference parity.
    if (typeof this.ale.setFloat === 'function') {
      try {
        this.ale.setFloat('repeat_action_probability', 0.0);
      } catch (_) {
        // Some builds may not expose this config key.
      }
    }

    this.uiLog(`ALE ready (version ${this.ALE.ALEInterface.getVersion()}).`);
    status.textContent = 'Engine: ALE WASM Ready';
    status.style.color = '#00ff66';

    await this.loadOpponentsManifest();
    await this.discoverOnnxOpponents();
    this.renderLeaderboard();
    this.populateOpponentSelect();
    this.wireControls();

    // Attempt to auto-load a bundled Pong ROM if present in public/roms.
    try {
      await this.loadBundledRom(DEFAULT_ROM_URL);
    } catch (err) {
      this.uiLog(`Auto-load failed for ${DEFAULT_ROM_URL}: ${err.message}`, true);
    }

    requestAnimationFrame(() => this.loop());
  }

  uiLog(msg, isError = false) {
    const box = document.getElementById('systemLog');
    if (!box) return;
    const color = isError ? '#ff4d78' : '#a0a0b0';
    box.innerHTML += `<div style="color:${color}">[${new Date().toLocaleTimeString()}] ${msg}</div>`;
    box.scrollTop = box.scrollHeight;
  }

  async loadOpponentsManifest() {
    try {
      const response = await fetch(OPPONENTS_MANIFEST_URL, { cache: 'no-store' });
      if (!response.ok) {
        this.uiLog(`Opponent manifest not found (${response.status}), using defaults.`);
        return;
      }

      const data = await response.json();
      if (!Array.isArray(data)) {
        this.uiLog('Opponent manifest format invalid, using defaults.', true);
        return;
      }

      const cleaned = data
        .filter((opp) => opp && typeof opp === 'object')
        .filter((opp) => typeof opp.id === 'string' && typeof opp.label === 'string' && typeof opp.kind === 'string')
        .filter((opp) => opp.enabled !== false);

      const hasHuman = cleaned.some((opp) => opp.kind === 'human');
      if (!hasHuman) {
        cleaned.unshift({ id: 'human', label: 'Human (you) vs Atari AI', kind: 'human' });
      }

      if (cleaned.length > 0) {
        this.opponents = cleaned;
        const stillExists = this.opponents.find((opp) => opp.id === this.selectedOpponent.id);
        this.selectedOpponent = stillExists || this.opponents[0];
        this.uiLog(`Loaded ${this.opponents.length} opponents from manifest.`);
      }
    } catch (err) {
      this.uiLog(`Failed to load opponent manifest: ${err.message}. Using defaults.`, true);
    }
  }

  async discoverOnnxOpponents() {
    try {
      const response = await fetch(MODELS_DIR_URL, { cache: 'no-store' });
      if (!response.ok) {
        this.uiLog(`Model directory scan skipped (HTTP ${response.status}).`);
        return;
      }

      const html = await response.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');
      const anchors = Array.from(doc.querySelectorAll('a[href]'));

      const modelFiles = anchors
        .map((a) => a.getAttribute('href') || '')
        .map((href) => {
          try {
            const u = new URL(href, response.url);
            return decodeURIComponent(u.pathname.split('/').pop() || '');
          } catch {
            return '';
          }
        })
        .filter((name) => /\.onnx$/i.test(name));

      const uniqueModelFiles = Array.from(new Set(modelFiles)).sort((a, b) => a.localeCompare(b));
      if (uniqueModelFiles.length === 0) {
        this.uiLog('Model directory scan found no .onnx files.');
        return;
      }

      const discovered = [];
      for (const fileName of uniqueModelFiles) {
        const modelPath = `./models/${fileName}`;
        const alreadyPresent = this.opponents.some((opp) => opp.kind === 'onnx' && opp.modelPath === modelPath);
        if (alreadyPresent) continue;
        discovered.push(this.buildOnnxOpponentFromFileName(fileName));
      }

      if (discovered.length > 0) {
        this.opponents.push(...discovered);
      }

      this.uiLog(`Discovered ${uniqueModelFiles.length} ONNX model file(s) in ./models.`);
    } catch (err) {
      this.uiLog(`Model directory scan failed: ${err.message}`, true);
    }
  }

  buildOnnxOpponentFromFileName(fileName) {
    const stem = fileName.replace(/\.onnx$/i, '');
    const id = `auto_${stem.replace(/[^a-zA-Z0-9]+/g, '_').toLowerCase()}`;
    const pretty = stem
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .trim();

    return {
      id,
      label: `${pretty} vs Atari AI`,
      kind: 'onnx',
      modelPath: `./models/${fileName}`,
      inferenceMode: 'argmax',
      temperature: 1.0,
    };
  }

  normalizeOpponentInferenceConfig(opponent) {
    if (!opponent || opponent.kind !== 'onnx') return;
    if (opponent.inferenceMode !== 'sample' && opponent.inferenceMode !== 'argmax') {
      opponent.inferenceMode = 'argmax';
    }
    const temp = Number(opponent.temperature);
    opponent.temperature = Number.isFinite(temp) ? Math.min(2.0, Math.max(0.2, temp)) : 1.0;
  }

  refreshInferenceControls() {
    const modeSelect = document.getElementById('inferenceModeSelect');
    const tempRange = document.getElementById('inferenceTempRange');
    const tempValue = document.getElementById('inferenceTempValue');
    if (!modeSelect || !tempRange || !tempValue) return;

    const onnxSelected = this.selectedOpponent && this.selectedOpponent.kind === 'onnx';
    modeSelect.disabled = !onnxSelected;
    tempRange.disabled = !onnxSelected;

    if (!onnxSelected) {
      modeSelect.value = 'argmax';
      tempRange.value = '1.0';
      tempValue.textContent = 'n/a';
      return;
    }

    this.normalizeOpponentInferenceConfig(this.selectedOpponent);
    const mode = this.selectedOpponent.inferenceMode || 'argmax';
    const temp = Number(this.selectedOpponent.temperature ?? 1.0).toFixed(1);
    modeSelect.value = mode;
    tempRange.value = temp;
    tempValue.textContent = temp;
  }

  applyInferenceControlsToSelectedOpponent() {
    if (!this.selectedOpponent || this.selectedOpponent.kind !== 'onnx') return;

    const modeSelect = document.getElementById('inferenceModeSelect');
    const tempRange = document.getElementById('inferenceTempRange');
    const tempValue = document.getElementById('inferenceTempValue');
    if (!modeSelect || !tempRange || !tempValue) return;

    this.selectedOpponent.inferenceMode = modeSelect.value === 'sample' ? 'sample' : 'argmax';
    const t = Number(tempRange.value);
    this.selectedOpponent.temperature = Number.isFinite(t) ? Math.min(2.0, Math.max(0.2, t)) : 1.0;
    tempValue.textContent = this.selectedOpponent.temperature.toFixed(1);

    this.updateStatusLine();
  }

  async loadLocalAleBundle() {
    if (typeof globalThis.createALEModule === 'function') {
      return globalThis.createALEModule;
    }

    await new Promise((resolve, reject) => {
      const existing = document.querySelector('script[data-ale-wasm-local="true"]');
      if (existing) {
        if (typeof globalThis.createALEModule === 'function') {
          resolve();
          return;
        }
        existing.addEventListener('load', () => resolve(), { once: true });
        existing.addEventListener('error', () => reject(new Error('ALE WASM local script failed to load')), { once: true });
        return;
      }

      const script = document.createElement('script');
      script.src = ALE_WASM_LOCAL;
      script.async = true;
      script.dataset.aleWasmLocal = 'true';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Failed to load ${ALE_WASM_LOCAL}`));
      document.head.appendChild(script);
    });

    if (typeof globalThis.createALEModule !== 'function') {
      throw new Error('Local ALE WASM bundle loaded, but createALEModule was not defined');
    }

    return globalThis.createALEModule;
  }

  populateOpponentSelect() {
    const select = document.getElementById('opponentSelect');
    select.innerHTML = '';
    for (const opp of this.opponents) {
      const option = document.createElement('option');
      option.value = opp.id;
      option.textContent = opp.label;
      select.appendChild(option);
    }
    select.value = this.selectedOpponent.id;
  }

  wireControls() {
    const startBtn = document.getElementById('startMatchBtn');
    const stopBtn = document.getElementById('stopMatchBtn');
    const clearBtn = document.getElementById('clearLeaderboardBtn');
    const opponentSelect = document.getElementById('opponentSelect');
    const playerInput = document.getElementById('playerNameInput');
    const inferenceModeSelect = document.getElementById('inferenceModeSelect');
    const inferenceTempRange = document.getElementById('inferenceTempRange');

    opponentSelect.addEventListener('change', async () => {
      const selected = this.opponents.find((o) => o.id === opponentSelect.value) || this.opponents[0];
      this.selectedOpponent = selected;
      this.normalizeOpponentInferenceConfig(this.selectedOpponent);
      this.refreshInferenceControls();
      if (selected.kind === 'onnx') {
        await this.ensureModelLoaded(selected.modelPath);
      }
      this.updateStatusLine();
    });

    inferenceModeSelect.addEventListener('change', () => {
      this.applyInferenceControlsToSelectedOpponent();
    });

    inferenceTempRange.addEventListener('input', () => {
      this.applyInferenceControlsToSelectedOpponent();
    });

    playerInput.addEventListener('change', () => {
      this.playerName = playerInput.value.trim() || 'Player1';
    });

    startBtn.addEventListener('click', async () => {
      this.playerName = playerInput.value.trim() || 'Player1';
      await this.startMatch();
    });

    stopBtn.addEventListener('click', () => {
      this.running = false;
      this.updateStatusLine('Match stopped.');
    });

    clearBtn.addEventListener('click', () => {
      localStorage.setItem(LEADERBOARD_KEY, JSON.stringify([]));
      this.renderLeaderboard();
      this.uiLog('Leaderboard cleared for this browser profile.');
    });

    this.refreshInferenceControls();
  }

  async loadRomFromFile(file) {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      await this.loadRomWithCompatibility({
        bytes,
        fileName: file.name || 'rom.bin',
      });
      this.onRomLoaded(file.name);
    } catch (err) {
      this.uiLog(`ROM load failed from file: ${err.message}`, true);
    }
  }

  async loadRomFromUrl(url) {
    const fileName = url.split('/').pop() || 'rom.bin';
    await this.loadRomWithCompatibility({
      url,
      fileName,
    });
    this.onRomLoaded(url);
  }

  async loadBundledRom(url) {
    await this.loadRomFromUrl(url);
  }

  async loadRomWithCompatibility({ url, bytes, fileName }) {
    if (url && typeof this.ale.loadROMFromURL === 'function') {
      await this.ale.loadROMFromURL(url);
      return;
    }

    let romBytes = bytes;
    if (!romBytes && url) {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} fetching ROM`);
      }
      romBytes = new Uint8Array(await response.arrayBuffer());
    }

    if (!romBytes) {
      throw new Error('No ROM bytes available to load');
    }

    if (typeof this.ale.loadROMFromFile === 'function') {
      const file = new File([romBytes], fileName || 'rom.bin', { type: 'application/octet-stream' });
      await this.ale.loadROMFromFile(file);
      return;
    }

    if (!this.ALE?.FS || typeof this.ale.loadROM !== 'function') {
      throw new Error('This ALE build does not expose loadROMFromFile and FS/loadROM fallback is unavailable');
    }

    const romPath = `/roms/${fileName || 'rom.bin'}`;
    try {
      this.ALE.FS.mkdir('/roms');
    } catch (_) {
      // Directory may already exist.
    }
    this.ALE.FS.writeFile(romPath, romBytes);
    await Promise.resolve(this.ale.loadROM(romPath));
  }

  onRomLoaded(source) {
    this.legalActions = this.ale.getLegalActionSet();
    this.minimalActions = this.ale.getMinimalActionSet();
    this.resolveSb3ActionMap();
    this.resolveControlActions();
    this.calibrateControlActions();
    this.romLoaded = true;
    this.uiLog(`ROM loaded: ${source}`);
    this.uiLog(`Legal actions: [${this.legalActions.join(', ')}]`);
    this.uiLog(`Minimal actions: [${this.minimalActions.join(', ')}]`);
    this.uiLog(`SB3 index->ALE map: [${this.sb3ActionMap.join(', ')}]`);
    this.uiLog(
      `Control map => SIDE:${this.controlSide || 'unknown'} UP:${this.controlActions.up} DOWN:${this.controlActions.down} FIRE:${this.controlActions.fire} NOOP:${this.controlActions.noop}`
    );
    this.updateStatusLine('ROM loaded. Ready to start match.');
  }

  resolveSb3ActionMap() {
    const defaultMap = [0, 1, 3, 4, 11, 12];

    // SB3 Atari Pong is trained over a 6-action space where model outputs are indices.
    // We must map index -> ALE action id using the ROM's minimal action set order.
    if (Array.isArray(this.minimalActions) && this.minimalActions.length >= 6) {
      this.sb3ActionMap = this.minimalActions.slice(0, 6);
      return;
    }

    this.sb3ActionMap = defaultMap;
  }

  mapSb3IndexToAleAction(index) {
    if (!Array.isArray(this.sb3ActionMap) || this.sb3ActionMap.length === 0) {
      return this.pickLegal([index]);
    }

    if (index >= 0 && index < this.sb3ActionMap.length) {
      return this.sb3ActionMap[index];
    }

    return this.pickLegal([index]);
  }

  softmax(logits, temperature = 1.0) {
    const t = Math.max(1e-6, Number(temperature) || 1.0);
    const maxLogit = Math.max(...logits);
    const exps = logits.map((v) => Math.exp((v - maxLogit) / t));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    if (!Number.isFinite(sumExp) || sumExp <= 0) {
      const uniform = 1 / Math.max(1, logits.length);
      return logits.map(() => uniform);
    }
    return exps.map((v) => v / sumExp);
  }

  sampleFromProbs(probs) {
    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return Math.max(0, probs.length - 1);
  }

  chooseActionIndexFromLogits(logits, opponent) {
    const mode = (opponent && opponent.inferenceMode) || 'argmax';
    if (mode === 'sample') {
      const temperature = opponent && opponent.temperature != null ? opponent.temperature : 1.0;
      const probs = this.softmax(logits, temperature);
      return this.sampleFromProbs(probs);
    }

    let argmax = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > logits[argmax]) argmax = i;
    }
    return argmax;
  }

  resolveControlActions() {
    const has = (a) => Array.isArray(this.legalActions) && this.legalActions.includes(a);
    const inMinimal = (a) => Array.isArray(this.minimalActions) && this.minimalActions.includes(a);

    const pick = (candidates, fallback = 0) => {
      for (const c of candidates) {
        if (has(c)) return c;
      }
      return fallback;
    };

    // Pong in ALE often uses left/right(+fire) actions for paddle motion.
    // Prefer minimal-action-compatible controls first, then canonical fallbacks.
    const up = pick(
      [
        ...(inMinimal(4) ? [4] : []),
        ...(inMinimal(12) ? [12] : []),
        ...(inMinimal(2) ? [2] : []),
        ...(inMinimal(10) ? [10] : []),
        4,
        12,
        2,
        10,
      ],
      0
    );
    const down = pick(
      [
        ...(inMinimal(3) ? [3] : []),
        ...(inMinimal(11) ? [11] : []),
        ...(inMinimal(5) ? [5] : []),
        ...(inMinimal(13) ? [13] : []),
        3,
        11,
        5,
        13,
      ],
      0
    );
    const fire = pick([1, 10, 13, 11, 12], 0);
    const noop = pick([0, 1], 0);

    this.controlActions = { up, down, fire, noop };
    this.controlSide = 'right';
  }

  estimatePaddleYFromGray(gray, width, height, side) {
    const xStart = side === 'left' ? 0 : Math.max(0, width - 16);
    const xEnd = side === 'left' ? Math.min(width, 16) : width;
    const ys = [];

    for (let y = 24; y < height; y++) {
      for (let x = xStart; x < xEnd; x++) {
        const v = gray[y * width + x];
        if (v > 120) ys.push(y);
      }
    }

    if (ys.length === 0) return null;
    ys.sort((a, b) => a - b);
    return ys[Math.floor(ys.length / 2)];
  }

  calibrateControlActions() {
    if (typeof this.ale.saveState !== 'function' || typeof this.ale.loadState !== 'function') {
      this.uiLog('Control calibration skipped: ALE save/load state is unavailable.', true);
      return;
    }

    try {
      this.ale.resetGame();
      const baselineState = new Uint8Array(this.ale.saveState());
      const width = this.ale.getScreenWidth();
      const height = this.ale.getScreenHeight();
      const baselineGray = this.ale.getScreenGrayscale();
      const baseLeft = this.estimatePaddleYFromGray(baselineGray, width, height, 'left');
      const baseRight = this.estimatePaddleYFromGray(baselineGray, width, height, 'right');

      if (baseLeft == null || baseRight == null) {
        this.uiLog('Control calibration skipped: could not detect paddles.', true);
        this.ale.loadState(baselineState);
        return;
      }

      const profiles = [];
      for (const action of this.legalActions || []) {
        this.ale.loadState(baselineState);
        for (let i = 0; i < 8; i++) this.ale.act(action);

        const gray = this.ale.getScreenGrayscale();
        const leftY = this.estimatePaddleYFromGray(gray, width, height, 'left');
        const rightY = this.estimatePaddleYFromGray(gray, width, height, 'right');
        if (leftY == null || rightY == null) continue;

        profiles.push({
          action,
          dLeft: leftY - baseLeft,
          dRight: rightY - baseRight,
        });
      }

      this.ale.loadState(baselineState);

      const solveForSide = (deltaKey) => {
        const moved = profiles
          .filter((p) => Math.abs(p[deltaKey]) >= 2)
          .sort((a, b) => Math.abs(b[deltaKey]) - Math.abs(a[deltaKey]));
        if (moved.length < 2) return null;

        const up = moved.find((p) => p[deltaKey] < 0);
        const down = moved.find((p) => p[deltaKey] > 0);
        if (!up || !down) return null;

        return {
          up: up.action,
          down: down.action,
          strength: Math.abs(up[deltaKey]) + Math.abs(down[deltaKey]),
        };
      };

      const leftSolve = solveForSide('dLeft');
      const rightSolve = solveForSide('dRight');

      let chosen = null;
      let side = 'right';
      if (leftSolve && rightSolve) {
        if (leftSolve.strength > rightSolve.strength) {
          chosen = leftSolve;
          side = 'left';
        } else {
          chosen = rightSolve;
          side = 'right';
        }
      } else if (rightSolve) {
        chosen = rightSolve;
        side = 'right';
      } else if (leftSolve) {
        chosen = leftSolve;
        side = 'left';
      }

      if (chosen) {
        this.controlActions.up = chosen.up;
        this.controlActions.down = chosen.down;
        this.controlSide = side;
        this.uiLog(`Calibrated controls on ${side} paddle: UP=${chosen.up}, DOWN=${chosen.down}`);
      } else {
        this.uiLog('Calibration found no clear movement actions; using fallback mapping.', true);
      }
    } catch (err) {
      this.uiLog(`Control calibration error: ${err.message}`, true);
    }
  }

  async warmupServe() {
    if (!this.controlActions) return;
    const fire = this.controlActions.fire;
    if (fire === 0 && !this.legalActions.includes(0)) return;

    // Kick the episode out of any waiting-to-serve state.
    for (let i = 0; i < 8; i++) {
      this.ale.act(fire);
    }
  }

  async ensureModelLoaded(path) {
    if (this.modelPath === path && this.aiSession) return;
    const status = document.getElementById('engineStatus');
    status.textContent = `Engine: Loading model ${path}...`;

    try {
      // Fast path for single-file ONNX models.
      this.aiSession = await ort.InferenceSession.create(path);
    } catch (err) {
      const msg = String(err?.message || err || '');
      const needsExternalData = /external data|MountedFiles|Deserialize tensor/i.test(msg);
      if (!needsExternalData) {
        throw err;
      }

      this.uiLog('Model appears to use external tensor data; loading sidecar .onnx.data file...');
      this.aiSession = await this.loadModelWithExternalData(path);
    }

    this.modelPath = path;
    this.uiLog(`Model ready: ${path}`);
  }

  async loadModelWithExternalData(path) {
    const modelResp = await fetch(path, { cache: 'no-store' });
    if (!modelResp.ok) {
      throw new Error(`Could not fetch model: HTTP ${modelResp.status}`);
    }
    const modelBytes = new Uint8Array(await modelResp.arrayBuffer());

    const sidecarPath = `${path}.data`;
    const sidecarResp = await fetch(sidecarPath, { cache: 'no-store' });
    if (!sidecarResp.ok) {
      throw new Error(`Could not fetch external data: ${sidecarPath} (HTTP ${sidecarResp.status})`);
    }
    const sidecarBytes = new Uint8Array(await sidecarResp.arrayBuffer());

    const modelFile = path.split('/').pop() || 'model.onnx';
    const externalFile = `${modelFile}.data`;

    // Some exports store the location token with literal quotes in the ONNX graph,
    // so we provide both variants to maximize compatibility.
    return await ort.InferenceSession.create(modelBytes, {
      externalData: [
        { path: externalFile, data: sidecarBytes },
        { path: `"${externalFile}"`, data: sidecarBytes },
      ],
    });
  }

  updateStatusLine(extra = '') {
    const status = document.getElementById('engineStatus');
    let base = `Engine: ALE WASM | Opponent: ${this.selectedOpponent.label}`;
    if (this.selectedOpponent && this.selectedOpponent.kind === 'onnx') {
      const mode = this.selectedOpponent.inferenceMode || 'argmax';
      const temp = Number(this.selectedOpponent.temperature ?? 1.0).toFixed(1);
      base += ` | Inference: ${mode} (T=${temp})`;
    }
    status.textContent = extra ? `${base} | ${extra}` : base;
  }

  resetActionTrend() {
    this.actionTrend = [];
    this.trendCycleStartSec = 0;
    // Start first bin immediately so we don't show an empty 0-1s region.
    this.trendBinStartMs = Date.now() - this.trendIntervalMs;
    this.trendBinCounts = { dominant: 0, total: 0 };
    this.lastChosenModelIndex = null;

    const now = Date.now();
    this.snapshotNextCaptureAtMs = now;
    this.snapshotCaptureStartMs = now;
    this.snapshotCaptureEvents = [];
    this.snapshotSeries = [];
    this.snapshotTickSeries = [];
    this.lastSnapshotCapturedAtMs = 0;
    this.lastDecisionNonDominant = false;
    this.lastDecisionIntent = null;
  }

  updatePeriodicSnapshot(intent, nonDominantTick) {
    if (!this.running) return;

    const now = Date.now();
    if (!this.snapshotCaptureStartMs && now >= this.snapshotNextCaptureAtMs) {
      this.snapshotCaptureStartMs = now;
      this.snapshotCaptureEvents = [];
    }

    if (!this.snapshotCaptureStartMs) return;

    const dt = now - this.snapshotCaptureStartMs;
    if (dt <= this.snapshotDurationMs) {
      this.snapshotCaptureEvents.push({ dt, intent, nonDominantTick: !!nonDominantTick });
      return;
    }

    const binCount = Math.max(1, Math.floor(this.snapshotDurationMs / this.snapshotBinMs));
    const bins = Array.from({ length: binCount }, () => ({ up: 0, down: 0, noop: 0, total: 0 }));

    for (const ev of this.snapshotCaptureEvents) {
      const idx = Math.min(binCount - 1, Math.max(0, Math.floor(ev.dt / this.snapshotBinMs)));
      bins[idx].total += 1;
      if (ev.intent === 'up') bins[idx].up += 1;
      else if (ev.intent === 'down') bins[idx].down += 1;
      else bins[idx].noop += 1;
    }

    this.snapshotSeries = bins.map((b, idx) => {
      const t = (idx + 0.5) * (this.snapshotBinMs / 1000);
      if (b.total === 0) return { t, upPct: 0, downPct: 0, noopPct: 0 };
      return {
        t,
        upPct: (b.up / b.total) * 100,
        downPct: (b.down / b.total) * 100,
        noopPct: (b.noop / b.total) * 100,
      };
    });

    this.snapshotTickSeries = this.snapshotCaptureEvents
      .filter((ev) => ev.nonDominantTick)
      .map((ev) => ({ t: Math.max(0, Math.min(1.0, ev.dt / 1000)), intent: ev.intent }));

    this.lastSnapshotCapturedAtMs = now;
    this.snapshotCaptureStartMs = null;
    this.snapshotCaptureEvents = [];
    this.snapshotNextCaptureAtMs = now + this.snapshotPeriodMs;
  }

  classifyActionIntent(actionId) {
    const controls = this.controlActions || {};
    const upCandidates = [controls.up, 2, 4, 10, 12].filter((v) => Number.isInteger(v));
    const downCandidates = [controls.down, 3, 5, 11, 13].filter((v) => Number.isInteger(v));

    if (upCandidates.includes(actionId)) return 'up';
    if (downCandidates.includes(actionId)) return 'down';
    return 'noop';
  }

  updateDominanceTrend(selectedIndex, dominantIndex) {
    if (!this.running) return;

    const now = Date.now();
    if (!this.trendBinStartMs) this.trendBinStartMs = now;

    this.trendBinCounts.total += 1;
    if (selectedIndex === dominantIndex) this.trendBinCounts.dominant += 1;

    if (now - this.trendBinStartMs < this.trendIntervalMs) {
      return;
    }

    const total = this.trendBinCounts.total;
    const dominantPct = total > 0 ? (this.trendBinCounts.dominant / total) * 100 : 100;
    const elapsedSec = this.matchStart ? (now - this.matchStart) / 1000 : 0;

    if (this.trendCycleStartSec === 0) {
      this.trendCycleStartSec = elapsedSec;
    }

    if (elapsedSec - this.trendCycleStartSec >= this.trendWindowSec) {
      this.actionTrend = [];
      this.trendCycleStartSec = elapsedSec;
    }

    this.actionTrend.push({ t: elapsedSec, dominantPct });
    if (this.actionTrend.length > this.trendMaxPoints) {
      this.actionTrend.shift();
    }

    this.trendBinCounts = { dominant: 0, total: 0 };
    this.trendBinStartMs = now;
  }

  renderActionTrend() {
    if (!this.actionTrendCtx || !this.actionTrendCanvas) return;

    const ctx = this.actionTrendCtx;
    const canvas = this.actionTrendCanvas;
    const w = canvas.width;
    const h = canvas.height;
    const padL = 42;
    const padR = 14;
    const padT = 12;
    const padB = 28;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#020205';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(0,255,204,0.24)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    const yTicks = [0, 25, 50, 75, 100];
    ctx.fillStyle = 'rgba(200, 255, 248, 0.75)';
    ctx.font = '11px Courier New';
    yTicks.forEach((pct) => {
      const y = padT + plotH - (pct / 100) * plotH;
      ctx.strokeStyle = 'rgba(0,255,204,0.12)';
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillText(`${pct}%`, 4, y + 4);
    });

    const windowStart = this.trendCycleStartSec;
    const windowEnd = windowStart + this.trendWindowSec;
    const span = Math.max(1e-6, windowEnd - windowStart);
    const xFor = (t) => padL + ((t - windowStart) / span) * plotW;
    const yFor = (pct) => padT + plotH - (pct / 100) * plotH;

    const xTicks = [0, 5, 10, 15, 20, 25, 30];
    ctx.fillStyle = 'rgba(200, 255, 248, 0.7)';
    xTicks.forEach((s) => {
      const x = padL + (s / this.trendWindowSec) * plotW;
      ctx.strokeStyle = 'rgba(0,255,204,0.10)';
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
      ctx.fillText(`${s}s`, x - 8, h - 8);
    });

    if (this.actionTrend.length >= 1) {
      const visibleTrend = this.actionTrend.filter((p) => p.t >= windowStart && p.t <= windowEnd);

      ctx.strokeStyle = '#ffcc44';
      ctx.lineWidth = 2;
      ctx.beginPath();
      visibleTrend.forEach((p, idx) => {
        const x = xFor(p.t);
        const y = yFor(p.dominantPct);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = 'rgba(200, 255, 248, 0.75)';
      ctx.fillText('Fixed 30s cycle (auto-reset)', w - 220, h - 8);

      const lastPoint = visibleTrend[visibleTrend.length - 1];
      if (lastPoint) {
        const nonDominantPct = Math.max(0, 100 - lastPoint.dominantPct);
        ctx.fillText(
          `Dominant: ${lastPoint.dominantPct.toFixed(1)}% | Non-dominant: ${nonDominantPct.toFixed(1)}%`,
          padL + 280,
          12
        );
      }
    }

    ctx.fillStyle = '#ffcc44';
    ctx.fillRect(padL + 8, 6, 10, 3);
    ctx.fillStyle = 'rgba(200, 255, 248, 0.85)';
    ctx.fillText('Dominant-choice % per second', padL + 22, 12);

    ctx.fillStyle = 'rgba(200, 255, 248, 0.66)';
    ctx.fillText('100% = always chose argmax, lower = more non-dominant picks', w - 430, 12);
  }

  renderActionSnapshot() {
    if (!this.actionSnapshotCtx || !this.actionSnapshotCanvas) return;

    const ctx = this.actionSnapshotCtx;
    const canvas = this.actionSnapshotCanvas;
    const w = canvas.width;
    const h = canvas.height;
    const padL = 42;
    const padR = 14;
    const padT = 12;
    const padB = 28;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#020205';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(0,255,204,0.24)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    const yTicks = [0, 25, 50, 75, 100];
    ctx.fillStyle = 'rgba(200, 255, 248, 0.75)';
    ctx.font = '11px Courier New';
    yTicks.forEach((pct) => {
      const y = padT + plotH - (pct / 100) * plotH;
      ctx.strokeStyle = 'rgba(0,255,204,0.12)';
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillText(`${pct}%`, 4, y + 4);
    });

    const xTicks = [0, 0.25, 0.5, 0.75, 1.0];
    xTicks.forEach((s) => {
      const x = padL + s * plotW;
      ctx.strokeStyle = 'rgba(0,255,204,0.10)';
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
      ctx.fillStyle = 'rgba(200, 255, 248, 0.7)';
      ctx.fillText(`${s.toFixed(2)}s`, x - 14, h - 8);
    });

    if (this.snapshotSeries.length > 0) {
      const xFor = (t) => padL + (t / 1.0) * plotW;
      const yFor = (pct) => padT + plotH - (pct / 100) * plotH;

      const drawLine = (key, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        this.snapshotSeries.forEach((p, idx) => {
          const x = xFor(Math.min(1.0, p.t));
          const y = yFor(p[key]);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      };

      drawLine('upPct', '#00ff88');
      drawLine('downPct', '#ff5f9a');
      drawLine('noopPct', '#8aa0b8');

      for (const tick of this.snapshotTickSeries) {
        const x = xFor(tick.t);
        let color = '#8aa0b8';
        if (tick.intent === 'up') color = '#00ff88';
        if (tick.intent === 'down') color = '#ff5f9a';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, h - padB + 2);
        ctx.lineTo(x, h - padB + 10);
        ctx.stroke();
      }
    }

    ctx.fillStyle = '#00ff88';
    ctx.fillRect(padL + 8, 6, 10, 3);
    ctx.fillStyle = 'rgba(200, 255, 248, 0.85)';
    ctx.fillText('UP %', padL + 22, 12);

    ctx.fillStyle = '#ff5f9a';
    ctx.fillRect(padL + 78, 6, 10, 3);
    ctx.fillStyle = 'rgba(200, 255, 248, 0.85)';
    ctx.fillText('DOWN %', padL + 92, 12);

    ctx.fillStyle = '#8aa0b8';
    ctx.fillRect(padL + 162, 6, 10, 3);
    ctx.fillStyle = 'rgba(200, 255, 248, 0.85)';
    ctx.fillText('NOOP %', padL + 176, 12);

    ctx.fillStyle = 'rgba(200, 255, 248, 0.66)';
    ctx.fillText('Bottom ticks: non-dominant sampled decisions', padL + 250, 12);

    const now = Date.now();
    let statusText = 'Waiting for first 1s capture...';
    if (this.snapshotCaptureStartMs) {
      const remaining = Math.max(0, this.snapshotDurationMs - (now - this.snapshotCaptureStartMs));
      statusText = `Capturing now (${(remaining / 1000).toFixed(2)}s left)`;
    } else if (this.lastSnapshotCapturedAtMs > 0) {
      const until = Math.max(0, this.snapshotNextCaptureAtMs - now);
      statusText = `Last capture complete | Next in ${(until / 1000).toFixed(1)}s`;
    } else {
      const untilFirst = Math.max(0, this.snapshotNextCaptureAtMs - now);
      statusText = `First capture in ${(untilFirst / 1000).toFixed(1)}s`;
    }
    ctx.fillStyle = 'rgba(200, 255, 248, 0.66)';
    ctx.fillText(statusText, w - 330, 12);
  }

  resetEpisodeState() {
    this.frameStack = [];
    this.rawFramePair = [];
    this.currentAIAction = this.pickLegal([0, 1]);
    this.inferenceBusy = false;
    this.frameCounter = 0;
    this.rewardSum = 0;
    this.matchStart = Date.now();
    this.resetActionTrend();
  }

  async startMatch() {
    if (!this.romLoaded) {
      this.uiLog('Load a Pong ROM first (file or URL).', true);
      return;
    }

    if (this.selectedOpponent.kind === 'onnx') {
      try {
        await this.ensureModelLoaded(this.selectedOpponent.modelPath);
      } catch (err) {
        this.uiLog(`Model load failed: ${err.message}`, true);
        return;
      }
    }

    this.resetEpisodeState();
    this.ale.resetGame();
    await this.warmupServe();
    this.invertControls = false;
    this.lastActionWasMove = false;
    this.autoInvertChecked = false;
    this.running = true;
    this.canvas.focus();
    this.updateStatusLine('Match running');
    this.uiLog(`Match started: ${this.playerName} vs ${this.selectedOpponent.label}`);
  }

  pickLegal(candidates) {
    if (!this.legalActions || this.legalActions.length === 0) return 0;
    for (const c of candidates) {
      if (this.legalActions.includes(c)) return c;
    }
    if (this.minimalActions && this.minimalActions.length > 0) {
      return this.minimalActions[0];
    }
    return this.legalActions[0];
  }

  getHumanAction() {
    const up = !!(this.keys['ArrowUp'] || this.keys['w']);
    const down = !!(this.keys['ArrowDown'] || this.keys['s']);

    const controls = this.controlActions || {
      up: this.pickLegal([2, 4, 10, 14]),
      down: this.pickLegal([5, 3, 13, 17]),
      fire: this.pickLegal([1, 10, 13, 11, 12]),
      noop: this.pickLegal([0, 1]),
    };

    const upAction = this.invertControls ? controls.down : controls.up;
    const downAction = this.invertControls ? controls.up : controls.down;

    if (up && !down) return upAction;
    if (down && !up) return downAction;
    return controls.noop;
  }

  captureRawGrayscaleFrame() {
    const width = this.ale.getScreenWidth();
    const height = this.ale.getScreenHeight();
    const gray = this.ale.getScreenGrayscale();

    // Keep only the latest two raw frames for max-pooling.
    this.rawFramePair.push(new Uint8ClampedArray(gray));
    if (this.rawFramePair.length > 2) this.rawFramePair.shift();

    return { width, height };
  }

  commitDecisionFrame(width, height) {
    if (!this.rawFramePair.length) return;

    let pooled;
    if (this.rawFramePair.length === 1) {
      pooled = this.rawFramePair[0];
    } else {
      const a = this.rawFramePair[0];
      const b = this.rawFramePair[1];
      pooled = new Uint8ClampedArray(a.length);
      for (let i = 0; i < a.length; i++) {
        pooled[i] = a[i] > b[i] ? a[i] : b[i];
      }
    }

    if (this.rawCanvas.width !== width || this.rawCanvas.height !== height) {
      this.rawCanvas.width = width;
      this.rawCanvas.height = height;
    }

    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < pooled.length; i++, j += 4) {
      const v = pooled[i];
      rgba[j] = v;
      rgba[j + 1] = v;
      rgba[j + 2] = v;
      rgba[j + 3] = 255;
    }

    this.rawCtx.putImageData(new ImageData(rgba, width, height), 0, 0);
    this.preCtx.drawImage(this.rawCanvas, 0, 0, 84, 84);

    const img = this.preCtx.getImageData(0, 0, 84, 84).data;
    const frame = new Float32Array(84 * 84);
    for (let i = 0, p = 0; i < img.length; i += 4, p++) {
      frame[p] = img[i];
    }

    if (this.frameStack.length === 0) {
      for (let i = 0; i < 4; i++) this.frameStack.push(frame);
    } else {
      this.frameStack.shift();
      this.frameStack.push(frame);
    }

    // Start the next max-pool window with the latest raw frame only.
    this.rawFramePair = this.rawFramePair.slice(-1);
  }

  async maybeInferAction() {
    if (!this.aiSession || this.inferenceBusy || this.frameStack.length < 4) return;
    if (this.frameCounter % 4 !== 0) return;

    this.inferenceBusy = true;
    try {
      const input = new Float32Array(4 * 84 * 84);
      let offset = 0;
      for (let c = 0; c < 4; c++) {
        for (let i = 0; i < 84 * 84; i++) {
          input[offset++] = this.frameStack[c][i];
        }
      }

      const results = await this.aiSession.run({ input: new ort.Tensor('float32', input, [1, 4, 84, 84]) });
      const logits = results.output.data;

      let dominantIndex = 0;
      for (let i = 1; i < logits.length; i++) {
        if (logits[i] > logits[dominantIndex]) dominantIndex = i;
      }

      const selectedIndex = this.chooseActionIndexFromLogits(logits, this.selectedOpponent);
      this.lastChosenModelIndex = selectedIndex;

      const selectedAleAction = this.mapSb3IndexToAleAction(selectedIndex);
      const dominantAleAction = this.mapSb3IndexToAleAction(dominantIndex);
      this.currentAIAction = selectedAleAction;

      const selectedIntent = this.classifyActionIntent(selectedAleAction);
      const dominantIntent = this.classifyActionIntent(dominantAleAction);
      const mode = (this.selectedOpponent && this.selectedOpponent.inferenceMode) || 'argmax';
      this.lastDecisionNonDominant = mode === 'sample' && selectedIntent !== dominantIntent;
      this.lastDecisionIntent = selectedIntent;
      this.updateDominanceTrend(selectedIndex, dominantIndex);
    } catch (err) {
      this.uiLog(`Inference error: ${err.message}`, true);
      this.lastDecisionNonDominant = false;
      this.lastDecisionIntent = null;
    } finally {
      this.inferenceBusy = false;
    }
  }

  renderFrame() {
    const frame = this.ale.getScreenImageData();
    if (this.nativeCanvas.width !== frame.width || this.nativeCanvas.height !== frame.height) {
      this.nativeCanvas.width = frame.width;
      this.nativeCanvas.height = frame.height;
    }
    this.nativeCtx.putImageData(frame, 0, 0);
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(this.nativeCanvas, 0, 0, this.canvas.width, this.canvas.height);

    const metrics = document.getElementById('metricsContent');
    const humanAction = this.selectedOpponent.kind === 'human' ? this.getHumanAction() : this.currentAIAction;
    metrics.innerHTML = [
      `<strong>Mode:</strong> ${this.selectedOpponent.label}<br>`,
      `<strong>Reward Sum:</strong> ${this.rewardSum}<br>`,
      `<strong>Frame:</strong> ${this.frameCounter}<br>`,
      `<strong>Running:</strong> ${this.running ? 'Yes' : 'No'}<br>`,
      `<strong>Model Index:</strong> ${this.lastChosenModelIndex == null ? 'n/a' : this.lastChosenModelIndex}<br>`,
      `<strong>Action:</strong> ${humanAction}<br>`,
      `<strong>Intent:</strong> ${this.currentActionIntent.toUpperCase()}`,
    ].join('');
  }

  finishEpisode() {
    this.running = false;
    const durationSec = Math.round((Date.now() - this.matchStart) / 1000);

    const won = this.rewardSum > 0;
    const record = {
      player: this.playerName,
      opponent: this.selectedOpponent.label,
      wins: won ? 1 : 0,
      losses: won ? 0 : 1,
      reward: this.rewardSum,
      durationSec,
      at: new Date().toISOString(),
    };

    const all = this.getLeaderboard();
    all.push(record);
    localStorage.setItem(LEADERBOARD_KEY, JSON.stringify(all));
    this.renderLeaderboard();

    this.uiLog(`Episode complete: ${won ? 'WIN' : 'LOSS'} | reward=${this.rewardSum}`);
    this.updateStatusLine('Episode complete');
  }

  getLeaderboard() {
    try {
      const raw = localStorage.getItem(LEADERBOARD_KEY);
      const data = raw ? JSON.parse(raw) : [];
      return Array.isArray(data) ? data : [];
    } catch {
      return [];
    }
  }

  renderLeaderboard() {
    const tbody = document.getElementById('leaderboardBody');
    const rows = this.getLeaderboard();

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;opacity:0.7">No matches yet.</td></tr>';
      return;
    }

    const agg = {};
    for (const r of rows) {
      const key = `${r.player}__${r.opponent}`;
      if (!agg[key]) {
        agg[key] = { player: r.player, opponent: r.opponent, wins: 0, losses: 0, games: 0, lastReward: 0 };
      }
      agg[key].wins += r.wins;
      agg[key].losses += r.losses;
      agg[key].games += 1;
      agg[key].lastReward = r.reward;
    }

    const sorted = Object.values(agg).sort((a, b) => {
      if (b.wins !== a.wins) return b.wins - a.wins;
      return a.losses - b.losses;
    });

    tbody.innerHTML = sorted
      .map((r) => {
        const rate = r.games ? Math.round((r.wins / r.games) * 100) : 0;
        return `<tr>
          <td>${r.player}</td>
          <td>${r.opponent}</td>
          <td>${r.wins}</td>
          <td>${r.losses}</td>
          <td>${rate}%</td>
          <td>${r.lastReward}</td>
        </tr>`;
      })
      .join('');
  }

  async stepGame() {
    if (!this.running) return;

    if (this.selectedOpponent.kind === 'onnx' && this.frameCounter % 4 === 0) {
      await this.maybeInferAction();
    }

    const action = this.selectedOpponent.kind === 'human' ? this.getHumanAction() : this.currentAIAction;
    if (this.selectedOpponent.kind === 'human') {
      const controls = this.controlActions || {};
      this.lastActionWasMove = action === controls.up || action === controls.down;
    }
    const reward = this.ale.act(action);
    this.currentActionIntent = this.classifyActionIntent(action);
    const isDecisionFrame = this.selectedOpponent.kind === 'onnx' && this.frameCounter % 4 === 0;
    const snapshotIntent = isDecisionFrame && this.lastDecisionIntent ? this.lastDecisionIntent : this.currentActionIntent;
    this.updatePeriodicSnapshot(snapshotIntent, isDecisionFrame && this.lastDecisionNonDominant);
    this.rewardSum += reward;

    const { width, height } = this.captureRawGrayscaleFrame();
    this.frameCounter += 1;

    // Match Atari wrapper cadence: one decision frame per 4 emulator steps.
    if (this.frameCounter % 4 === 0) {
      this.commitDecisionFrame(width, height);
    }

    // If movement keys are being sent but nothing looks different early in the match,
    // swap direction mapping once as a practical fallback for ROM flavor differences.
    if (
      this.selectedOpponent.kind === 'human' &&
      !this.autoInvertChecked &&
      this.frameCounter > 180 &&
      this.lastActionWasMove
    ) {
      this.invertControls = true;
      this.autoInvertChecked = true;
      this.uiLog('Auto-adjust: inverted human control mapping for this ROM variant.');
    }

    if (this.ale.gameOver()) {
      this.finishEpisode();
    }
  }

  async loop() {
    const now = performance.now();
    if (!this.lastStepAt) this.lastStepAt = now;

    while (now - this.lastStepAt >= this.stepMs) {
      this.lastStepAt += this.stepMs;
      await this.stepGame();
    }

    if (this.romLoaded) {
      this.renderFrame();
    }

    this.renderActionTrend();
    this.renderActionSnapshot();

    requestAnimationFrame(() => this.loop());
  }
}

window.addEventListener('DOMContentLoaded', async () => {
  const app = new ALECasualApp();
  try {
    await app.init();
  } catch (err) {
    const status = document.getElementById('engineStatus');
    status.textContent = 'Engine: ALE WASM init failed';
    status.style.color = '#ff4d78';
    console.error(err);
  }
});
