/* Real-time, in-browser inference with the actual trained ANC network.
 *
 * The 35 MB U-Net/LSTM runs via onnxruntime-web. The AudioContext runs at 22050 Hz
 * so taps/secondary/recording are all sample-exact. FIR convolution happens in
 * real time with ConvolverNodes (per-mic control filter -> sum/mean -> secondary
 * path = the ear estimate); a background loop re-runs the network every ~0.5 s and
 * swaps the ConvolverNode IRs -- exactly like the device updating its filters.
 *
 *   before (ANC off) = ear            after (ANC on) = ear - earEstimate
 *
 * Two input sources:
 *   'recording' - the real multi-channel glasses recording (in-distribution, ~12 dB)
 *   'sim'       - synthetic mics from the scene geometry (OUT-of-distribution): the
 *                 network was never trained on these, so cancellation may be weak.
 *                 Dragging the source live re-seeds the input and re-predicts filters.
 */
(function () {
  const g = (id) => document.getElementById(id);
  const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js';

  let A = null, session = null, recFile = null;     // recFile: cached recording channels
  let rec = null, source = 'recording', curGeom = '';
  let ctx = null, nodes = null, startTime = 0, running = false, makeup = 1;
  let ancOn = true, inferBusy = false, stopReq = false, drawRaf = 0;

  function setStatus(s) { const el = g('rmLiveStatus'); if (el) el.textContent = s; }

  async function loadOrt() {
    if (window.ort) return;
    await new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = ORT_URL; s.onload = res; s.onerror = () => rej(new Error('ort load failed'));
      document.head.appendChild(s);
    });
  }

  async function loadCore() {
    if (session) return;
    setStatus('loading runtime…');
    await loadOrt();
    try { ort.env.wasm.numThreads = 1; } catch (e) {}
    setStatus('loading model (~36 MB, first time only)…');
    A = await fetch('anc_assets.json').then((r) => r.json());
    session = await ort.InferenceSession.create('anc_model.onnx', { executionProviders: ['wasm'] });
  }

  async function loadRecording() {
    if (recFile) return;
    setStatus('loading recording…');
    const buf = await fetch('anc_recording.f32').then((r) => r.arrayBuffer());
    const f = new Float32Array(buf), L = A.recording.length;
    recFile = { gl: [0, 1, 2, 3].map((i) => f.subarray(i * L, (i + 1) * L)), ear: f.subarray(4 * L, 5 * L), len: L };
  }

  function setMakeup() {
    let pk = 1e-4; for (let i = 0; i < rec.len; i++) { const a = Math.abs(rec.ear[i]); if (a > pk) pk = a; }
    makeup = 0.6 / pk;
  }

  async function buildRec() {
    if (source === 'recording') { await loadRecording(); rec = recFile; }
    else {
      if (!window.__ancSim) throw new Error('simulation hook missing');
      rec = window.__ancSim.genMics(6, A.orig_sr);          // 6 s loop @22050
      curGeom = window.__ancSim.geomKey();
    }
    setMakeup();
  }

  // 22050 -> 8820 resample, mirroring torchaudio's polyphase kernel (cross-correlation)
  function resample(x) {
    const UP = A.UP, DOWN = A.DOWN, W = A.resample_width, K = A.resample_kernel;
    const pad = W, plen = x.length + pad + (pad + DOWN);
    const xp = new Float32Array(plen); xp.set(x, pad);
    const nOut = ((plen - W) / DOWN | 0) + 1;
    const out = new Float32Array(nOut * UP);
    for (let i = 0; i < nOut; i++) {
      const p = i * DOWN;
      for (let ph = 0; ph < UP; ph++) {
        const k = K[ph]; let acc = 0;
        for (let j = 0; j < W; j++) acc += xp[p + j] * k[j];
        out[i * UP + ph] = acc;
      }
    }
    return out.subarray(0, Math.ceil(UP * x.length / DOWN));
  }

  function window_(chan, start, len) {
    const L = rec.len, w = new Float32Array(len);
    for (let i = 0; i < len; i++) w[i] = chan[(start + i) % L];
    return w;
  }

  async function inferAt(startSample) {
    const est = A.est_samples, Td = A.Tdown, down = new Float32Array(4 * Td);
    for (let c = 0; c < 4; c++) {
      const d = resample(window_(rec.gl[c], startSample, est));
      down.set(d.subarray(0, Td), c * Td);
    }
    const out = await session.run({ mics_down: new ort.Tensor('float32', down, [1, 4, Td]) });
    return out.taps.data; // Float32, 4 * out_len
  }

  function tapsBuffer(taps, c) {
    const N = A.out_len, b = ctx.createBuffer(1, N, A.orig_sr), d = b.getChannelData(0);
    for (let i = 0; i < N; i++) d[i] = taps[c * N + i];
    return b;
  }
  function irBuffer(arr) {
    const b = ctx.createBuffer(1, arr.length, A.orig_sr), d = b.getChannelData(0);
    for (let i = 0; i < arr.length; i++) d[i] = arr[i];
    return b;
  }

  function makeSources() {
    const L = rec.len;
    const gbuf = ctx.createBuffer(4, L, A.orig_sr);
    for (let c = 0; c < 4; c++) gbuf.getChannelData(c).set(rec.gl[c]);
    const gsrc = ctx.createBufferSource(); gsrc.buffer = gbuf; gsrc.loop = true;
    gsrc.connect(nodes.split);
    const ebuf = ctx.createBuffer(1, L, A.orig_sr); ebuf.getChannelData(0).set(rec.ear);
    const esrc = ctx.createBufferSource(); esrc.buffer = ebuf; esrc.loop = true;
    esrc.connect(nodes.eB); esrc.connect(nodes.eR);
    const t0 = ctx.currentTime + 0.08; gsrc.start(t0); esrc.start(t0); startTime = t0;
    nodes.gsrc = gsrc; nodes.esrc = esrc;
  }

  function buildGraph() {
    ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: A.orig_sr });
    const split = ctx.createChannelSplitter(4);
    const antiSum = ctx.createGain();
    // Two parallel convolver banks. Hard-swapping a live ConvolverNode.buffer resets its
    // state and clicks; instead we load the new IRs into the idle bank and crossfade to it,
    // so each ~0.5 s filter update is glitch-free.
    const banks = [[], []];
    const bankGain = [ctx.createGain(), ctx.createGain()];
    bankGain[0].gain.value = 0; bankGain[1].gain.value = 0;   // first setFilters fades one in
    for (let bk = 0; bk < 2; bk++) {
      for (let c = 0; c < 4; c++) {
        const cv = ctx.createConvolver(); cv.normalize = false;
        cv.buffer = ctx.createBuffer(1, 1, A.orig_sr);
        split.connect(cv, c, 0); cv.connect(bankGain[bk]); banks[bk].push(cv);
      }
      bankGain[bk].connect(antiSum);
    }
    const meanG = ctx.createGain(); meanG.gain.value = 1 / A.num_mics; antiSum.connect(meanG);
    const secConv = ctx.createConvolver(); secConv.normalize = false;
    secConv.buffer = irBuffer(A.secondary); meanG.connect(secConv);

    const beforeBus = ctx.createGain(), residualBus = ctx.createGain();
    const eB = ctx.createGain(); eB.connect(beforeBus);
    const eR = ctx.createGain(); eR.connect(residualBus);
    const inv = ctx.createGain(); inv.gain.value = -1; secConv.connect(inv); inv.connect(residualBus);

    const gBefore = ctx.createGain(), gAfter = ctx.createGain();
    beforeBus.connect(gBefore); residualBus.connect(gAfter);
    const master = ctx.createGain(); master.gain.value = makeup;
    gBefore.connect(master); gAfter.connect(master); master.connect(ctx.destination);

    const anBefore = ctx.createAnalyser(); anBefore.fftSize = 8192; anBefore.smoothingTimeConstant = 0.8;
    const anAfter = ctx.createAnalyser(); anAfter.fftSize = 8192; anAfter.smoothingTimeConstant = 0.8;
    beforeBus.connect(anBefore); residualBus.connect(anAfter);

    nodes = { split, banks, bankGain, active: -1, eB, eR, gBefore, gAfter, anBefore, anAfter,
              fB: new Float32Array(anBefore.frequencyBinCount), fA: new Float32Array(anAfter.frequencyBinCount) };
    makeSources();
    applyAB();
  }

  function applyAB() {
    if (!nodes) return; const t = ctx.currentTime;
    nodes.gBefore.gain.setTargetAtTime(ancOn ? 0 : 1, t, 0.02);
    nodes.gAfter.gain.setTargetAtTime(ancOn ? 1 : 0, t, 0.02);
  }
  function setFilters(taps) {
    if (!nodes) return;
    const next = nodes.active === 0 ? 1 : 0;            // load IRs into the idle bank
    for (let c = 0; c < 4; c++) nodes.banks[next][c].buffer = tapsBuffer(taps, c);
    const t = ctx.currentTime, TC = 0.03;              // ~30 ms equal-ish crossfade
    nodes.bankGain[next].gain.setTargetAtTime(1, t, TC);
    if (nodes.active >= 0) nodes.bankGain[nodes.active].gain.setTargetAtTime(0, t, TC);
    nodes.active = next;
  }
  function playhead() { return Math.floor(((ctx.currentTime - startTime) * A.orig_sr)) % rec.len; }

  // sim: re-synthesize the mics for the new scene geometry without tearing down the graph
  function reseed() {
    if (!running || source !== 'sim') return;
    try { nodes.gsrc.stop(); nodes.esrc.stop(); } catch (e) {}
    rec = window.__ancSim.genMics(6, A.orig_sr);
    curGeom = window.__ancSim.geomKey();
    setMakeup();
    makeSources();
  }

  async function inferLoop() {
    if (stopReq) return;
    if (!inferBusy) {
      inferBusy = true;
      try {
        if (source === 'sim' && window.__ancSim.geomKey() !== curGeom) reseed();
        const pos = playhead();
        const start = ((pos - A.est_samples) % rec.len + rec.len) % rec.len;
        const taps = await inferAt(start);
        if (!stopReq && nodes) setFilters(taps);
      } catch (e) { setStatus('inference error: ' + e.message); }
      inferBusy = false;
    }
    if (!stopReq) setTimeout(inferLoop, 450);
  }

  // --- spectrum (red = before, gold = after) ---
  const fx = (cv, f) => { const lo = Math.log10(50), hi = Math.log10(2000); return (Math.log10(f) - lo) / (hi - lo) * cv.width; };
  const dy = (cv, db) => { const top = -18, bot = -95; return (1 - (Math.max(bot, Math.min(top, db)) - bot) / (top - bot)) * cv.height; };
  function path(cx, cv, arr) {
    const bins = arr.length, sr = ctx.sampleRate; cx.beginPath(); let st = false;
    for (let i = 1; i < bins; i++) { const f = i * sr / (2 * bins); if (f < 50) continue; if (f > 2000) break;
      const x = fx(cv, f), y = dy(cv, arr[i]); st ? cx.lineTo(x, y) : (cx.moveTo(x, y), st = true); }
  }
  function draw() {
    if (!running) return;
    const cv = g('rmspectrum'), cx = cv.getContext('2d');
    nodes.anBefore.getFloatFrequencyData(nodes.fB); nodes.anAfter.getFloatFrequencyData(nodes.fA);
    cx.clearRect(0, 0, cv.width, cv.height);
    cx.fillStyle = 'rgba(255,255,255,0.05)'; cx.fillRect(fx(cv, 100), 0, fx(cv, 1000) - fx(cv, 100), cv.height);
    path(cx, cv, nodes.fB); cx.lineTo(fx(cv, 2000), cv.height); cx.lineTo(fx(cv, 50), cv.height); cx.closePath();
    cx.fillStyle = 'rgba(188,18,42,0.18)'; cx.fill();
    path(cx, cv, nodes.fB); cx.strokeStyle = 'rgba(188,18,42,0.9)'; cx.lineWidth = 1.5; cx.stroke();
    path(cx, cv, nodes.fA); cx.strokeStyle = '#E8B923'; cx.lineWidth = 2.4; cx.stroke();
    drawRaf = requestAnimationFrame(draw);
  }

  async function start() {
    await loadCore();
    await buildRec();
    buildGraph();
    if (ctx.state === 'suspended') await ctx.resume();
    setStatus('computing first filter…');
    try { setFilters(await inferAt(0)); } catch (e) { setStatus('inference error: ' + e.message); return; }
    running = true; stopReq = false; inferBusy = false;
    setStatus(source === 'sim'
      ? 'live · real net on SIMULATED mics (out-of-distribution) — drag the source, toggle ANC'
      : 'live · running the real network in your browser — toggle ANC');
    inferLoop(); draw();
  }
  function stop() {
    const wasRunning = running;
    stopReq = true; running = false; cancelAnimationFrame(drawRaf);
    if (nodes) { try { nodes.gsrc.stop(); nodes.esrc.stop(); } catch (e) {} }
    if (ctx) { try { ctx.close(); } catch (e) {} }
    nodes = null; ctx = null;
    const btn = g('rmLive'); if (btn && wasRunning) btn.textContent = '⚡ Run in real time';
  }
  window.__ancLiveStop = stop;
  window.__ancOnSceneChange = () => { if (running && source === 'sim') reseed(); };

  document.addEventListener('DOMContentLoaded', () => {
    const btn = g('rmLive'); if (!btn) return;
    btn.addEventListener('click', async () => {
      if (running) { stop(); btn.textContent = '⚡ Run in real time'; setStatus('stopped'); return; }
      btn.disabled = true;
      try { await start(); btn.textContent = '■ Stop'; }
      catch (e) { setStatus('error: ' + e.message); }
      finally { btn.disabled = false; }
    });
    document.querySelectorAll('[data-rmsrc]').forEach((b) => {
      b.addEventListener('click', async () => {
        document.querySelectorAll('[data-rmsrc]').forEach((x) => x.classList.remove('active'));
        b.classList.add('active');
        const wasRunning = running;
        source = b.dataset.rmsrc;
        if (wasRunning) { stop(); btn.disabled = true; try { await start(); btn.textContent = '■ Stop'; } finally { btn.disabled = false; } }
      });
    });
    const anc = g('rmAnc');
    if (anc) anc.addEventListener('click', () => {
      if (!running) return;
      ancOn = !ancOn; applyAB();
      anc.textContent = 'ANC: ' + (ancOn ? 'ON' : 'OFF');
      anc.classList.toggle('on', ancOn);
    });
  });
})();
