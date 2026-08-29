// ============================================================
//  FingerIO — shared core
//
//  Everything that is not a page: the OFDM waveform, the audio
//  engine, the per-symbol channel-estimate pipeline, and the
//  drawing routines. Loaded by fingerio/ (one device does
//  everything) and by fingerio-remote/ (a phone runs this; the
//  laptop draws what the phone sends).
//
//  Pipeline:
//   1. Build a length-N OFDM symbol with random ±1 BPSK on the
//      subcarriers inside [bandLo, bandHi], Hermitian-symmetric so
//      the IFFT is real.
//   2. Tile the symbol into an AudioBuffer and play it on loop.
//      Because the symbol repeats, every length-N window of the
//      received signal is a circular convolution with the channel.
//   3. Capture mic samples into a ring buffer.
//   4. Re-sync periodically by cross-correlating the most recent
//      ~2N received samples against one symbol.
//   5. For each new aligned frame: FFT → divide by X(k) on active
//      subcarriers → IFFT → |h[n]|.
//   6. Subtract a per-bin EMA to remove static clutter.
//   7. Push a column to the range×time heatmap; report the
//      strongest moving range bin as the "finger" distance.
//
//  Pages own their DOM. Anything the core writes to — the stat
//  readouts, the canvases — is looked up by id, so a page that
//  omits an element simply does not get that readout.
// ============================================================

const $ = id => document.getElementById(id);
function setStat(id, html) { const el = $(id); if (el) el.innerHTML = html; }

const SOUND_C = 343;              // m/s
const ACTIVE_BIN_FLOOR = 1e-6;
const TRACK_WINDOW_S = 8;         // span of the tracked-distance plot
const COL_HZ = 30;                // heatmap columns pushed per second

// Every knob the DSP obeys. Pages copy their controls into this, and the
// remote host ships it to the phone verbatim.
const P = {
  bandLo: 18000, bandHi: 22000, symN: 256,
  bgTau: 0.6, gain: 0.30, dispRangeCm: 50
};

// ============================================================
//  Render buffers. Everything drawn comes from here and nowhere
//  else: prepareDraw() fills them locally, and the remote host
//  fills them from network packets. That is what keeps the two
//  pages pixel-identical.
// ============================================================
const R = {
  fs: 0, N: 256, nBins: 0,
  h: new Float32Array(1024),      // |h[n]| this frame, first nBins valid
  bg: new Float32Array(1024),     // clutter estimate, first nBins valid
  hScale: 1,                      // curves are drawn against this maximum
  hPeak: 0, peakBin: -1, distCm: null,
  cirRate: 0, syncOff: 0,
  colQueue: [],                   // Uint8Array columns, oldest first
  trackHist: []                   // { t (ms), cm (number|null), mag (0..1) }
};

// ============================================================
//  Radix-2 complex FFT, in place. Length must be a power of two.
// ============================================================
function fft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let pRe = 1, pIm = 0;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k],        aIm = im[i + k];
        const bRe = re[i + k + half], bIm = im[i + k + half];
        const tRe = bRe * pRe - bIm * pIm;
        const tIm = bRe * pIm + bIm * pRe;
        re[i + k]        = aRe + tRe;
        im[i + k]        = aIm + tIm;
        re[i + k + half] = aRe - tRe;
        im[i + k + half] = aIm - tIm;
        const npRe = pRe * wRe - pIm * wIm;
        pIm = pRe * wIm + pIm * wRe;
        pRe = npRe;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// ============================================================
//  Build one OFDM symbol of length N for sample rate fs and
//  active band [fmin, fmax]. Returns { x, Xre, Xim, active,
//  kmin, kmax }.
// ============================================================
function buildSymbol(N, fs, fmin, fmax) {
  const Xre = new Float64Array(N);
  const Xim = new Float64Array(N);
  const active = new Uint8Array(N);
  const kmin = Math.max(1, Math.ceil(fmin * N / fs));
  const kmax = Math.min(N / 2 - 1, Math.floor(fmax * N / fs));
  // A deterministic seed, so the symbol is identical between transmit
  // and the receiver's expectation — including across two devices.
  let seed = 0x13371337 >>> 0;
  function rnd() {
    // xorshift32
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >>> 17;
    seed ^= seed << 5;  seed >>>= 0;
    return ((seed & 0x7fffffff) / 0x7fffffff) * 2 - 1;
  }
  for (let k = kmin; k <= kmax; k++) {
    const sgn = rnd() >= 0 ? 1 : -1;   // BPSK
    Xre[k] = sgn;
    Xim[k] = 0;
    Xre[N - k] = sgn;
    Xim[N - k] = 0;
    active[k] = 1;
    active[N - k] = 1;
  }
  const re = Xre.slice();
  const im = Xim.slice();
  fft(re, im, true);
  const x = new Float32Array(N);
  let peak = 0;
  for (let n = 0; n < N; n++) {
    x[n] = re[n];
    if (Math.abs(x[n]) > peak) peak = Math.abs(x[n]);
  }
  if (peak > 0) for (let n = 0; n < N; n++) x[n] /= peak;
  return { x, Xre, Xim, active, kmin, kmax };
}

// ============================================================
//  DSP state
// ============================================================
const dsp = {
  N: 256,
  symbol: null,
  ring: null, ringMask: 0, ringWrite: 0,
  nextFrame: 0, lastSyncAt: -1e9,
  bgEMA: null,
  h: null, motion: null,          // scratch, reused every frame
  hPeak: 0, movePeak: 0, movePeakBin: -1,
  cirRateEMA: 0, lastFrameWall: 0,
  lastCol: 0                      // wall clock of the last queued column
};

function dspInit(fs) {
  dsp.N = P.symN;
  dsp.symbol = buildSymbol(dsp.N, fs, P.bandLo, P.bandHi);
  if (dsp.symbol.kmax < dsp.symbol.kmin) {
    throw new Error('Empty band — band low is above band high at this symbol length.');
  }
  // Ring buffer for received samples: must be >> 2N.
  const ringSize = 1 << Math.max(15, Math.ceil(Math.log2(dsp.N * 16)));
  dsp.ring = new Float32Array(ringSize);
  dsp.ringMask = ringSize - 1;
  dsp.ringWrite = 0;
  dsp.nextFrame = 0;
  dsp.lastSyncAt = -1e9;
  dsp.bgEMA = null;
  dsp.h = new Float32Array(dsp.N);
  dsp.motion = new Float32Array(dsp.N);
  dsp.hPeak = 0; dsp.movePeak = 0; dsp.movePeakBin = -1;
  dsp.cirRateEMA = 0; dsp.lastFrameWall = 0; dsp.lastCol = 0;
  sync.L = 0;
  R.fs = fs;
  R.N = dsp.N;
  R.colQueue.length = 0;
  R.trackHist.length = 0;
  R.cirRate = 0; R.syncOff = 0; R.distCm = null; R.peakBin = -1;
}

// How many range bins the display asks for, given the current range
// setting. Both the heatmap and the CIR plot stop here.
function displayBins() {
  const fs = R.fs || 48000;
  const N = R.N || 256;
  const maxBinFloat = (P.dispRangeCm / 100) * 2 * fs / SOUND_C;
  return Math.min(N / 2, Math.max(8, Math.ceil(maxBinFloat)));
}

// ============================================================
//  Per-frame processing: take N aligned samples, recover h[n],
//  update the background, and find the strongest mover.
// ============================================================
const work = { re: null, im: null };
function ensureWork(N) {
  if (!work.re || work.re.length !== N) {
    work.re = new Float64Array(N);
    work.im = new Float64Array(N);
  }
}

function processFrame(yStart) {
  const N = dsp.N;
  const ring = dsp.ring;
  const mask = dsp.ringMask;
  ensureWork(N);
  const Yre = work.re, Yim = work.im;
  for (let n = 0; n < N; n++) {
    Yre[n] = ring[(yStart + n) & mask];
    Yim[n] = 0;
  }
  fft(Yre, Yim, false);
  // Divide by X(k) on active bins only; |X(k)| = 1, so it is a
  // multiplication by the conjugate. Inactive bins are zeroed.
  const Xre = dsp.symbol.Xre;
  const Xim = dsp.symbol.Xim;
  const active = dsp.symbol.active;
  for (let k = 0; k < N; k++) {
    if (active[k]) {
      const a = Yre[k], b = Yim[k];
      const c = Xre[k], d = Xim[k];
      Yre[k] = a * c + b * d;
      Yim[k] = b * c - a * d;
    } else {
      Yre[k] = 0; Yim[k] = 0;
    }
  }
  fft(Yre, Yim, true);

  const h = dsp.h;
  let hPeak = ACTIVE_BIN_FLOOR;
  for (let n = 0; n < N; n++) {
    const m = Math.hypot(Yre[n], Yim[n]);
    h[n] = m;
    if (m > hPeak) hPeak = m;
  }

  // Background EMA over the per-bin magnitude.
  const fs = R.fs;
  const dt = N / fs;
  const tau = Math.max(0.05, P.bgTau);
  const alpha = 1 - Math.exp(-dt / tau);
  if (!dsp.bgEMA || dsp.bgEMA.length !== N) {
    dsp.bgEMA = h.slice();
  } else {
    const bg = dsp.bgEMA;
    for (let n = 0; n < N; n++) bg[n] = bg[n] + alpha * (h[n] - bg[n]);
  }

  // Motion = max(0, |h| − bg). Skip the immediate direct-path bins
  // (first ~3 cm): speaker→mic coupling lives there and dominates.
  const bgArr = dsp.bgEMA;
  const motion = dsp.motion;
  let movePeak = 0, movePeakBin = -1;
  const minBin = Math.max(2, Math.round(0.03 * 2 * fs / SOUND_C));
  for (let n = 0; n < N / 2; n++) {
    let v = h[n] - bgArr[n];
    if (v < 0) v = 0;
    motion[n] = v;
    if (n >= minBin && v > movePeak) { movePeak = v; movePeakBin = n; }
  }

  dsp.hPeak = hPeak;
  dsp.movePeak = movePeak;
  dsp.movePeakBin = movePeakBin;
}

// ============================================================
//  Sync: cross-correlate the most recent 2N samples of received
//  signal against the known symbol; the lag at the peak is the
//  start of an aligned period in the ring buffer.
//
//  FFT-based correlation: pad x and y to length L = 2N, compute
//  Y·conj(X), inverse FFT, take the peak over the first N taps.
// ============================================================
const sync = { Xre: null, Xim: null, L: 0, Yre: null, Yim: null };
function ensureSync(N) {
  const L = 2 * N;
  if (sync.L === L) return;
  sync.L = L;
  sync.Xre = new Float64Array(L);
  sync.Xim = new Float64Array(L);
  sync.Yre = new Float64Array(L);
  sync.Yim = new Float64Array(L);
  const x = dsp.symbol.x;
  for (let n = 0; n < N; n++) sync.Xre[n] = x[n];
  fft(sync.Xre, sync.Xim, false);
}

function findSync(yEnd) {
  // yEnd is the absolute write index (one past the last received
  // sample). Correlate the last 2N samples ending there.
  const N = dsp.N;
  const L = 2 * N;
  ensureSync(N);
  const ring = dsp.ring;
  const mask = dsp.ringMask;
  const Yre = sync.Yre, Yim = sync.Yim;
  const start = yEnd - L;
  for (let n = 0; n < L; n++) {
    Yre[n] = ring[(start + n) & mask];
    Yim[n] = 0;
  }
  fft(Yre, Yim, false);
  for (let k = 0; k < L; k++) {
    const a = Yre[k], b = Yim[k];
    const c = sync.Xre[k], d = -sync.Xim[k];
    Yre[k] = a * c - b * d;
    Yim[k] = a * d + b * c;
  }
  fft(Yre, Yim, true);
  let best = -1, bestVal = -1;
  for (let n = 0; n < N; n++) {
    const m = Yre[n] * Yre[n] + Yim[n] * Yim[n];
    if (m > bestVal) { bestVal = m; best = n; }
  }
  // Lag `best` means the symbol begins at absolute sample start + best.
  return start + best;
}

// ============================================================
//  Sample intake. Called with whatever the drain hands over: write
//  into the ring, re-sync when due, and process every aligned frame
//  whose last sample has arrived.
// ============================================================
function ofdmSamples(input, n) {
  if (!dsp.symbol) return;
  const ring = dsp.ring;
  const mask = dsp.ringMask;
  for (let i = 0; i < n; i++) ring[(dsp.ringWrite + i) & mask] = input[i];
  dsp.ringWrite += n;

  const N = dsp.N;
  const fs = R.fs || 48000;
  // Re-sync about every 0.5 s, and once at the very start.
  if (dsp.ringWrite - dsp.lastSyncAt > fs * 0.5 && dsp.ringWrite > 2 * N) {
    const lag = findSync(dsp.ringWrite);
    if (lag >= 0) {
      // Snap nextFrame onto the new alignment, just past whatever we
      // have already consumed.
      const delta = dsp.ringWrite - lag;
      const periods = Math.ceil(delta / N);
      dsp.nextFrame = lag + periods * N;
      dsp.lastSyncAt = dsp.ringWrite;
    }
  }

  while (dsp.nextFrame + N <= dsp.ringWrite
         && dsp.nextFrame > 0
         && dsp.ringWrite - dsp.nextFrame < dsp.ring.length - N) {
    const frame = dsp.nextFrame;
    dsp.nextFrame += N;
    processFrame(frame);
    onFrame();
  }
}

// One finished CIR: update the rate estimate, record a tracked-distance
// sample, and queue a heatmap column if enough wall time has passed.
function onFrame() {
  const now = performance.now();
  const dt = now - dsp.lastFrameWall;
  // Several frames can land inside one clock tick when the drain is
  // catching up; those carry no timing information.
  if (dsp.lastFrameWall && dt > 0) {
    const rate = 1000 / dt;
    dsp.cirRateEMA = dsp.cirRateEMA
      ? dsp.cirRateEMA + 0.05 * (rate - dsp.cirRateEMA)
      : rate;
    dsp.lastFrameWall = now;
  } else if (!dsp.lastFrameWall) {
    dsp.lastFrameWall = now;
  }

  const valid = dsp.movePeakBin >= 0 && dsp.movePeak > 0.02 * dsp.hPeak;
  const cm = valid ? binToCm(dsp.movePeakBin) : null;
  const mag = dsp.hPeak > 0 ? Math.min(1, dsp.movePeak / dsp.hPeak) : 0;
  pushTrack(now, cm, mag);

  // The heatmap scrolls one pixel column per queued column. The CIR
  // arrives far faster than the display needs, so thin it here rather
  // than throwing away columns at draw time.
  if (now - dsp.lastCol >= 1000 / COL_HZ) {
    dsp.lastCol = now;
    R.colQueue.push(quantizeColumn());
    if (R.colQueue.length > 120) R.colQueue.splice(0, R.colQueue.length - 120);
  }
}

function binToCm(bin) {
  return (bin * SOUND_C / (2 * (R.fs || 48000))) * 100;
}

function pushTrack(t, cm, mag) {
  R.trackHist.push({ t, cm, mag });
  const cutoff = t - (TRACK_WINDOW_S + 1) * 1000;
  while (R.trackHist.length && R.trackHist[0].t < cutoff) R.trackHist.shift();
}

// A heatmap column, already normalized the way the display wants it:
// byte 0..255 is motion relative to the frame's own scale, so the
// column travels over the network without needing its scale factor.
function quantizeColumn() {
  const maxBin = displayBins();
  const norm = Math.max(0.02, dsp.hPeak * 0.6);
  const motion = dsp.motion;
  const col = new Uint8Array(maxBin);
  for (let n = 0; n < maxBin; n++) {
    const t = Math.min(1, motion[n] / norm);
    col[n] = Math.round(t * 255);
  }
  return col;
}

// ============================================================
//  prepareDraw — copy the DSP's latest state into R. The local page
//  calls this every animation frame; the phone calls it once per
//  packet and ships what it produced.
// ============================================================
function prepareDraw() {
  const maxBin = displayBins();
  R.nBins = maxBin;
  R.hPeak = dsp.hPeak;
  R.peakBin = dsp.movePeakBin;
  R.cirRate = dsp.cirRateEMA;
  R.syncOff = dsp.nextFrame % dsp.N;
  const valid = dsp.movePeakBin >= 0 && dsp.movePeak > 0.01 * dsp.hPeak;
  R.distCm = valid ? binToCm(dsp.movePeakBin) : null;

  const bg = dsp.bgEMA;
  let hi = 0;
  for (let n = 0; n < maxBin; n++) {
    const a = dsp.h[n], b = bg ? bg[n] : 0;
    R.h[n] = a;
    R.bg[n] = b;
    if (a > hi) hi = a;
    if (b > hi) hi = b;
  }
  R.hScale = hi > 0 ? hi : 1;
}

// ============================================================
//  Canvases
// ============================================================
const view = { ctx: null, W: 0, H: 0, dpr: 1 };

function setupCanvases() {
  const rtCanvas = $('rtCanvas');
  if (!rtCanvas) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(200, Math.floor((rtCanvas.clientWidth || 800) * dpr));
  const H = Math.max(120, Math.floor((rtCanvas.clientHeight || 320) * dpr));
  rtCanvas.width = W;
  rtCanvas.height = H;
  const ctx = rtCanvas.getContext('2d');
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, W, H);
  for (const id of ['cirCanvas', 'trackCanvas']) {
    const cv = $(id);
    if (!cv) continue;
    cv.width  = Math.max(200, Math.floor((cv.clientWidth  || 800) * dpr));
    cv.height = Math.max(80,  Math.floor((cv.clientHeight || 180) * dpr));
  }
  view.ctx = ctx; view.W = W; view.H = H; view.dpr = dpr;
}
window.addEventListener('resize', () => { if (view.ctx) setupCanvases(); });

// Color ramp: black → red → gold → white, for v in [0,1].
function colorRamp(v) {
  v = Math.max(0, Math.min(1, v));
  if (v < 0.33) {
    const t = v / 0.33;
    return [Math.round(188 * t), Math.round(18 * t), Math.round(42 * t)];
  } else if (v < 0.66) {
    const t = (v - 0.33) / 0.33;
    return [188 + Math.round((232 - 188) * t),
            18  + Math.round((185 - 18)  * t),
            42  + Math.round((35  - 42)  * t)];
  }
  const t = (v - 0.66) / 0.34;
  return [232 + Math.round((255 - 232) * t),
          185 + Math.round((255 - 185) * t),
          35  + Math.round((255 - 35)  * t)];
}

// Scroll the heatmap left by one pixel and paint the newest column.
function drawColumn(col) {
  if (!view.ctx || !col || !col.length) return;
  const { ctx, W, H } = view;
  const img = ctx.getImageData(1, 0, W - 1, H);
  ctx.putImageData(img, 0, 0);
  const strip = ctx.createImageData(1, H);
  const maxBin = col.length;
  for (let y = 0; y < H; y++) {
    // Top of canvas = farthest range; bottom = closest.
    const r = (H - 1 - y) / (H - 1);
    const binFloat = r * maxBin;
    const b0 = Math.min(maxBin - 1, Math.floor(binFloat));
    const b1 = Math.min(maxBin - 1, b0 + 1);
    const f = binFloat - b0;
    const t = (col[b0] * (1 - f) + col[b1] * f) / 255;
    const g = Math.pow(t, 0.7);          // mild gamma to lift mid-tones
    const [Rc, Gc, Bc] = colorRamp(g);
    const idx = y * 4;
    strip.data[idx]     = Rc;
    strip.data[idx + 1] = Gc;
    strip.data[idx + 2] = Bc;
    strip.data[idx + 3] = 255;
  }
  ctx.putImageData(strip, W - 1, 0);
}

function drawCIR() {
  const cv = $('cirCanvas');
  if (!cv || !cv.width) return;
  const dpr = view.dpr;
  const W = cv.width, H = cv.height;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, W, H);
  const maxBin = R.nBins;
  if (!maxBin) return;
  const fs = R.fs || 48000;
  const maxRangeM = P.dispRangeCm / 100;
  const hi = R.hScale > 0 ? R.hScale : 1;
  const padX = 24 * dpr, padTop = 8 * dpr, padBot = 18 * dpr;
  const plotW = W - padX - 8 * dpr;
  const plotH = H - padTop - padBot;

  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1 * dpr;
  ctx.font = (10 * dpr) + 'px "Roboto Mono", monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.textAlign = 'center';
  for (let cm = 10; cm <= maxRangeM * 100; cm += 10) {
    const binF = (cm / 100) * 2 * fs / SOUND_C;
    const x = padX + (binF / maxBin) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + plotH);
    ctx.stroke();
    ctx.fillText(cm + ' cm', x, H - 4 * dpr);
  }
  const curve = (arr, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width * dpr;
    ctx.beginPath();
    for (let n = 0; n < maxBin; n++) {
      const x = padX + (n / maxBin) * plotW;
      const y = padTop + plotH - (arr[n] / hi) * plotH;
      if (n === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  curve(R.bg, '#5b6068', 1.5);
  curve(R.h,  '#bc122a', 1.6);

  const peakBin = R.peakBin;
  if (peakBin >= 0 && peakBin < maxBin) {
    const x = padX + (peakBin / maxBin) * plotW;
    const y = padTop + plotH - (R.h[peakBin] / hi) * plotH;
    ctx.fillStyle = '#E8B923';
    ctx.beginPath(); ctx.arc(x, y, 3 * dpr, 0, Math.PI * 2); ctx.fill();
  }
}

// Tracked-distance plot: hand position over the last few seconds.
function drawTrack() {
  const cv = $('trackCanvas');
  if (!cv || !cv.width) return;
  const dpr = view.dpr;
  const W = cv.width, H = cv.height;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, W, H);

  const padL = 32 * dpr, padR = 8 * dpr;
  const padT = 10 * dpr, padB = 16 * dpr;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const maxCm = P.dispRangeCm;
  const now = performance.now();
  const tEnd = now;
  const tStart = now - TRACK_WINDOW_S * 1000;
  const yOf = cm => padT + plotH * (1 - cm / maxCm);
  const xOf = t  => padL + plotW * (t - tStart) / (tEnd - tStart);

  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1 * dpr;
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.font = (10 * dpr) + 'px "Roboto Mono", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const cmStep = maxCm <= 30 ? 5 : 10;
  for (let cm = 0; cm <= maxCm; cm += cmStep) {
    const y = yOf(cm);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(cm + ' cm', padL - 4 * dpr, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= TRACK_WINDOW_S; s += 2) {
    const x = xOf(tEnd - s * 1000);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.fillText('-' + s + 's', x, padT + plotH + 2 * dpr);
  }

  const samples = R.trackHist.filter(s => s.t >= tStart);
  if (samples.length < 2) return;

  // Light EMA smoothing on cm so the line reads cleanly. Skips invalid
  // samples and resets after long invalid gaps.
  const smoothed = new Array(samples.length);
  let sm = null, lastValidT = -1e9;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.cm == null) { smoothed[i] = { t: s.t, cm: null, mag: s.mag }; continue; }
    if (sm == null || s.t - lastValidT > 250) sm = s.cm;
    else sm = sm + 0.35 * (s.cm - sm);
    lastValidT = s.t;
    smoothed[i] = { t: s.t, cm: sm, mag: s.mag };
  }

  ctx.strokeStyle = '#E8B923';
  ctx.lineWidth = 2 * dpr;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  let pen = false;
  for (const s of smoothed) {
    if (s.cm == null) { pen = false; continue; }
    const x = xOf(s.t), y = yOf(s.cm);
    if (!pen) { ctx.moveTo(x, y); pen = true; } else { ctx.lineTo(x, y); }
  }
  ctx.stroke();

  // Strength dots — size proportional to motion magnitude.
  for (const s of smoothed) {
    if (s.cm == null) continue;
    const x = xOf(s.t), y = yOf(s.cm);
    const r = Math.max(1, Math.min(5, s.mag * 12)) * dpr;
    ctx.fillStyle = 'rgba(188,18,42,0.45)';
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  let latest = null;
  for (let i = smoothed.length - 1; i >= 0; i--) {
    if (smoothed[i].cm != null) { latest = smoothed[i]; break; }
  }
  if (latest && now - latest.t < 300) {
    const x = xOf(latest.t), y = yOf(latest.cm);
    ctx.fillStyle = '#E8B923';
    ctx.beginPath(); ctx.arc(x, y, 4 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#0c0c0e';
    ctx.beginPath(); ctx.arc(x, y, 1.6 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#E8B923';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.font = 'bold ' + (11 * dpr) + 'px "Roboto Mono", monospace';
    ctx.fillText(latest.cm.toFixed(1) + ' cm', x - 6 * dpr, y - 4 * dpr);
  }
}

// Everything the display shows, from R alone.
function drawAll() {
  const cols = R.colQueue.splice(0, R.colQueue.length);
  for (const col of cols) drawColumn(col);
  drawCIR();
  drawTrack();
  setStat('srStat',   (R.fs ? Math.round(R.fs) : '—') + '<span class="unit"> Hz</span>');
  setStat('rateStat', (R.cirRate ? R.cirRate.toFixed(0) : '—') + '<span class="unit"> Hz</span>');
  setStat('syncStat', R.syncOff + '<span class="unit"> mod N</span>');
  setStat('distStat', (R.distCm == null ? '—' : R.distCm.toFixed(1)) + '<span class="unit"> cm</span>');
}

// ============================================================
//  Audio engine
// ============================================================
const eng = {
  ctx: null, src: null, gain: null, mic: null, sp: null, node: null,
  stream: null, running: false, capture: '',
  // Capture is decoupled from processing. The audio thread only hands
  // over samples; the DSP drains them on the main thread. If a frame
  // runs long the queue grows and is caught up on the next tick —
  // running the DSP inside the audio callback instead means the browser
  // simply skips callbacks it cannot deliver, and the samples are gone.
  queue: [], qLen: 0, drainTimer: null,
  recv: 0, proc: 0, dropped: 0, rateEMA: 0, lastDrain: 0
};

const CAPTURE_WORKLET = `
class FingerCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(1024);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('finger-capture', FingerCapture);
`;

// About four seconds of audio. Past that something is badly wrong and
// holding the backlog only makes the catch-up worse.
const MAX_QUEUE = 4 * 48000;

function pushCapture(buf) {
  if (!eng.running) return;
  eng.queue.push(buf);
  eng.qLen += buf.length;
  eng.recv += buf.length;
  while (eng.qLen > MAX_QUEUE) {
    const old = eng.queue.shift();
    eng.qLen -= old.length;
    eng.dropped += old.length;
    // A hole in the stream shifts every later sample, so the symbol
    // grid no longer lines up with the transmitter. Find it again.
    dsp.lastSyncAt = -1e9;
  }
}

// Runs the DSP over everything captured since the last call.
function drainCapture() {
  if (!eng.running || !eng.queue.length) return;
  const batch = eng.queue;
  eng.queue = [];
  eng.qLen = 0;
  let count = 0;
  // Feed the DSP in bounded slices. A backlog handed over in one lump
  // can be longer than the ring, in which case the oldest samples would
  // be overwritten before the frame grid reached them; slicing keeps
  // the grid inside the ring no matter how far behind we are.
  const MAX_CHUNK = Math.max(2048, dsp.ring ? (dsp.ring.length >> 2) : 4096);
  for (const buf of batch) {
    for (let off = 0; off < buf.length; off += MAX_CHUNK) {
      const sub = buf.subarray(off, Math.min(off + MAX_CHUNK, buf.length));
      ofdmSamples(sub, sub.length);
    }
    count += buf.length;
  }
  eng.proc += count;
  const now = performance.now();
  if (eng.lastDrain) {
    const rate = count / ((now - eng.lastDrain) / 1000);
    eng.rateEMA = eng.rateEMA ? eng.rateEMA + 0.1 * (rate - eng.rateEMA) : rate;
  }
  eng.lastDrain = now;
}

// Page hooks. The defaults do nothing, so a page can ignore any of them.
let onStatus = () => {};
let onEngine = () => {};
let onLayout = () => {};
function setStatus(msg, kind) { onStatus(msg, kind); }

function buildTxBuffer(ctx) {
  const fs = ctx.sampleRate;
  // A multiple of N keeps the loop seamless: the symbol has to butt up
  // against itself, or the receiver's circular-convolution assumption
  // breaks once per buffer.
  const tiles = Math.max(8, Math.ceil(fs / dsp.N));
  const buf = ctx.createBuffer(1, tiles * dsp.N, fs);
  const ch = buf.getChannelData(0);
  for (let t = 0; t < tiles; t++) ch.set(dsp.symbol.x, t * dsp.N);
  return buf;
}

async function engStart() {
  try {
    if (eng.running) engStop();
    setStatus('Setting up audio…', 'busy');
    // Created on a user gesture and then kept: a phone will not hand out
    // a fresh context without another tap.
    if (!eng.ctx) eng.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (eng.ctx.state === 'suspended') await eng.ctx.resume();
    const fs = eng.ctx.sampleRate;
    dspInit(fs);

    eng.src = eng.ctx.createBufferSource();
    eng.src.buffer = buildTxBuffer(eng.ctx);
    eng.src.loop = true;
    eng.gain = eng.ctx.createGain();
    eng.gain.gain.value = P.gain;
    eng.src.connect(eng.gain).connect(eng.ctx.destination);

    setStatus('Requesting microphone…', 'busy');
    eng.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 }
    });
    eng.mic = eng.ctx.createMediaStreamSource(eng.stream);
    const sink = eng.ctx.createGain();
    sink.gain.value = 0;
    sink.connect(eng.ctx.destination);

    let worklet = false;
    if (eng.ctx.audioWorklet) {
      try {
        if (!eng.workletReady) {
          const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: 'application/javascript' }));
          eng.workletReady = eng.ctx.audioWorklet.addModule(url);
        }
        await eng.workletReady;
        eng.node = new AudioWorkletNode(eng.ctx, 'finger-capture', { numberOfOutputs: 1 });
        eng.node.port.onmessage = e => pushCapture(e.data);
        eng.mic.connect(eng.node);
        eng.node.connect(sink);
        worklet = true;
      } catch (e) {
        console.warn('AudioWorklet unavailable, falling back to ScriptProcessor', e);
        eng.workletReady = null;
      }
    }
    if (!worklet) {
      // Same discipline as the worklet: copy out and leave; no DSP here.
      eng.sp = eng.ctx.createScriptProcessor(2048, 1, 1);
      eng.sp.onaudioprocess = ev => pushCapture(ev.inputBuffer.getChannelData(0).slice(0));
      eng.mic.connect(eng.sp);
      eng.sp.connect(sink);
    }
    eng.capture = worklet ? 'worklet' : 'scriptprocessor';

    onLayout();
    setupCanvases();
    eng.src.start();
    eng.running = true;
    startDrain();
    setStatus('Listening at ' + Math.round(fs) + ' Hz — wave a hand above the speaker.', 'ok');
    onEngine(true);
  } catch (err) {
    console.error(err);
    setStatus('Could not start: ' + err.message, 'error');
    engStop();
  }
}

function startDrain() {
  resetCapture();
  if (eng.drainTimer) return;
  // A timer, not requestAnimationFrame: rAF stops entirely in a hidden
  // tab, and a stalled DSP is worse than a stalled picture.
  eng.drainTimer = setInterval(drainCapture, 16);
}
function resetCapture() {
  eng.queue = []; eng.qLen = 0;
  eng.recv = 0; eng.proc = 0; eng.dropped = 0;
  eng.rateEMA = 0; eng.lastDrain = 0;
}

function engStop() {
  eng.running = false;
  if (eng.drainTimer) { clearInterval(eng.drainTimer); eng.drainTimer = null; }
  eng.queue = []; eng.qLen = 0;
  try { eng.src && eng.src.stop(); } catch (e) {}
  for (const node of [eng.sp, eng.node, eng.mic, eng.gain, eng.src]) {
    if (node) try { node.disconnect(); } catch (e) {}
  }
  if (eng.stream) eng.stream.getTracks().forEach(t => t.stop());
  eng.src = eng.gain = eng.mic = eng.sp = eng.node = eng.stream = null;
  setStatus('Stopped.', '');
  onEngine(false);
}
