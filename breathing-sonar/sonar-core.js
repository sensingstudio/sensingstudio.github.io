// ============================================================
//  Breathing Sonar — shared sonar core
//
//  Everything that is not a page: the DSP for both methods, the
//  drawing routines, and the synthetic demo source. Loaded by
//  breathing-sonar/ (one laptop does everything) and by
//  breathing-sonar-remote/ (a phone runs this; the laptop draws it).
//
//  Method 1 (CW):   one tone, quadrature demodulation, clutter
//                   cancellation, arc fit, unwrapped phase.
//  Method 2 (FMCW): repeating sweep, dechirp, FFT over M sweeps,
//                   per-cell 30 s FFT for range and rate.
//                   After Nandakumar et al., ApneaApp, MobiSys'15.
//
//  Pages own their DOM. Anything the core writes to — the stat
//  readouts, the canvases — is looked up by id, so a page that
//  omits an element simply does not get that readout.
// ============================================================

const $ = id => document.getElementById(id);
function setStat(id, html) { const el = $(id); if (el) el.innerHTML = html; }

// Which method is running. Pages set this; the core reads it.
let method = 'cw';

// Every knob the DSP obeys. Pages copy their controls into this, and the
// remote host ships it to the phone verbatim.
const P = {
  toneHz: 18000, clutterTau: 6, smoothTau: 0.25,
  bandLo: 18000, bandHi: 20000, chirpN: 512, chirps: 8,
  rangeCm: 120, gain: 0.35
};

const C_SOUND   = 343;      // m/s
const BR_LO_HZ  = 0.12;     // 7.2 breaths/min — below this is drift,
const BR_HI_HZ  = 0.70;     // 42 breaths/min    not respiration
const BR_WIN_S  = 30;       // paper: 30 s FFT for the breathing peak
const BR_CONF   = 5.0;      // peak-to-band-mean below this is not breathing
const ENV_WIN_S = 8;        // RMS window for the amplitude envelope
const ENV_HIST_S = 120;     // envelope strip span

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
        const aRe = re[i + k], aIm = im[i + k];
        const bRe = re[i + k + half], bIm = im[i + k + half];
        const tRe = bRe * pRe - bIm * pIm;
        const tIm = bRe * pIm + bIm * pRe;
        re[i + k] = aRe + tRe;
        im[i + k] = aIm + tIm;
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
//  Tiny ring buffer of floats, newest-last semantics.
// ============================================================
function makeRing(n) { return { buf: new Float32Array(n), n, w: 0, filled: 0 }; }
function ringPush(r, v) {
  r.buf[r.w] = v;
  r.w = (r.w + 1) % r.n;
  if (r.filled < r.n) r.filled++;
}
function ringLast(r, count, out) {
  count = Math.min(count, r.filled);
  let idx = (r.w - count + r.n) % r.n;
  for (let i = 0; i < count; i++) {
    out[i] = r.buf[idx];
    idx = idx + 1 === r.n ? 0 : idx + 1;
  }
  return count;
}

// ============================================================
//  Breathing analysis: 30 s window, Hann, zero-padded FFT, peak
//  hunt inside the respiration band. Shared by both methods.
// ============================================================
const anaWork = { buf: null, re: null, im: null };
function analyzeBreathing(ring, rate) {
  const want = Math.round(BR_WIN_S * rate);
  const L = Math.min(ring.filled, want);
  if (!anaWork.buf || anaWork.buf.length < L) anaWork.buf = new Float32Array(want + 8);
  ringLast(ring, L, anaWork.buf);
  return analyzeArray(anaWork.buf, L, rate);
}
function analyzeArray(x, L, rate) {
  if (L < Math.max(32, rate * 8)) return null;   // need ~8 s before guessing

  let mean = 0;
  for (let i = 0; i < L; i++) mean += x[i];
  mean /= L;

  let NFFT = 1;
  while (NFFT < L * 2) NFFT <<= 1;
  if (!anaWork.re || anaWork.re.length !== NFFT) {
    anaWork.re = new Float64Array(NFFT);
    anaWork.im = new Float64Array(NFFT);
  }
  const re = anaWork.re, im = anaWork.im;
  re.fill(0); im.fill(0);
  let winSum = 0;
  for (let i = 0; i < L; i++) {
    const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (L - 1));
    re[i] = (x[i] - mean) * w;
    winSum += w;
  }
  fft(re, im, false);

  const df = rate / NFFT;
  // Amplitude calibration: a sinusoid of amplitude A peaks at A.
  const scale = 2 / winSum;
  const kEnd = Math.min(NFFT / 2 - 1, Math.floor(1.05 / df));
  const kLo = Math.max(1, Math.ceil(BR_LO_HZ / df));
  // Never look above 0.45·rate: a slow frame rate cannot represent
  // the top of the band, and the peak hunt must stay inside spec[].
  const kHi = Math.min(NFFT / 2 - 2, kEnd - 2,
                       Math.floor(Math.min(BR_HI_HZ, 0.45 * rate) / df));
  if (kHi <= kLo) return null;
  const spec = new Float32Array(kEnd);
  for (let k = 0; k < kEnd; k++) spec[k] = Math.hypot(re[k], im[k]) * scale;

  // The peak is hunted one bin inside the band. Phase drift has a
  // 1/f shape that always piles onto the lowest bin available, and
  // a "peak" pinned to the band edge is that, not a breath.
  let peakK = kLo + 1, peakV = -1, bandSum = 0;
  for (let k = kLo; k <= kHi; k++) bandSum += spec[k];
  for (let k = kLo + 1; k <= kHi - 1; k++) {
    if (spec[k] > peakV) { peakV = spec[k]; peakK = k; }
  }
  if (peakV < 0) return null;
  const bandMean = bandSum / (kHi - kLo + 1);
  // Parabolic interpolation around the peak for sub-bin frequency.
  const a = spec[peakK - 1], b = spec[peakK], c = spec[peakK + 1];
  const denom = (a - 2 * b + c);
  const delta = denom !== 0 ? 0.5 * (a - c) / denom : 0;
  const fPeak = (peakK + Math.max(-1, Math.min(1, delta))) * df;

  // Peak-to-peak of the periodic component only. Reading this off
  // the spectral peak rather than the raw trace matters at low
  // carrier frequencies, where λ/4π multiplies phase jitter into
  // centimetres of nonexistent chest movement.
  const ampPP = 2 * peakV;
  return {
    bpm: fPeak * 60,
    conf: bandMean > 0 ? peakV / bandMean : 0,
    ampPP, spec, df, kLo, kHi, peakK, fPeak, seconds: L / rate
  };
}

// Running RMS of the last ENV_WIN_S seconds of a ring.
function envelopeRms(ring, rate) {
  const L = Math.min(ring.filled, Math.round(ENV_WIN_S * rate));
  if (!anaWork.buf || anaWork.buf.length < L) anaWork.buf = new Float32Array(L + 8);
  ringLast(ring, L, anaWork.buf);
  return envelopeArray(anaWork.buf, L, rate);
}
// RMS of the newest ENV_WIN_S seconds of a chronological array of length n.
function envelopeArray(all, n, rate) {
  const L = Math.min(n, Math.round(ENV_WIN_S * rate));
  if (L < 8) return 0;
  const x = all.subarray ? all.subarray(n - L, n) : all.slice(n - L, n);
  let m = 0;
  for (let i = 0; i < L; i++) m += x[i];
  m /= L;
  let s = 0;
  for (let i = 0; i < L; i++) { const d = x[i] - m; s += d * d; }
  return Math.sqrt(s / L);
}

// Event scoring in the spirit of ApneaApp's amplitude test.
function makeEventState() {
  return { baseline: 0, hist: makeRing(Math.round(ENV_HIST_S * 2) + 4), label: 'waiting', cls: 'idle', seen: false };
}
function updateEvent(ev, env, conf) {
  const periodic = conf >= BR_CONF;
  // The baseline is only allowed to grow on signal that actually
  // looks like breathing. Otherwise an empty room quietly sets its
  // own noise as "normal" and every reading looks healthy.
  if (periodic) {
    ev.seen = true;
    if (ev.baseline <= 0) ev.baseline = env;
    const a = env > ev.baseline ? 0.08 : 0.004;
    ev.baseline += a * (env - ev.baseline);
  } else if (ev.baseline > 0) {
    ev.baseline += 0.004 * (env - ev.baseline);
  }
  const ratio = ev.baseline > 0 ? env / ev.baseline : 0;
  ringPush(ev.hist, ratio);
  if (!ev.seen || ev.baseline <= 1e-9) { ev.label = 'waiting for breathing'; ev.cls = 'idle'; }
  else if (ratio < 0.15 || (ratio < 0.3 && !periodic)) { ev.label = 'no motion — apnea-like'; ev.cls = 'bad'; }
  else if (ratio < 0.3) { ev.label = 'reduced <30% — hypopnea-like'; ev.cls = 'warn'; }
  else if (ratio < 0.7) { ev.label = 'reduced'; ev.cls = 'warn'; }
  else if (!periodic) { ev.label = 'motion, not periodic'; ev.cls = 'warn'; }
  else { ev.label = 'breathing'; ev.cls = ''; }
  return ratio;
}
function paintBadge(el, ev) {
  el.textContent = ev.label;
  el.className = 'event-badge' + (ev.cls ? ' ' + ev.cls : '');
}

// ============================================================
//  Canvas helpers
// ============================================================
function prep(cv, clear) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(160, Math.floor((cv.clientWidth || 600) * dpr));
  const H = Math.max(80, Math.floor((cv.clientHeight || 200) * dpr));
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext('2d');
  if (clear !== false) { ctx.fillStyle = '#0c0c0e'; ctx.fillRect(0, 0, W, H); }
  return { ctx, W, H, dpr };
}
function axisText(ctx, dpr) {
  ctx.font = (10 * dpr) + 'px "Roboto Mono", monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
}

// Time-series plot. data holds `n` chronological samples.
function drawTrace(cv, data, n, opts) {
  const o = opts || {};
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 40 * dpr, padR = 8 * dpr, padT = 10 * dpr, padB = 18 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  if (n < 2) {
    axisText(ctx, dpr);
    ctx.textAlign = 'center';
    ctx.fillText(o.waiting || 'collecting…', W / 2, H / 2);
    return;
  }
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { const v = data[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (o.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi), o.minSpan || 1e-9); lo = -m; hi = m; }
  if (hi - lo < (o.minSpan || 1e-9)) { const mid = (hi + lo) / 2, h = (o.minSpan || 1e-9) / 2; lo = mid - h; hi = mid + h; }
  const pad = (hi - lo) * 0.12;
  lo -= pad; hi += pad;
  const yOf = v => padT + plotH - (v - lo) / (hi - lo) * plotH;
  const xOf = i => padL + (i / (n - 1)) * plotW;

  // zero line + y labels
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1 * dpr;
  axisText(ctx, dpr);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let g = 0; g <= 2; g++) {
    const v = lo + (hi - lo) * g / 2;
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(o.fmtY ? o.fmtY(v) : v.toFixed(1), padL - 5 * dpr, y);
  }
  // x labels in seconds before now
  if (o.spanSec) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    for (let f = 0; f <= 1.0001; f += 0.25) {
      const x = padL + f * plotW;
      const sec = Math.round(o.spanSec * (1 - f));
      ctx.fillText(sec === 0 ? 'now' : '-' + sec + 's', x, H - 5 * dpr);
    }
  }
  ctx.strokeStyle = o.color || '#E8B923';
  ctx.lineWidth = (o.width || 1.8) * dpr;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xOf(i), y = yOf(data[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  if (o.label) {
    axisText(ctx, dpr);
    ctx.textAlign = 'left';
    ctx.fillText(o.label, padL + 6 * dpr, padT + 11 * dpr);
  }
}

// Breathing spectrum, x axis in breaths per minute.
function drawSpectrum(cv, res) {
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 40 * dpr, padR = 8 * dpr, padT = 10 * dpr, padB = 20 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  axisText(ctx, dpr);
  if (!res) {
    ctx.textAlign = 'center';
    ctx.fillText('collecting… (needs ~8 s)', W / 2, H / 2);
    return;
  }
  const maxBpm = 60;
  const kMax = Math.min(res.spec.length - 1, Math.floor((maxBpm / 60) / res.df));
  let peak = 1e-12;
  for (let k = 1; k <= kMax; k++) if (res.spec[k] > peak) peak = res.spec[k];
  const xOf = bpm => padL + (bpm / maxBpm) * plotW;
  // band shading
  ctx.fillStyle = 'rgba(255,255,255,0.05)';
  ctx.fillRect(xOf(BR_LO_HZ * 60), padT, xOf(BR_HI_HZ * 60) - xOf(BR_LO_HZ * 60), plotH);
  // grid
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1 * dpr;
  ctx.textAlign = 'center';
  for (let bpm = 0; bpm <= maxBpm; bpm += 10) {
    const x = xOf(bpm);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.fillText(bpm, x, H - 6 * dpr);
  }
  // curve
  ctx.strokeStyle = '#bc122a';
  ctx.lineWidth = 1.8 * dpr;
  ctx.beginPath();
  for (let k = 1; k <= kMax; k++) {
    const x = xOf(k * res.df * 60);
    const y = padT + plotH - (res.spec[k] / peak) * plotH;
    if (k === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // peak marker
  const px = xOf(res.bpm);
  ctx.strokeStyle = '#E8B923';
  ctx.setLineDash([4 * dpr, 3 * dpr]);
  ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, padT + plotH); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#E8B923';
  ctx.textAlign = px > W * 0.75 ? 'right' : 'left';
  ctx.font = 'bold ' + (11 * dpr) + 'px "Roboto Mono", monospace';
  ctx.fillText(res.bpm.toFixed(1) + ' /min', px + (px > W * 0.75 ? -6 : 6) * dpr, padT + 12 * dpr);
  axisText(ctx, dpr);
  ctx.textAlign = 'right';
  ctx.fillText('breaths per minute', W - padR, H - 6 * dpr);
}

// Envelope strip: ratio of current RMS to its own baseline.
function drawEnvelope(cv) {
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 34 * dpr, padR = 8 * dpr, padT = 6 * dpr, padB = 14 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const yOf = r => padT + plotH - Math.max(0, Math.min(1.3, r)) / 1.3 * plotH;
  axisText(ctx, dpr);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const [r, col] of [[1.0, 'rgba(255,255,255,0.12)'], [0.7, 'rgba(255,255,255,0.25)'], [0.3, 'rgba(188,18,42,0.8)']]) {
    const y = yOf(r);
    ctx.strokeStyle = col;
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.fillText(Math.round(r * 100) + '%', padL - 5 * dpr, y);
  }
  const n = R.envN;
  if (n < 2) return;
  const tmp = R.envHist;
  // envFill is how much of the ENV_HIST_S window has been lived through,
  // so a young session draws a short line against the right-hand edge.
  const fill = Math.max(0.02, Math.min(1, R.envFill || 1));
  const x0 = padL + (1 - fill) * plotW;
  ctx.strokeStyle = '#E8B923';
  ctx.lineWidth = 1.8 * dpr;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = x0 + (i / (n - 1)) * (plotW * fill);
    const y = yOf(tmp[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  axisText(ctx, dpr);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('last ' + ENV_HIST_S + ' s', W - padR, H - 3 * dpr);
}

// ============================================================
//  Signal check — raw microphone diagnostics, shared by both
//  methods. Everything downstream is a long chain of inference;
//  this is the one place you can see what actually came in.
// ============================================================
const scope = {
  ring: makeRing(8192), fs: 48000, rms: 0, peak: 0, clip: 0,
  work: { re: new Float64Array(4096), im: new Float64Array(4096) },
  spec: new Float32Array(2048), specN: 0
};
function scopeReset(fs) {
  scope.ring = makeRing(8192);
  scope.fs = fs; scope.rms = 0; scope.peak = 0; scope.clip = 0;
  scope.specN = 0;
}
function scopePush(input, n) {
  const r = scope.ring;
  let sum = 0, pk = 0;
  for (let i = 0; i < n; i++) {
    const v = input[i];
    ringPush(r, v);
    sum += v * v;
    const a = Math.abs(v);
    if (a > pk) pk = a;
    if (a > 0.985) scope.clip++;
  }
  const rms = Math.sqrt(sum / n);
  scope.rms = scope.rms ? scope.rms + 0.15 * (rms - scope.rms) : rms;
  scope.peak = Math.max(pk, scope.peak * 0.92);
}
function dbfs(v) { return 20 * Math.log10(v + 1e-9); }

function drawScope() {
  const cv = $('scopeCanvas');
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 40 * dpr, padR = 8 * dpr, padT = 8 * dpr, padB = 16 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = R.waveN;
  axisText(ctx, dpr);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  // full scale rails
  for (const v of [1, -1]) {
    const y = padT + plotH * (0.5 - v * 0.5);
    ctx.strokeStyle = 'rgba(188,18,42,0.8)';
    ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.fillText(v.toFixed(1), padL - 5 * dpr, y);
  }
  const yMid = padT + plotH * 0.5;
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.beginPath(); ctx.moveTo(padL, yMid); ctx.lineTo(W - padR, yMid); ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.fillText('0', padL - 5 * dpr, yMid);
  if (n < 4) {
    ctx.textAlign = 'center';
    ctx.fillText('no audio yet', W / 2, yMid);
    return;
  }
  const buf = R.wave;
  ctx.strokeStyle = '#E8B923';
  ctx.lineWidth = 1.4 * dpr;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = padL + (i / (n - 1)) * plotW;
    const y = padT + plotH * (0.5 - Math.max(-1.2, Math.min(1.2, buf[i])) * 0.5);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  axisText(ctx, dpr);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(R.waveMs.toFixed(0) + ' ms', W - padR, H - 4 * dpr);
}

// Spectrum of the newest 4096 samples, Hann-windowed, in dBFS.
function computeRxSpectrum() {
  const N = 4096;
  if (scope.ring.filled < N) { scope.specN = 0; return; }
  const { re, im } = scope.work;
  const buf = new Float32Array(N);
  ringLast(scope.ring, N, buf);
  for (let i = 0; i < N; i++) {
    re[i] = buf[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)));
    im[i] = 0;
  }
  fft(re, im, false);
  // Hann coherent gain 0.5, so a full-scale sine peaks at N/4.
  const norm = 4 / N;
  for (let k = 0; k < N / 2; k++) scope.spec[k] = Math.hypot(re[k], im[k]) * norm;
  scope.specN = N / 2;
}

function drawRxSpectrum() {
  const cv = $('rxSpecCanvas');
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 44 * dpr, padR = 8 * dpr, padT = 10 * dpr, padB = 20 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const fMax = R.fs / 2;
  const dbLo = -110, dbHi = 0;
  const xOf = f => padL + (f / fMax) * plotW;
  const yOf = d => padT + plotH * (1 - (Math.max(dbLo, Math.min(dbHi, d)) - dbLo) / (dbHi - dbLo));
  // transmit band
  const bLo = method === 'cw' ? cw.f0 - 150 : fm.f0;
  const bHi = method === 'cw' ? cw.f0 + 150 : fm.f1;
  ctx.fillStyle = 'rgba(255,255,255,0.14)';
  ctx.fillRect(xOf(bLo), padT, Math.max(2 * dpr, xOf(bHi) - xOf(bLo)), plotH);
  axisText(ctx, dpr);
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1 * dpr;
  ctx.textAlign = 'center';
  for (let f = 0; f <= fMax; f += 4000) {
    const x = xOf(f);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.fillText((f / 1000).toFixed(0), x, H - 6 * dpr);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let d = 0; d >= dbLo; d -= 20) {
    const y = yOf(d);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(d, padL - 5 * dpr, y);
  }
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'right';
  ctx.fillText('kHz', W - padR, H - 6 * dpr);
  if (!R.specN) {
    ctx.textAlign = 'center';
    ctx.fillText('filling the window…', W / 2, padT + plotH / 2);
    return;
  }
  ctx.strokeStyle = '#bc122a';
  ctx.lineWidth = 1.3 * dpr;
  ctx.beginPath();
  const df = R.specDf;
  for (let k = 1; k < R.specN; k++) {
    const x = xOf(k * df), y = yOf(dbfs(R.spec[k]));
    if (k === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// Band energy: how much of the received power sits in the band we
// are actually transmitting in.
function bandFraction() {
  if (!scope.specN) return NaN;
  const df = scope.fs / (scope.specN * 2);
  const lo = method === 'cw' ? cw.f0 - 200 : fm.f0 - 100;
  const hi = method === 'cw' ? cw.f0 + 200 : fm.f1 + 100;
  let inBand = 0, total = 0;
  for (let k = 1; k < scope.specN; k++) {
    const p = scope.spec[k] * scope.spec[k];
    total += p;
    const f = k * df;
    if (f >= lo && f <= hi) inBand += p;
  }
  return total > 0 ? 10 * Math.log10(inBand / total) : NaN;
}

function drawSignalCheck() {
  drawScope();
  drawRxSpectrum();
  setStat('sigRms', (R.rms > 0 ? dbfs(R.rms).toFixed(1) : '&mdash;') + '<span class="unit"> dBFS</span>');
  setStat('sigPeak', (R.peak > 0 ? dbfs(R.peak).toFixed(1) : '&mdash;') + '<span class="unit"> dBFS</span>');
  setStat('sigBand', (isFinite(R.band) ? R.band.toFixed(1) : '&mdash;') + '<span class="unit"> dB of total</span>');
  const clipEl = $('sigClip');
  if (clipEl) clipEl.textContent = R.clip;
}

const IQ_HIST_S = 12;

// ============================================================
//  From a window of complex echo samples to millimetres.
//
//  Both methods end up here. The echo traces an arc as the chest
//  moves; the angle swept by that arc is the phase we want. The
//  obvious move — subtract the window mean and take atan2 — is
//  wrong for short arcs: the mean sits ON the arc, not at its
//  centre, so a chest that swings 30 degrees of true phase reads
//  as nearly a full turn. That error is invisible at 18 kHz, where
//  a breath sweeps most of a circle anyway, and enormous at 2 kHz.
//
//  So fit the circle (Kasa least squares) and measure the angle
//  about its centre. When the arc is too short to pin the centre
//  down, say so: the waveform is still recoverable by projecting
//  onto the arc's long axis, but its scale in millimetres is not.
// ============================================================
const MIN_ARC_RAD = 0.30;  // shorter arcs cannot pin down a centre
const MIN_SAGITTA = 4.0;   // …and the bulge must clear the scatter

function fitArcCentre(zr, zi, L, out) {
  let mx = 0, my = 0;
  for (let i = 0; i < L; i++) { mx += zr[i]; my += zi[i]; }
  mx /= L; my /= L;
  let Suu = 0, Svv = 0, Suv = 0, Suw = 0, Svw = 0, Sw = 0;
  for (let i = 0; i < L; i++) {
    const u = zr[i] - mx, v = zi[i] - my, w = u * u + v * v;
    Suu += u * u; Svv += v * v; Suv += u * v;
    Suw += u * w; Svw += v * w; Sw += w;
  }
  const D = Suu * Svv - Suv * Suv;
  const spread = Math.sqrt(Sw / L);
  out.mx = mx; out.my = my; out.spread = spread;
  if (!(Math.abs(D) > 1e-30) || spread <= 0) { out.ok = false; out.cx = mx; out.cy = my; return; }
  const a = 0.5 * (Suw * Svv - Svw * Suv) / D;
  const b = 0.5 * (Svw * Suu - Suw * Suv) / D;
  const R = Math.sqrt(a * a + b * b + Sw / L);
  const cx = mx + a, cy = my + b;
  // How well do the samples actually lie on that circle? A fit to a
  // near-straight arc lands its centre inside the noise and this
  // residual comes out as large as the radius itself.
  let sr = 0;
  for (let i = 0; i < L; i++) {
    const d = Math.hypot(zr[i] - cx, zi[i] - cy) - R;
    sr += d * d;
  }
  out.rms = Math.sqrt(sr / L);
  out.ok = isFinite(R) && R > 0;
  out.cx = cx; out.cy = cy; out.R = R;
}

const arcFit = { ok: false, cx: 0, cy: 0, mx: 0, my: 0, R: 0, spread: 0 };

// Builds both candidate motion traces from a window of complex echo:
//
//   out[]    phase about the fitted arc centre — millimetres, and the
//            only version that survives a chest sweeping past λ/2
//   outAlt[] projection onto the trajectory's long axis — right shape
//            and right rate for a short arc, but no absolute scale
//
// Which one is right depends on how far the chest swings in wavelengths,
// which is exactly what we do not know in advance. So compute both and
// let resolveMotion pick the one that actually comes out periodic.
function complexMotion(zr, zi, L, lamM, alpha, out, outAlt, arcR, arcI) {
  // Low-pass the complex echo BEFORE fitting anything. Breathing is below
  // 1 Hz, so this costs no signal, and it matters more than it looks:
  // isotropic noise on a short stroke is exactly what lets a circle fit
  // wrap a bogus centre around the data and invent a full turn of phase.
  let sr = 0, si = 0;
  for (let i = 0; i < L; i++) {
    if (i === 0) { sr = zr[i]; si = zi[i]; }
    else { sr += alpha * (zr[i] - sr); si += alpha * (zi[i] - si); }
    arcR[i] = sr; arcI[i] = si;
  }
  fitArcCentre(arcR, arcI, L, arcFit);
  let sa = 0, sb = 0, sc = 0;
  for (let i = 0; i < L; i++) {
    const u = arcR[i] - arcFit.mx, v = arcI[i] - arcFit.my;
    sa += u * u; sb += u * v; sc += v * v;
  }
  for (let i = 0; i < L; i++) {
    arcR[i] -= arcFit.cx;
    arcI[i] -= arcFit.cy;
  }
  let prev = 0, cum = 0, lo = 0, hi = 0;
  for (let i = 0; i < L; i++) {
    const ang = Math.atan2(arcI[i], arcR[i]);
    if (i === 0) prev = ang;
    let d = ang - prev;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    prev = ang;
    cum += d;
    out[i] = 1000 * lamM * cum / (4 * Math.PI);
    if (cum < lo) lo = cum;
    if (cum > hi) hi = cum;
  }
  const span = hi - lo;
  const th = 0.5 * Math.atan2(2 * sb, sa - sc);
  const ct = Math.cos(th), st = Math.sin(th);
  for (let i = 0; i < L; i++) outAlt[i] = arcR[i] * ct + arcI[i] * st;
  demean(out, L);
  demean(outAlt, L);
  // Is the curvature real? Compare the arc's bulge away from its chord
  // against how far the samples scatter off the fitted circle. Fit noise
  // and the bulge is no bigger than the scatter — which is what a 1 kHz
  // carrier produces against a 4 mm chest.
  const sagitta = arcFit.R * (1 - Math.cos(Math.min(Math.PI, span) / 2));
  return { n: L, span, ok: arcFit.ok, sagittaOk: sagitta >= MIN_SAGITTA * arcFit.rms };
}

function demean(a, L) {
  let m = 0;
  for (let i = 0; i < L; i++) m += a[i];
  m /= L;
  for (let i = 0; i < L; i++) a[i] -= m;
}

// Picks the trace, and decides whether it carries a millimetre scale.
// S needs { motion, motionAlt, arcR, arcI } and receives
// { motionN, scaled, span, res }.
function resolveMotion(S, zr, zi, L, lamM, alpha, rate) {
  const g = complexMotion(zr, zi, L, lamM, alpha, S.motion, S.motionAlt, S.arcR, S.arcI);
  const resPhase = analyzeArray(S.motion, L, rate);
  const resAlt = analyzeArray(S.motionAlt, L, rate);
  // Prefer the phase reading; take the projection only when it is clearly
  // the more periodic of the two, which is what a sub-λ/8 arc looks like.
  const useAlt = resAlt && (!resPhase || resAlt.conf > resPhase.conf * 1.3);
  if (useAlt) S.motion.set(S.motionAlt.subarray(0, L), 0);
  S.motionN = L;
  S.span = g.span;
  S.res = useAlt ? resAlt : resPhase;
  // Millimetres need the phase branch AND an identifiable centre: either
  // the bulge clears the scatter, or the trace loops far enough and comes
  // out cleanly periodic, which noise cannot fake.
  S.scaled = !useAlt && g.ok && g.span >= MIN_ARC_RAD
    && (g.sagittaOk || (g.span >= 1.5 && S.res && S.res.conf >= BR_CONF));
  return S.res;
}

// ============================================================
//  Render data
//
//  Everything the canvases need, decimated to display resolution and
//  detached from the ring buffers that produced it. The single-device
//  page fills this from its own DSP; the remote host fills the same
//  fields from what the phone sends. Both then call the same drawing
//  code, so the two pages cannot drift apart.
// ============================================================
const R = {
  wave: new Float32Array(1024), waveN: 0, waveMs: 20,
  spec: new Float32Array(1024), specN: 0, specDf: 0, fs: 48000,
  rms: 0, peak: 0, clip: 0, band: NaN,
  cwArcR: new Float32Array(400), cwArcI: new Float32Array(400), cwArcN: 0,
  cwLvlS: new Float32Array(300), cwLvlC: new Float32Array(300), cwLvlN: 0,
  motion: new Float32Array(400), motionN: 0, scaled: false, span: 0,
  envHist: new Float32Array(320), envN: 0, envFill: 0, envLabel: '', envCls: 'idle',
  fmProfile: null, fmSharp: null, fmFine: null,
  fmLock: -1, fm2nd: -1, fmManual: false,
  cmPerCell: 8.6, nCells: 0, minCell: 2, M: 8,
  lam: 0.019, bpm: NaN, conf: 0, res: null
};

// Evenly spaced resample of `n` samples (from `off`) down to at most maxN.
function decimateInto(src, n, dst, maxN, off) {
  off = off || 0;
  const m = Math.min(n, maxN);
  if (m <= 0) return 0;
  if (m === 1) { dst[0] = src[off]; return 1; }
  for (let j = 0; j < m; j++) dst[j] = src[off + Math.round(j * (n - 1) / (m - 1))];
  return m;
}
const ringTmp = { buf: null };
function ringInto(ring, count, dst, maxN) {
  if (!ring) return 0;
  const n = Math.min(ring.filled, count);
  if (n <= 0) return 0;
  if (!ringTmp.buf || ringTmp.buf.length < n) ringTmp.buf = new Float32Array(n);
  ringLast(ring, n, ringTmp.buf);
  return decimateInto(ringTmp.buf, n, dst, maxN);
}

// Fill R from local DSP state. `heavy` also refreshes the scope waveform
// and the received spectrum, which are the expensive parts.
function prepareDraw(heavy) {
  if (heavy) {
    R.fs = scope.fs;
    R.waveN = ringInto(scope.ring, Math.round(R.waveMs / 1000 * scope.fs), R.wave, 1024);
    computeRxSpectrum();
    R.specN = decimateInto(scope.spec, scope.specN, R.spec, 1024);
    R.specDf = R.specN > 1 ? (scope.fs / 2) / (R.specN - 1) : 0;
    R.rms = scope.rms; R.peak = scope.peak; R.clip = scope.clip;
    R.band = bandFraction();
  }
  const S = method === 'cw' ? cw : fm;
  R.motionN = decimateInto(S.motion, S.motionN, R.motion, 400);
  R.scaled = S.scaled; R.span = S.span; R.lam = S.lam || cw.lam;
  R.res = S.res;
  R.bpm = S.res ? S.res.bpm : NaN;
  R.conf = S.res ? S.res.conf : 0;
  const ev = S.ev;
  if (ev) {
    R.envN = ringInto(ev.hist, ev.hist.n, R.envHist, 320);
    R.envFill = ev.hist.filled / ev.hist.n;
    R.envLabel = ev.label; R.envCls = ev.cls;
  }
  if (method === 'cw') {
    const n = Math.min(cw.motionN, Math.round(IQ_HIST_S * cw.bbRate));
    const off = cw.motionN - n;
    R.cwArcN = decimateInto(cw.arcR, n, R.cwArcR, 400, off);
    decimateInto(cw.arcI, n, R.cwArcI, 400, off);
    R.cwLvlN = ringInto(cw.lvlSig, cw.lvlSig ? cw.lvlSig.n : 0, R.cwLvlS, 300);
    ringInto(cw.lvlClu, cw.lvlClu ? cw.lvlClu.n : 0, R.cwLvlC, 300);
  } else {
    R.fmProfile = fm.profile; R.fmSharp = fm.sharp; R.fmFine = fm.fineMag;
    R.fmLock = fm.lockCell; R.fm2nd = fm.prof2nd; R.fmManual = fm.manualLock;
    R.cmPerCell = fm.cmPerCell; R.nCells = fm.nCells; R.minCell = fm.minCell; R.M = fm.M;
  }
}

// ============================================================
//  Method 1 — continuous wave, quadrature demodulation, phase
// ============================================================
const cw = {
  fs: 48000, f0: 18000, lam: 0.019, decim: 512, bbRate: 0,
  loPhase: 0, dPhase: 0, accI: 0, accQ: 0, accN: 0,
  clRe: 0, clIm: 0, haveCl: false, ampEMA: 0, clMag: 0,
  zRe: null, zIm: null, lvlSig: null, lvlClu: null,
  motion: null, motionAlt: null, motionN: 0, scaled: false, span: 0,
  arcR: null, arcI: null,
  res: null, ev: null, txCycles: 0, txLen: 0
};

// Build a looping tone whose buffer holds a whole number of cycles,
// so the loop point is phase-continuous. The rounded frequency is
// what the demodulator uses — transmitter and receiver stay locked.
function cwBuildTx(fs) {
  const target = P.toneHz;
  const len = Math.round(fs * 0.25);
  const cycles = Math.max(1, Math.round(target * len / fs));
  cw.txLen = len;
  cw.txCycles = cycles;
  return cycles * fs / len;   // exact transmitted frequency
}

function cwInit(fs) {
  cw.fs = fs;
  cw.f0 = cwBuildTx(fs);
  cw.lam = C_SOUND / cw.f0;
  cw.decim = 512;
  cw.bbRate = fs / cw.decim;
  cw.loPhase = 0;
  cw.dPhase = 2 * Math.PI * cw.f0 / fs;
  cw.accI = cw.accQ = cw.accN = 0;
  cw.clRe = cw.clIm = 0; cw.haveCl = false;
  cw.ampEMA = 0; cw.clMag = 0;
  const win = Math.round(BR_WIN_S * cw.bbRate) + 8;
  cw.zRe = makeRing(win); cw.zIm = makeRing(win);
  const iqN = Math.round(IQ_HIST_S * cw.bbRate);
  cw.lvlSig = makeRing(iqN); cw.lvlClu = makeRing(iqN);
  cw.motion = new Float32Array(win);
  cw.motionAlt = new Float32Array(win);
  cw.arcR = new Float32Array(win); cw.arcI = new Float32Array(win);
  cw.motionN = 0; cw.scaled = false; cw.span = 0;
  cw.res = null;
  cw.ev = makeEventState();
  $('cwRate').innerHTML = cw.bbRate.toFixed(1) + '<span class="unit"> Hz</span>';
}

// One complex baseband sample. The running clutter estimate is kept
// only for the level readout; the motion trace is rebuilt from the
// whole window at analysis time, where the arc can be fitted.
function cwEmit(zr, zi) {
  const rate = cw.bbRate;
  const aCl = 1 - Math.exp(-1 / (rate * Math.max(0.5, P.clutterTau)));
  if (!cw.haveCl) { cw.clRe = zr; cw.clIm = zi; cw.haveCl = true; }
  else { cw.clRe += aCl * (zr - cw.clRe); cw.clIm += aCl * (zi - cw.clIm); }
  cw.clMag = Math.hypot(cw.clRe, cw.clIm);
  const amp = Math.hypot(zr - cw.clRe, zi - cw.clIm);
  cw.ampEMA += 0.02 * (amp - cw.ampEMA);
  ringPush(cw.zRe, zr); ringPush(cw.zIm, zi);
  ringPush(cw.lvlSig, amp); ringPush(cw.lvlClu, cw.clMag);
}

function cwSamples(input, n) {
  for (let i = 0; i < n; i++) {
    const s = input[i];
    cw.accI += s * Math.cos(cw.loPhase);
    cw.accQ -= s * Math.sin(cw.loPhase);
    cw.loPhase += cw.dPhase;
    if (cw.loPhase > 2 * Math.PI) cw.loPhase -= 2 * Math.PI;
    if (++cw.accN >= cw.decim) {
      cwEmit(cw.accI / cw.decim, cw.accQ / cw.decim);
      cw.accI = cw.accQ = cw.accN = 0;
    }
  }
}

const cwWork = { zr: null, zi: null };
function cwAnalyze() {
  const L = Math.min(cw.zRe.filled, Math.round(BR_WIN_S * cw.bbRate));
  if (L > 16) {
    if (!cwWork.zr || cwWork.zr.length < L) {
      cwWork.zr = new Float32Array(L + 8);
      cwWork.zi = new Float32Array(L + 8);
    }
    ringLast(cw.zRe, L, cwWork.zr);
    ringLast(cw.zIm, L, cwWork.zi);
    const alpha = 1 - Math.exp(-1 / (cw.bbRate * Math.max(0.02, P.smoothTau)));
    resolveMotion(cw, cwWork.zr, cwWork.zi, L, cw.lam, alpha, cw.bbRate);
    const env = envelopeArray(cw.motion, cw.motionN, cw.bbRate);
    updateEvent(cw.ev, env, cw.res ? cw.res.conf : 0);
    paintBadge($('cwEvent'), cw.ev);
  }

  const good = cw.res && cw.res.conf >= BR_CONF;
  $('cwBpm').innerHTML = good
    ? cw.res.bpm.toFixed(1) + '<span class="unit"> /min</span>'
    : '&mdash;<span class="unit"> /min</span>';
  $('cwDisp').innerHTML = (good && cw.scaled)
    ? cw.res.ampPP.toFixed(2) + '<span class="unit"> mm p-p</span>'
    : '&mdash;<span class="unit">' + (good && !cw.scaled ? ' arc too short' : ' mm p-p') + '</span>';
  const snr = cw.clMag > 0 && cw.ampEMA > 0 ? 20 * Math.log10(cw.ampEMA / cw.clMag) : NaN;
  $('cwSnr').innerHTML = isFinite(snr)
    ? snr.toFixed(1) + '<span class="unit"> dB</span>'
    : '&mdash;<span class="unit"> dB</span>';
}

function cwDrawIQ() {
  const cv = $('iqCanvas');
  const { ctx, W, H, dpr } = prep(cv);
  const n = R.cwArcN;
  const cx = W / 2, cy = H / 2;
  const rad = Math.min(W, H) * 0.42;
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath(); ctx.moveTo(cx - rad * 1.15, cy); ctx.lineTo(cx + rad * 1.15, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - rad * 1.15); ctx.lineTo(cx, cy + rad * 1.15); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, rad, 0, 2 * Math.PI); ctx.stroke();
  axisText(ctx, dpr);
  ctx.textAlign = 'left';  ctx.fillText('I', cx + rad * 1.15 - 10 * dpr, cy - 6 * dpr);
  ctx.textAlign = 'center'; ctx.fillText('Q', cx + 9 * dpr, cy - rad * 1.15 + 10 * dpr);
  // fitted arc centre, which is where the static room sits
  ctx.strokeStyle = '#5b6068';
  ctx.lineWidth = 1.4 * dpr;
  const k = 5 * dpr;
  ctx.beginPath();
  ctx.moveTo(cx - k, cy - k); ctx.lineTo(cx + k, cy + k);
  ctx.moveTo(cx - k, cy + k); ctx.lineTo(cx + k, cy - k);
  ctx.stroke();
  if (n < 4) {
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.textAlign = 'center';
    ctx.fillText('collecting…', cx, cy + rad + 14 * dpr);
    return;
  }
  let mx = 1e-12;
  for (let i = 0; i < n; i++) {
    const m = Math.hypot(R.cwArcR[i], R.cwArcI[i]);
    if (m > mx) mx = m;
  }
  const sc = rad / mx;
  ctx.lineWidth = 1.6 * dpr;
  for (let i = 1; i < n; i++) {
    const t = i / (n - 1);
    ctx.strokeStyle = 'rgba(232,185,35,' + (0.08 + 0.9 * t * t).toFixed(3) + ')';
    ctx.beginPath();
    ctx.moveTo(cx + R.cwArcR[i - 1] * sc, cy - R.cwArcI[i - 1] * sc);
    ctx.lineTo(cx + R.cwArcR[i] * sc, cy - R.cwArcI[i] * sc);
    ctx.stroke();
  }
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(cx + R.cwArcR[n - 1] * sc, cy - R.cwArcI[n - 1] * sc, 3 * dpr, 0, 2 * Math.PI);
  ctx.fill();
  axisText(ctx, dpr);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('λ/2 = ' + (R.lam * 500).toFixed(1) + ' mm per turn', 6 * dpr, 6 * dpr);
  ctx.fillText('arc ' + (R.span * 180 / Math.PI).toFixed(0) + '° per breath'
    + (R.scaled ? '' : ' — too short to scale'), 6 * dpr, 20 * dpr);
}

// Two levels on one dB axis: how loud the room's fixed echo is,
// and how loud the part that moves is. If the gold trace sits on
// the floor, there is no target — check the signal panel above.
function cwDrawLevels() {
  const cv = $('cwLevelCanvas');
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 44 * dpr, padR = 8 * dpr, padT = 8 * dpr, padB = 18 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = cw.lvlSig ? cw.lvlSig.filled : 0;
  axisText(ctx, dpr);
  if (n < 4) {
    ctx.textAlign = 'center';
    ctx.fillText('collecting…', W / 2, H / 2);
    return;
  }
  const sig = new Float32Array(n), clu = new Float32Array(n);
  ringLast(cw.lvlSig, n, sig); ringLast(cw.lvlClu, n, clu);
  let hi = -200, lo = 200;
  for (let i = 0; i < n; i++) {
    const a = dbfs(sig[i]), b = dbfs(clu[i]);
    hi = Math.max(hi, a, b); lo = Math.min(lo, a, b);
  }
  lo = Math.max(lo, hi - 80);
  const yOf = d => padT + plotH * (1 - (Math.max(lo, Math.min(hi, d)) - lo) / (hi - lo || 1));
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1 * dpr;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let g = 0; g <= 2; g++) {
    const d = lo + (hi - lo) * g / 2, y = yOf(d);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.fillText(d.toFixed(0) + ' dB', padL - 5 * dpr, y);
  }
  for (const [arr, col] of [[clu, '#5b6068'], [sig, '#E8B923']]) {
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padL + (i / (n - 1)) * plotW, y = yOf(dbfs(arr[i]));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  axisText(ctx, dpr);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('last ' + IQ_HIST_S + ' s', W - padR, H - 4 * dpr);
}

function cwDraw() {
  drawSignalCheck();
  cwDrawLevels();
  cwDrawIQ();
  drawTrace($('cwDispCanvas'), R.motion, R.motionN, {
    color: '#E8B923', symmetric: true, minSpan: R.scaled ? 0.4 : 1e-9,
    spanSec: BR_WIN_S, fmtY: v => R.scaled ? v.toFixed(2) : v.toExponential(0),
    label: R.scaled ? 'mm' : 'arb. units — arc too short for a mm scale'
  });
  drawSpectrum($('cwSpecCanvas'), R.res);
  drawEnvelope($('cwEnvCanvas'));
}

// ============================================================
//  Method 2 — FMCW range cells (ApneaApp)
//
//  Note on bin structure. The sweep repeats every N samples, so a
//  static echo dechirps to a signal that is *periodic* with the
//  sweep. Transforming over M sweeps therefore puts all of its
//  energy on bins that are multiples of M — one comb line per
//  physical range cell, spaced c/2B apart. The M-1 fine bins in
//  between are Doppler bins for that cell. Breathing is far too
//  slow to leave a cell, so the signal we track is the way the
//  energy of a cell rises and falls as the chest moves inside it.
// ============================================================
const fm = {
  fs: 48000, N: 512, M: 8, frameLen: 4096, T: 0, f0: 18000, f1: 20000, B: 2000,
  chirp: null, refCos: null, refSin: null, win: null,
  ring: null, mask: 0, write: 0, nextFrame: 0, synced: false, syncReq: true,
  cmPerCell: 8.6, nCells: 0, minCell: 2,
  histRe: null, histIm: null, fineMag: null, HISTN: 0, fw: 0, ffilled: 0,
  bg: null, mag: null, motion: null, motionAlt: null, motionN: 0, scaled: false, span: 0,
  arcR: null, arcI: null, lam: 0.018,
  frameRate: 0, lastFrameWall: 0, rateEMA: 0,
  profile: null, sharp: null, prof2nd: -1, lockCell: -1, manualLock: false, rangeCm: NaN,
  res: null, ev: null, wf: null, work: null, colQueue: []
};

// Linear sweep, phase-continuous across the loop point: the centre
// frequency is snapped so the sweep spans a whole number of cycles.
function fmBuildChirp(fs) {
  const N = P.chirpN;
  let lo = P.bandLo, hi = P.bandHi;
  if (hi < lo + 250) { hi = lo + 250; P.bandHi = hi; }
  const B = hi - lo;
  const k = Math.max(1, Math.round(((lo + hi) / 2) * N / fs));
  const fc = k * fs / N;
  fm.N = N; fm.B = B;
  fm.f0 = fc - B / 2; fm.f1 = fc + B / 2;
  fm.T = N / fs;
  const x = new Float32Array(N), rc = new Float32Array(N), rs = new Float32Array(N);
  const rate = B / fm.T;
  for (let n = 0; n < N; n++) {
    const t = n / fs;
    const ph = 2 * Math.PI * (fm.f0 * t + 0.5 * rate * t * t);
    x[n] = Math.sin(ph);
    // Mix with e^{+j phi}: an echo delayed by dt beats at
    // +(B/T)·dt, so range lands on positive FFT bins.
    rc[n] = Math.cos(ph);
    rs[n] = Math.sin(ph);
  }
  fm.chirp = x; fm.refCos = rc; fm.refSin = rs;
}

function fmInit(fs) {
  fm.fs = fs;
  fmBuildChirp(fs);
  fm.M = P.chirps;
  fm.frameLen = fm.N * fm.M;
  fm.frameRate = fs / fm.frameLen;
  // One range cell per comb line: spacing is set by bandwidth alone.
  fm.cmPerCell = 100 * C_SOUND / (2 * fm.B);
  fm.nCells = Math.max(6, Math.min(Math.floor((fm.frameLen / 2 - 2) / fm.M),
                                   Math.ceil(200 / fm.cmPerCell) + 2));
  fm.minCell = Math.max(1, Math.ceil(12 / fm.cmPerCell));

  // Hann across the whole frame. Without it the direct-path leak —
  // three orders of magnitude louder than a chest — smears over
  // every cell and buries the target.
  fm.win = new Float32Array(fm.frameLen);
  for (let i = 0; i < fm.frameLen; i++) {
    fm.win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (fm.frameLen - 1));
  }

  const ringSize = 1 << Math.ceil(Math.log2(fm.frameLen * 4));
  fm.ring = new Float32Array(ringSize);
  fm.mask = ringSize - 1;
  fm.write = 0; fm.nextFrame = 0; fm.synced = false; fm.syncReq = true;

  fm.HISTN = Math.ceil(BR_WIN_S * fm.frameRate) + 8;
  // Keep the complex cell value, not just its magnitude: the range
  // search wants magnitude, but the rate readout wants phase.
  fm.histRe = new Float32Array(fm.nCells * fm.HISTN);
  fm.histIm = new Float32Array(fm.nCells * fm.HISTN);
  fm.motion = new Float32Array(fm.HISTN);
  fm.motionAlt = new Float32Array(fm.HISTN);
  fm.arcR = new Float32Array(fm.HISTN); fm.arcI = new Float32Array(fm.HISTN);
  fm.motionN = 0; fm.scaled = false; fm.span = 0;
  fm.lam = C_SOUND / ((fm.f0 + fm.f1) / 2);
  fm.fw = 0; fm.ffilled = 0;
  fm.bg = new Float32Array(fm.nCells);
  fm.mag = new Float32Array(fm.nCells);
  fm.profile = new Float32Array(fm.nCells);
  fm.sharp = new Float32Array(fm.nCells);
  fm.lockCell = -1; fm.manualLock = false; fm.prof2nd = -1; fm.rangeCm = NaN;
  fm.res = null;
  fm.ev = makeEventState();
  fm.work = { re: new Float64Array(fm.frameLen), im: new Float64Array(fm.frameLen) };
  // Every FFT bin up to the last cell, kept for the diagnostic plot.
  fm.fineMag = new Float32Array(Math.min(fm.frameLen / 2, fm.nCells * fm.M + fm.M));
  fm.rateEMA = 0; fm.lastFrameWall = 0;
  fm.colQueue = [];

  $('fmBinStat').innerHTML = fm.cmPerCell.toFixed(1) + '<span class="unit"> cm</span>';
  $('fmFrameStat').innerHTML = fm.frameRate.toFixed(1) + '<span class="unit"> Hz</span>';
  fmSetupWaterfall();
}

// Cross-correlate the newest 2N samples against one sweep; the peak
// lag is where a sweep begins in the ring. Run once at start (and on
// demand) — the frame grid must stay fixed or slow-time analysis
// sees jumps that are not breathing.
const fmSync = { Xre: null, Xim: null, L: 0 };
function fmFindSync(end) {
  const N = fm.N, L = 2 * N;
  if (fmSync.L !== L) {
    fmSync.L = L;
    fmSync.Xre = new Float64Array(L);
    fmSync.Xim = new Float64Array(L);
  }
  fmSync.Xre.fill(0); fmSync.Xim.fill(0);
  for (let n = 0; n < N; n++) fmSync.Xre[n] = fm.chirp[n];
  fft(fmSync.Xre, fmSync.Xim, false);
  const Yre = new Float64Array(L), Yim = new Float64Array(L);
  const start = end - L;
  for (let n = 0; n < L; n++) Yre[n] = fm.ring[(start + n) & fm.mask];
  fft(Yre, Yim, false);
  for (let k = 0; k < L; k++) {
    const a = Yre[k], b = Yim[k], c = fmSync.Xre[k], d = -fmSync.Xim[k];
    Yre[k] = a * c - b * d;
    Yim[k] = a * d + b * c;
  }
  fft(Yre, Yim, true);
  let best = 0, bestV = -1;
  for (let n = 0; n < N; n++) {
    const m = Yre[n] * Yre[n] + Yim[n] * Yim[n];
    if (m > bestV) { bestV = m; best = n; }
  }
  return start + best;
}

// Dechirp M sweeps and transform: delay becomes beat frequency, so
// the comb line at bin c·M is the range cell at c·(c_sound/2B).
function fmProcessFrame(startAbs) {
  const { re, im } = fm.work;
  const N = fm.N, FL = fm.frameLen;
  for (let m = 0; m < FL; m++) {
    const s = fm.ring[(startAbs + m) & fm.mask] * fm.win[m];
    const p = m % N;
    re[m] = s * fm.refCos[p];
    im[m] = s * fm.refSin[p];
  }
  fft(re, im, false);
  for (let k = 0; k < fm.fineMag.length; k++) fm.fineMag[k] = Math.hypot(re[k], im[k]) / FL;
  const mag = fm.mag;
  const aBg = 1 - Math.exp(-1 / (fm.frameRate * 4));
  const f = fm.fw;
  for (let c = 0; c < fm.nCells; c++) {
    const k = c * fm.M;
    const zr = re[k] / FL, zi = im[k] / FL;
    mag[c] = Math.hypot(zr, zi);
    fm.histRe[c * fm.HISTN + f] = zr;
    fm.histIm[c * fm.HISTN + f] = zi;
    fm.bg[c] += aBg * (mag[c] - fm.bg[c]);
  }
  fm.fw = (fm.fw + 1) % fm.HISTN;
  if (fm.ffilled < fm.HISTN) fm.ffilled++;

  const now = performance.now();
  if (fm.lastFrameWall) {
    const r = 1000 / (now - fm.lastFrameWall);
    fm.rateEMA = fm.rateEMA ? fm.rateEMA + 0.05 * (r - fm.rateEMA) : r;
  }
  fm.lastFrameWall = now;
  // Queue the column; the page drains it — the local page straight into
  // the waterfall, the remote phone into its next packet.
  const dev = new Float32Array(fm.nCells);
  for (let c = 0; c < fm.nCells; c++) dev[c] = Math.abs(mag[c] - fm.bg[c]);
  fm.colQueue.push(dev);
  if (fm.colQueue.length > 240) fm.colQueue.shift();
}

function fmSamples(input, n) {
  const ring = fm.ring, mask = fm.mask;
  for (let i = 0; i < n; i++) ring[(fm.write + i) & mask] = input[i];
  fm.write += n;

  if (fm.syncReq && fm.write > 4 * fm.N) {
    const lag = fmFindSync(fm.write);
    const delta = fm.write - lag;
    fm.nextFrame = lag + Math.ceil(delta / fm.N) * fm.N;
    fm.syncReq = false;
    fm.synced = true;
  }
  if (!fm.synced) return;
  while (fm.nextFrame + fm.frameLen <= fm.write
         && fm.write - fm.nextFrame < fm.ring.length - fm.frameLen) {
    fmProcessFrame(fm.nextFrame);
    fm.nextFrame += fm.frameLen;
  }
}

// Chest motion for one cell, read from phase rather than magnitude.
// ApneaApp watches a bin's amplitude rise and fall; that works, but
// it folds once the chest sweeps past half a wavelength (18 mm here),
// reporting double the true rate. The phase of the same complex bin
// does not fold, and — when the arc is long enough to fit a circle
// to — comes out in millimetres.
const fmWork2 = { zr: null, zi: null };
function fmCellMotion(cell, L) {
  const base = cell * fm.HISTN;
  if (!fmWork2.zr || fmWork2.zr.length < L) {
    fmWork2.zr = new Float32Array(L + 8);
    fmWork2.zi = new Float32Array(L + 8);
  }
  let idx = (fm.fw - L + fm.HISTN) % fm.HISTN;
  for (let i = 0; i < L; i++) {
    fmWork2.zr[i] = fm.histRe[base + idx];
    fmWork2.zi[i] = fm.histIm[base + idx];
    idx = idx + 1 === fm.HISTN ? 0 : idx + 1;
  }
  const alpha = 1 - Math.exp(-1 / (fm.frameRate * 0.4));
  resolveMotion(fm, fmWork2.zr, fmWork2.zi, L, fm.lam, alpha, fm.frameRate);
}

// Per-cell breathing test: absolute strength of the strongest peak
// in the respiration band, plus how sharp that peak is against the
// rest of the band. Strength alone would follow the loudest echo;
// sharpness alone would happily lock onto numerical noise.
const scanWork = { re: null, im: null, x: null, NFFT: 0 };
function fmScanCell(cell, L, out) {
  const base = cell * fm.HISTN;
  if (!scanWork.x || scanWork.x.length < L) scanWork.x = new Float32Array(L + 8);
  const x = scanWork.x;
  let idx = (fm.fw - L + fm.HISTN) % fm.HISTN;
  let mean = 0;
  for (let i = 0; i < L; i++) {
    x[i] = Math.hypot(fm.histRe[base + idx], fm.histIm[base + idx]);
    mean += x[i];
    idx = idx + 1 === fm.HISTN ? 0 : idx + 1;
  }
  mean /= L;
  let NFFT = 1;
  while (NFFT < L * 2) NFFT <<= 1;
  if (scanWork.NFFT !== NFFT) {
    scanWork.NFFT = NFFT;
    scanWork.re = new Float64Array(NFFT);
    scanWork.im = new Float64Array(NFFT);
  }
  const re = scanWork.re, im = scanWork.im;
  re.fill(0); im.fill(0);
  for (let i = 0; i < L; i++) {
    const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (L - 1));
    re[i] = (x[i] - mean) * w;
  }
  fft(re, im, false);
  const df = fm.frameRate / NFFT;
  const kLo = Math.max(1, Math.ceil(BR_LO_HZ / df));
  const kHi = Math.min(NFFT / 2 - 2,
                       Math.floor(Math.min(BR_HI_HZ, 0.45 * fm.frameRate) / df));
  if (kHi <= kLo) { out.amp = 0; out.sharp = 0; out.mean = mean; return; }
  let peak = 0, sum = 0;
  for (let k = kLo; k <= kHi; k++) {
    const m = Math.hypot(re[k], im[k]);
    sum += m;
    if (m > peak) peak = m;
  }
  // Strength counts the harmonics too: a cell right on the chest
  // swings through so much phase that its magnitude folds, moving
  // energy out of the fundamental and into 2f, 3f.
  const kH = Math.min(NFFT / 2 - 2, Math.floor((BR_HI_HZ * 2) / df));
  let energy = 0;
  for (let k = kLo; k <= kH; k++) energy += re[k] * re[k] + im[k] * im[k];
  const bandMean = sum / (kHi - kLo + 1);
  out.amp = Math.sqrt(energy) / L;
  out.sharp = bandMean > 0 ? peak / bandMean : 0;
  out.mean = mean;
}

const scanOut = { amp: 0, sharp: 0, mean: 0 };
function fmAnalyze() {
  const L = Math.min(fm.ffilled, Math.round(BR_WIN_S * fm.frameRate));
  if (L < Math.max(24, fm.frameRate * 8)) return;
  // ApneaApp scans outward from the phone until a cell shows a
  // breathing peak; we score every cell so the search is visible.
  const amps = new Float32Array(fm.nCells);
  const means = new Float32Array(fm.nCells);
  const list = [];
  for (let c = fm.minCell; c < fm.nCells; c++) {
    fmScanCell(c, L, scanOut);
    amps[c] = scanOut.amp;
    means[c] = scanOut.mean;
    fm.sharp[c] = scanOut.sharp;
    list.push(scanOut.amp);
  }
  list.sort((a, b) => a - b);
  const medAmp = list.length ? list[Math.floor(list.length / 2)] : 0;
  // A cell is a candidate only if its breathing peak is sharp; the
  // score then ranks candidates by how much motion energy they hold
  // relative to a typical cell in the room.
  let best = -1, bestV = 0;
  for (let c = fm.minCell; c < fm.nCells; c++) {
    const score = amps[c] / (medAmp + 1e-12);
    fm.profile[c] = fm.profile[c] * 0.5 + score * 0.5;
    if (fm.sharp[c] >= BR_CONF && fm.profile[c] > bestV) { bestV = fm.profile[c]; best = c; }
  }
  // A real second subject stands a clear 35 cm from the first —
  // anything closer is that subject's own spill into neighbouring
  // cells, which tracks its rate exactly and would double-count it.
  let second = -1, secondV = 0;
  const guard = Math.max(4, Math.ceil(35 / fm.cmPerCell));
  for (let c = fm.minCell; c < fm.nCells; c++) {
    if (best >= 0 && Math.abs(c - best) < guard) continue;
    if (fm.sharp[c] >= BR_CONF && fm.profile[c] > secondV) { secondV = fm.profile[c]; second = c; }
  }
  fm.prof2nd = secondV > Math.max(1.5, 0.35 * bestV) ? second : -1;

  if (!fm.manualLock && best >= 0) fm.lockCell = best;
  if (fm.lockCell < fm.minCell || fm.lockCell >= fm.nCells) fm.lockCell = best;
  if (fm.lockCell < 0) return;

  // Cells are 8.6 cm apart; interpolate the neighbours by echo
  // strength for a range estimate finer than one cell.
  const lo = Math.max(fm.minCell, fm.lockCell - 1);
  const hi = Math.min(fm.nCells - 1, fm.lockCell + 1);
  let wsum = 0, num = 0;
  for (let c = lo; c <= hi; c++) { wsum += means[c]; num += means[c] * c; }
  fm.rangeCm = (wsum > 0 ? num / wsum : fm.lockCell) * fm.cmPerCell;

  fmCellMotion(fm.lockCell, L);
  const env = envelopeArray(fm.motion, fm.motionN, fm.frameRate);
  updateEvent(fm.ev, env, fm.res ? fm.res.conf : 0);
  paintBadge($('fmEvent'), fm.ev);

  if (fm.res && fm.res.conf >= BR_CONF) {
    $('fmBpm').innerHTML = fm.res.bpm.toFixed(1) + '<span class="unit"> /min</span>';
    $('fmRangeStat').innerHTML = fm.rangeCm.toFixed(0) + '<span class="unit"> cm</span>';
    $('fmDisp').innerHTML = fm.scaled
      ? fm.res.ampPP.toFixed(2) + '<span class="unit"> mm p-p</span>'
      : '&mdash;<span class="unit"> arc too short</span>';
  } else {
    $('fmBpm').innerHTML = '&mdash;<span class="unit"> /min</span>';
    $('fmDisp').innerHTML = '&mdash;<span class="unit"> mm p-p</span>';
  }
  $('fmFrameStat').innerHTML = (fm.rateEMA || fm.frameRate).toFixed(1) + '<span class="unit"> Hz</span>';
}

// ---- waterfall ----
function fmSetupWaterfall() {
  const cv = $('fmWaterfall');
  // The phone has no canvases; it only computes.
  if (!cv) { fm.wf = null; return; }
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(200, Math.floor((cv.clientWidth || 800) * dpr));
  const H = Math.max(120, Math.floor((cv.clientHeight || 320) * dpr));
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, W, H);
  fm.wf = { ctx, W, H, dpr, gutter: Math.round(40 * dpr) };
}

function fmColor(v) {
  v = Math.max(0, Math.min(1, v));
  if (v < 0.33) { const t = v / 0.33; return [Math.round(188 * t), Math.round(18 * t), Math.round(42 * t)]; }
  if (v < 0.66) { const t = (v - 0.33) / 0.33; return [188 + Math.round(44 * t), 18 + Math.round(167 * t), 42 - Math.round(7 * t)]; }
  const t = (v - 0.66) / 0.34;
  return [232 + Math.round(23 * t), 185 + Math.round(70 * t), 35 + Math.round(220 * t)];
}

// dev[] is |cell magnitude − its slow background|, one value per cell.
function fmDrawColumn(dev) {
  if (!fm.wf) return;
  const { ctx, W, H, dpr, gutter } = fm.wf;
  const plotW = W - gutter;
  if (plotW < 8) return;
  const img = ctx.getImageData(gutter + 1, 0, plotW - 1, H);
  ctx.putImageData(img, gutter, 0);
  const col = ctx.createImageData(1, H);
  const maxCm = P.rangeCm;
  const maxCell = Math.min(R.nCells - 1, Math.max(3, Math.ceil(maxCm / R.cmPerCell)));
  // Normalise the fluctuation by the strongest deviation on screen.
  let norm = 1e-12;
  for (let c = R.minCell; c <= maxCell; c++) if (dev[c] > norm) norm = dev[c];
  for (let y = 0; y < H; y++) {
    const r = (H - 1 - y) / (H - 1);
    const cf = r * maxCell;
    const c0 = Math.floor(cf), c1 = Math.min(maxCell, c0 + 1), f = cf - c0;
    const d0 = dev[c0], d1 = dev[c1];
    const v = (d0 * (1 - f) + d1 * f) / norm;
    const [R, G, Bc] = fmColor(Math.pow(v, 0.6));
    const i = y * 4;
    col.data[i] = R; col.data[i + 1] = G; col.data[i + 2] = Bc; col.data[i + 3] = 255;
  }
  ctx.putImageData(col, W - 1, 0);
  // locked-cell tick on the newest column, leaving a scrolling trail
  if (R.fmLock >= 0 && R.fmLock <= maxCell) {
    const y = Math.round((H - 1) * (1 - R.fmLock / maxCell));
    ctx.fillStyle = '#5cd7e0';
    ctx.fillRect(W - 2, y - 1, 2, 2);
  }
  // range gutter, repainted every frame so it never scrolls away
  ctx.fillStyle = '#0c0c0e';
  ctx.fillRect(0, 0, gutter, H);
  ctx.font = (10 * dpr) + 'px "Roboto Mono", monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const step = maxCm > 120 ? 40 : 20;
  for (let cm = 0; cm <= maxCm; cm += step) {
    const y = (H - 1) * (1 - (cm / R.cmPerCell) / maxCell);
    if (y < 6 * dpr || y > H - 4 * dpr) continue;
    ctx.fillText(cm + ' cm', gutter - 5 * dpr, y);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath(); ctx.moveTo(gutter, y); ctx.lineTo(W, y); ctx.stroke();
  }
}

function fmDrawProfile() {
  const cv = $('fmProfCanvas');
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 34 * dpr, padR = 8 * dpr, padT = 10 * dpr, padB = 20 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxCm = P.rangeCm;
  const maxCell = Math.min(R.nCells - 1, Math.max(3, Math.ceil(maxCm / R.cmPerCell)));
  let hi = 4;
  for (let c = R.minCell; c <= maxCell; c++) if (R.fmProfile[c] > hi) hi = R.fmProfile[c];
  const xOf = c => padL + (c / maxCell) * plotW;
  const yOf = v => padT + plotH - Math.max(0, Math.min(1, v / hi)) * plotH;
  axisText(ctx, dpr);
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1 * dpr;
  ctx.textAlign = 'center';
  const step = maxCm > 120 ? 40 : 20;
  for (let cm = 0; cm <= maxCm; cm += step) {
    const x = xOf(cm / R.cmPerCell);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.fillText(cm + ' cm', x, H - 6 * dpr);
  }
  // one bar per range cell — the profile really is this coarse
  const bw = Math.max(2 * dpr, (plotW / (maxCell + 1)) * 0.6);
  for (let c = R.minCell; c <= maxCell; c++) {
    const x = xOf(c), y = yOf(R.fmProfile[c]);
    ctx.fillStyle = R.fmSharp[c] > 3 ? '#bc122a' : 'rgba(188,18,42,0.35)';
    ctx.fillRect(x - bw / 2, y, bw, padT + plotH - y);
  }
  const mark = (c, col, txt) => {
    if (c < 0 || c > maxCell) return;
    const x = xOf(c);
    ctx.strokeStyle = col;
    ctx.setLineDash([4 * dpr, 3 * dpr]);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = col;
    ctx.font = 'bold ' + (10 * dpr) + 'px "Roboto Mono", monospace';
    ctx.textAlign = x > W * 0.7 ? 'right' : 'left';
    ctx.fillText(txt + ' ' + (c * R.cmPerCell).toFixed(0) + ' cm', x + (x > W * 0.7 ? -5 : 5) * dpr, padT + 11 * dpr);
  };
  mark(R.fm2nd, '#E8B923', '2nd');
  mark(R.fmLock, '#5cd7e0', R.fmManual ? 'locked' : 'auto');
}

// Pages override this: locally it moves the lock, remotely it asks the
// phone to move it.
let onLockCell = c => {
  fm.lockCell = c; fm.manualLock = c >= 0;
  fm.res = null; fm.ev = makeEventState();
};
if ($('fmProfCanvas')) $('fmProfCanvas').addEventListener('click', ev => {
  if (!R.nCells) return;
  const rect = ev.currentTarget.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = ev.currentTarget.width;
  const padL = 34 * dpr, padR = 8 * dpr;
  const px = (ev.clientX - rect.left) * (W / rect.width);
  const maxCm = P.rangeCm;
  const maxCell = Math.min(R.nCells - 1, Math.max(3, Math.ceil(maxCm / R.cmPerCell)));
  const c = Math.round(((px - padL) / (W - padL - padR)) * maxCell);
  onLockCell(c < R.minCell || c > maxCell ? -1 : c);
});

// Every bin of the dechirped frame, not just the comb lines we
// read. A still scene puts energy only on the cyan ticks; that is
// the periodicity argument made visible.
function fmDrawRawProfile() {
  const cv = $('rawProfCanvas');
  const { ctx, W, H, dpr } = prep(cv);
  const padL = 44 * dpr, padR = 8 * dpr, padT = 10 * dpr, padB = 20 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxCm = P.rangeCm;
  const maxCell = Math.min(R.nCells - 1, Math.max(3, Math.ceil(maxCm / R.cmPerCell)));
  const kMax = Math.min(R.fmFine ? R.fmFine.length - 1 : 0, maxCell * R.M);
  const cmPerFine = R.cmPerCell / R.M;
  axisText(ctx, dpr);
  if (kMax < 4) {
    ctx.textAlign = 'center';
    ctx.fillText('waiting for a frame…', W / 2, H / 2);
    return;
  }
  let hi = -200;
  for (let k = 0; k <= kMax; k++) hi = Math.max(hi, dbfs(R.fmFine[k]));
  const lo = hi - 80;
  const xOf = k => padL + (k / kMax) * plotW;
  const yOf = d => padT + plotH * (1 - (Math.max(lo, Math.min(hi, d)) - lo) / (hi - lo));
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1 * dpr;
  ctx.textAlign = 'center';
  const step = maxCm > 120 ? 40 : 20;
  for (let cm = 0; cm <= maxCm; cm += step) {
    const x = xOf(cm / cmPerFine);
    if (x > W - padR) continue;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.fillText(cm + ' cm', x, H - 6 * dpr);
  }
  // cell ticks along the bottom
  for (let c = 0; c <= maxCell; c++) {
    const x = xOf(c * R.M);
    ctx.strokeStyle = c === R.fmLock ? '#E8B923' : 'rgba(92,215,224,0.55)';
    ctx.lineWidth = (c === R.fmLock ? 2 : 1) * dpr;
    ctx.beginPath();
    ctx.moveTo(x, padT + plotH);
    ctx.lineTo(x, padT + plotH - (c === R.fmLock ? 10 : 6) * dpr);
    ctx.stroke();
  }
  ctx.strokeStyle = '#bc122a';
  ctx.lineWidth = 1.4 * dpr;
  ctx.beginPath();
  for (let k = 0; k <= kMax; k++) {
    const x = xOf(k), y = yOf(dbfs(R.fmFine[k]));
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  axisText(ctx, dpr);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('peak ' + hi.toFixed(0) + ' dB, floor ' + lo.toFixed(0) + ' dB', padL + 6 * dpr, padT + 2 * dpr);
}

function fmDraw() {
  drawSignalCheck();
  fmDrawRawProfile();
  fmDrawProfile();
  if (R.fmLock >= 0 && R.motionN > 4) {
    drawTrace($('fmMotionCanvas'), R.motion, R.motionN, {
      color: '#5cd7e0', symmetric: true, minSpan: R.scaled ? 0.4 : 1e-9,
      spanSec: BR_WIN_S, fmtY: v => R.scaled ? v.toFixed(2) : v.toExponential(0),
      label: R.scaled ? 'mm, from the phase of the locked cell'
                      : 'arb. units — arc too short for a mm scale'
    });
  } else {
    drawTrace($('fmMotionCanvas'), new Float32Array(2), 0, { waiting: 'searching for a breathing cell…' });
  }
  drawSpectrum($('fmSpecCanvas'), R.res);
  drawEnvelope($('fmEnvCanvas'));
}

// ============================================================
//  Demo signal — synthesises the received waveform (direct leak,
//  a static wall, and one breathing chest) and feeds it through
//  the identical processing chain. No microphone involved.
// ============================================================
const demo = { n: 0, t0: 0, timer: null, chunk: new Float32Array(4096) };
function smoothstep(x) { return x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x); }
function demoDist(t) {
  const CYCLE = 45;
  const ph = t % CYCLE;
  // 4 mm of chest travel at 15 breaths/min, with a 10 s hold each cycle
  const hold = smoothstep(ph - 34) - smoothstep(ph - 44);
  const amp = 0.004 * (1 - hold);
  return 0.45 + amp * Math.sin(2 * Math.PI * 0.25 * t);
}
function wrap(u, n) { const m = u % n; return m < 0 ? m + n : m; }
function chirpAt(u) {
  const N = fm.N;
  const p = wrap(u, N);
  const i0 = Math.floor(p), i1 = (i0 + 1) % N, f = p - i0;
  return fm.chirp[i0] * (1 - f) + fm.chirp[i1] * f;
}
function demoGenerate(out, count, fs) {
  const noise = 0.003;
  if (method === 'cw') {
    const len = cw.txLen, w = cw.dPhase;
    const tone = u => Math.sin(w * wrap(u, len));
    for (let i = 0; i < count; i++) {
      const n = demo.n + i, t = n / fs;
      const dChest = demoDist(t);
      out[i] = 0.50 * tone(n - 2 * 0.03 / C_SOUND * fs)
             + 0.10 * tone(n - 2 * 1.35 / C_SOUND * fs)
             + 0.045 * tone(n - 2 * dChest / C_SOUND * fs)
             + noise * (Math.random() * 2 - 1);
    }
  } else {
    for (let i = 0; i < count; i++) {
      const n = demo.n + i, t = n / fs;
      const dChest = demoDist(t);
      out[i] = 0.50 * chirpAt(n - 2 * 0.03 / C_SOUND * fs)
             + 0.12 * chirpAt(n - 2 * 1.35 / C_SOUND * fs)
             + 0.055 * chirpAt(n - 2 * dChest / C_SOUND * fs)
             + noise * (Math.random() * 2 - 1);
    }
  }
  demo.n += count;
}
function demoTick() {
  const fs = 48000;
  const want = Math.round((performance.now() - demo.t0) / 1000 * fs);
  let todo = Math.min(want - demo.n, fs * 0.3);
  while (todo > 0) {
    const c = Math.min(todo, demo.chunk.length);
    demoGenerate(demo.chunk, c, fs);
    scopePush(demo.chunk, c);
    if (method === 'cw') cwSamples(demo.chunk, c); else fmSamples(demo.chunk, c);
    todo -= c;
  }
}

// ============================================================
//  Audio engine
// ============================================================
const eng = { ctx: null, src: null, gain: null, mic: null, sp: null, stream: null, running: false, demoMode: false };

// Page hooks. The defaults do nothing, so a page can ignore any of them.
let onStatus = () => {};
let onEngine = () => {};
let onLayout = () => {};
function setStatus(msg, kind) { onStatus(msg, kind); }

function buildTxBuffer(ctx) {
  const fs = ctx.sampleRate;
  if (method === 'cw') {
    const buf = ctx.createBuffer(1, cw.txLen, fs);
    const ch = buf.getChannelData(0);
    for (let n = 0; n < cw.txLen; n++) ch[n] = Math.sin(cw.dPhase * n);
    return buf;
  }
  const tiles = Math.max(8, Math.ceil(fs / fm.N));
  const buf = ctx.createBuffer(1, tiles * fm.N, fs);
  const ch = buf.getChannelData(0);
  for (let t = 0; t < tiles; t++) ch.set(fm.chirp, t * fm.N);
  return buf;
}

async function engStart(demoMode) {
  try {
    if (eng.running) engStop();
    eng.demoMode = !!demoMode;

    if (eng.demoMode) {
      const fs = 48000;
      scopeReset(fs);
      if (method === 'cw') cwInit(fs); else fmInit(fs);
      onLayout();
      if (method === 'fmcw') fmSetupWaterfall();
      demo.n = 0;
      demo.t0 = performance.now();
      demo.timer = setInterval(demoTick, 25);
      eng.running = true;
      setStatus('Demo signal — synthetic echo, no microphone in use.', 'ok');
      onEngine(true);
      return;
    }

    setStatus('Setting up audio…', 'busy');
    // Created on a user gesture and then kept: a phone will not hand out
    // a fresh context without another tap.
    if (!eng.ctx) eng.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (eng.ctx.state === 'suspended') await eng.ctx.resume();
    const fs = eng.ctx.sampleRate;
    scopeReset(fs);
    if (method === 'cw') cwInit(fs); else fmInit(fs);

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
    eng.sp = eng.ctx.createScriptProcessor(1024, 1, 1);
    eng.sp.onaudioprocess = ev => {
      if (!eng.running) return;
      const inp = ev.inputBuffer.getChannelData(0);
      scopePush(inp, inp.length);
      if (method === 'cw') cwSamples(inp, inp.length); else fmSamples(inp, inp.length);
    };
    eng.mic.connect(eng.sp);
    const sink = eng.ctx.createGain();
    sink.gain.value = 0;
    eng.sp.connect(sink).connect(eng.ctx.destination);

    onLayout();
    if (method === 'fmcw') fmSetupWaterfall();
    eng.src.start();
    eng.running = true;
    setStatus('Listening at ' + Math.round(fs) + ' Hz — sit still and breathe normally. Give it 30 s to fill the window.', 'ok');
    onEngine(true);
  } catch (err) {
    console.error(err);
    setStatus('Could not start: ' + err.message, 'error');
    engStop();
  }
}

function engStop() {
  eng.running = false;
  if (demo.timer) { clearInterval(demo.timer); demo.timer = null; }
  try { eng.src && eng.src.stop(); } catch (e) {}
  for (const node of [eng.sp, eng.mic, eng.gain, eng.src]) {
    if (node) try { node.disconnect(); } catch (e) {}
  }
  if (eng.stream) eng.stream.getTracks().forEach(t => t.stop());
  eng.src = eng.gain = eng.mic = eng.sp = eng.stream = null;
  setStatus('Stopped.', '');
  onEngine(false);
}
