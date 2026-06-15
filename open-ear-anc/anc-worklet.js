/* Adaptive open-ear ANC canceller, running on the audio render thread (AudioWorklet).
 *
 * This is the same multichannel normalized-LMS virtual-sensing canceller that previously
 * ran in a main-thread ScriptProcessorNode — moved here so the per-sample DSP no longer
 * competes with rendering/GC on the main thread (no more underrun clicks under load).
 *
 * The mono noise comes in; we synthesize what each ACTIVE frame mic picks up (the source
 * delayed by its distance + independent sensor noise), an NLMS FIR per mic reconstructs the
 * ear sound, and the residual ear − (anti-noise applied late by the processing latency) is
 * the output. Scene/parameter changes arrive via port messages.
 *
 * IMPORTANT: process() must allocate NOTHING (it runs ~375×/s on the audio thread; any
 * garbage triggers GC pauses that crackle/distort the output). All geometry is recomputed
 * only when parameters change, into preallocated buffers.
 */

const C = 343;                                   // speed of sound (m/s)
// glasses frame mic positions (meters; face forward = +y) — must match the main thread
const MICS = [
  [-0.075, 0.060], [-0.030, 0.082], [0.030, 0.082], [0.075, 0.060],
  [-0.090, 0.010], [-0.090, -0.045], [0.090, 0.010], [0.090, -0.045]
];
const EAR0 = [-0.095, -0.020];                   // left (target) ear
const EAR_R = [-EAR0[0], EAR0[1]];               // mirrored right ear
const _byDist = (ear) => (a, b) =>
  Math.hypot(MICS[a][0] - ear[0], MICS[a][1] - ear[1]) - Math.hypot(MICS[b][0] - ear[0], MICS[b][1] - ear[1]);
const LEFT_MICS = MICS.map((m, i) => i).filter(i => MICS[i][0] < 0).sort(_byDist(EAR0));
const RIGHT_MICS = MICS.map((m, i) => i).filter(i => MICS[i][0] > 0).sort(_byDist(EAR_R));

class AdaptiveAncProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.T = 40; this.NMAX = 8; this.Mx = 1024; this.Mm = 128;   // taps/mic, max mics, ring sizes
    this.xbuf = new Float32Array(this.Mx);        // incoming noise (source signal)
    this.w = new Float32Array(this.NMAX * this.T);// per-mic FIR weights (flattened)
    this.mbuf = new Float32Array(this.NMAX * this.Mm); // each active mic's recent signal
    this.Yh = new Float32Array(this.Mx);          // anti-noise estimate history (for latency delay)
    this.MU = 0.5; this.LEAK = 1e-4; this.SIG = 0.5;
    this.xpos = 0; this.mpos = 0; this.ypos = 0; this.curN = -1;
    // scene/params, updated from the main thread
    this.algo = 'adaptive'; this.ancOn = false; this.latencyMs = 0.1;
    this.theta = 40; this.dist = 1.5; this.nMics = 4; this.calibrated = false;
    // secondary path (speaker->ear) mismatch: amplitude error gS + a small irreducible group
    // delay tauSamp. The delay's phase error grows with frequency, so high tones (≳2 kHz) can't
    // be cancelled no matter how low the processing latency is. Calibration shrinks both.
    this.gS = 0.86; this.tauSamp = 0;
    // preallocated geometry caches (filled by updateGeom on parameter change — never in process)
    this.idx = new Int32Array(this.NMAX);
    this.di0 = new Int32Array(this.NMAX);
    this.dif = new Float32Array(this.NMAX);
    this.N = 0; this.tg0 = 0; this.tgf = 0; this.latSamp = 5;
    this.updateGeom();
    this.port.onmessage = (e) => { Object.assign(this, e.data); this.updateGeom(); };
  }

  // Recompute mic/ear delays for the current scene — called only on parameter change.
  updateGeom() {
    const SR = sampleRate, perSide = this.nMics;
    this.gS = this.calibrated ? 0.97 : 0.86;                       // secondary-path amplitude match
    this.tauSamp = (this.calibrated ? 10e-6 : 40e-6) * SR;         // irreducible secondary-path group delay
    const th = this.theta * Math.PI / 180, Px = this.dist * Math.sin(th), Py = this.dist * Math.cos(th);
    this.latSamp = Math.min(this.Mx - 1, Math.round(Math.max(0, this.latencyMs * 1e-3) * SR));
    // active mic indices (left then right, nearest-ear first)
    let N = 0;
    for (let s = 0; s < perSide && s < LEFT_MICS.length; s++) this.idx[N++] = LEFT_MICS[s];
    for (let s = 0; s < perSide && s < RIGHT_MICS.length; s++) this.idx[N++] = RIGHT_MICS[s];
    this.N = N;
    const dEar = Math.hypot(EAR0[0] - Px, EAR0[1] - Py);
    let dmin = dEar;
    for (let i = 0; i < N; i++) { const m = MICS[this.idx[i]]; const d = Math.hypot(m[0] - Px, m[1] - Py); if (d < dmin) dmin = d; }
    const dE = (dEar - dmin) / C * SR;
    let mld = 0;
    for (let i = 0; i < N; i++) {
      const m = MICS[this.idx[i]]; const d = (Math.hypot(m[0] - Px, m[1] - Py) - dmin) / C * SR;
      this.di0[i] = d | 0; this.dif[i] = d - (d | 0);
      const rel = d - dE; if (rel > mld) mld = rel;
    }
    const tgt = dE + mld; this.tg0 = tgt | 0; this.tgf = tgt - this.tg0;
    if (N !== this.curN) { this.w.fill(0); this.curN = N; }   // active-mic set changed → reset filters
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    const inp = inputs[0] && inputs[0][0];
    if (!inp || this.algo !== 'adaptive') { out.fill(0); return true; }

    const en = this.ancOn ? 1 : 0, Nn = out.length;
    const T = this.T, Mx = this.Mx, Mm = this.Mm, w = this.w, xbuf = this.xbuf, mbuf = this.mbuf, Yh = this.Yh;
    const MU = this.MU, LEAK = this.LEAK, SIG = this.SIG;
    const N = this.N, di0 = this.di0, dif = this.dif, tg0 = this.tg0, tgf = this.tgf, latSamp = this.latSamp;
    const gS = this.gS, tauSamp = this.tauSamp;
    let xpos = this.xpos, mpos = this.mpos, ypos = this.ypos;

    for (let n = 0; n < Nn; n++) {
      xbuf[xpos] = inp[n];
      // what each active mic hears = fractionally-delayed source + its own sensor noise
      for (let i = 0; i < N; i++) {
        const d0 = di0[i], df = dif[i];
        const a = xbuf[(xpos - d0 + Mx) % Mx], b = xbuf[(xpos - d0 - 1 + Mx) % Mx];
        mbuf[i * Mm + mpos] = a + (b - a) * df + SIG * (Math.random() * 2 - 1);
      }
      // true ear sound (adaptation target)
      const ea = xbuf[(xpos - tg0 + Mx) % Mx], eb = xbuf[(xpos - tg0 - 1 + Mx) % Mx];
      const ear = ea + (eb - ea) * tgf;
      let yhat = 0, energy = 1e-6;
      for (let i = 0; i < N; i++) { const b = i * Mm;
        for (let k = 0; k < T; k++) { const mv = mbuf[b + ((mpos - k + Mm) % Mm)]; yhat += w[i * T + k] * mv; energy += mv * mv; } }
      Yh[ypos] = yhat;
      // anti-noise reaches the ear late by the processing latency AND the secondary-path group
      // delay, scaled by the secondary-path amplitude match (fractional read, linear interp)
      let yDel = 0;
      if (en) {
        const fp = ypos - latSamp - tauSamp, i0 = Math.floor(fp), fr = fp - i0;
        const ya = Yh[((i0 % Mx) + Mx) % Mx], yb = Yh[(((i0 + 1) % Mx) + Mx) % Mx];
        yDel = gS * (ya + (yb - ya) * fr);
      }
      const e = en ? (ear - yDel) : ear;
      out[n] = e < -2 ? -2 : (e > 2 ? 2 : e);
      if (en) {
        const step = MU * (ear - yhat) / energy;             // adapt on the un-delayed prediction error
        for (let i = 0; i < N; i++) { const b = i * Mm;
          for (let k = 0; k < T; k++) { const j = b + ((mpos - k + Mm) % Mm); w[i * T + k] = w[i * T + k] * (1 - LEAK) + step * mbuf[j]; } }
      }
      xpos = (xpos + 1) % Mx; mpos = (mpos + 1) % Mm; ypos = (ypos + 1) % Mx;
    }
    this.xpos = xpos; this.mpos = mpos; this.ypos = ypos;
    return true;
  }
}

registerProcessor('adaptive-anc', AdaptiveAncProcessor);
