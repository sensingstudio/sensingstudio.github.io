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
const selectedMics = (perSide) => [...LEFT_MICS.slice(0, perSide), ...RIGHT_MICS.slice(0, perSide)];
const srcPos = (thetaDeg, d) => { const th = thetaDeg * Math.PI / 180; return [d * Math.sin(th), d * Math.cos(th)]; };

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
    this.theta = 40; this.dist = 1.5; this.nMics = 4;
    this.port.onmessage = (e) => { Object.assign(this, e.data); };
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    const inp = inputs[0] && inputs[0][0];
    if (!inp || this.algo !== 'adaptive') { out.fill(0); return true; }

    const SR = sampleRate, en = this.ancOn ? 1 : 0, Nn = out.length;
    const T = this.T, Mx = this.Mx, Mm = this.Mm, w = this.w, xbuf = this.xbuf, mbuf = this.mbuf, Yh = this.Yh;
    const MU = this.MU, LEAK = this.LEAK, SIG = this.SIG;
    const latSamp = Math.min(Mx - 1, Math.round(Math.max(0, this.latencyMs * 1e-3) * SR));
    // geometry: per-active-mic delay + the ear delay (relative to nearest receiver)
    const P = srcPos(this.theta, this.dist), idx = selectedMics(this.nMics), N = idx.length;
    const ds = idx.map(mi => Math.hypot(MICS[mi][0] - P[0], MICS[mi][1] - P[1]));
    const dEar = Math.hypot(EAR0[0] - P[0], EAR0[1] - P[1]);
    const dmin = Math.min(dEar, ...ds);
    const di = ds.map(d => (d - dmin) / C * SR);
    let dE = (dEar - dmin) / C * SR, mld = 0; for (const d of di) mld = Math.max(mld, d - dE);
    const tgt = dE + mld;
    const di0 = di.map(d => d | 0), dif = di.map((d, i) => d - di0[i]);
    const tg0 = tgt | 0, tgf = tgt - tg0;
    if (N !== this.curN) { w.fill(0); this.curN = N; }   // active-mic set changed → reset filters
    const readX = (d0, df) => { const a = xbuf[(this.xpos - d0 + Mx) % Mx], b = xbuf[(this.xpos - d0 - 1 + Mx) % Mx]; return a + (b - a) * df; };

    let xpos = this.xpos, mpos = this.mpos, ypos = this.ypos;
    for (let n = 0; n < Nn; n++) {
      const x = inp[n]; xbuf[xpos] = x;
      for (let i = 0; i < N; i++) mbuf[i * Mm + mpos] = readX(di0[i], dif[i]) + SIG * (Math.random() * 2 - 1);
      const ear = readX(tg0, tgf);
      let yhat = 0, energy = 1e-6;
      for (let i = 0; i < N; i++) { const b = i * Mm;
        for (let k = 0; k < T; k++) { const mv = mbuf[b + ((mpos - k + Mm) % Mm)]; yhat += w[i * T + k] * mv; energy += mv * mv; } }
      Yh[ypos] = yhat;
      const yDel = en ? Yh[(ypos - latSamp + Mx) % Mx] : 0;
      const e = en ? (ear - yDel) : ear;
      out[n] = Math.max(-2, Math.min(2, e));
      if (en) {
        const step = MU * (ear - yhat) / energy;
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
