/**
 * tempo-worklet.js — real-time, pitch-preserving tempo change (WSOLA).
 *
 * The main thread posts the decoded PCM once ({channels, length}); thereafter the
 * processor streams a time-stretched version of it, looping seamlessly. Tempo is a
 * live AudioParam ('speed'): 1 = normal, 2 = twice as fast, 0.5 = half speed — pitch
 * unchanged. Doing this on the audio thread means changing speed never restarts the
 * track and never blocks the page (the cost is a tiny per-grain cross-correlation).
 */
class TempoProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [{ name: 'speed', defaultValue: 1, minValue: 0.25, maxValue: 4, automationRate: 'k-rate' }];
  }
  constructor() {
    super();
    this.ready = false;
    this.Fs = 2048;            // grain size
    this.Hs = this.Fs >> 1;    // synthesis hop (50% overlap)
    this.tol = 256;            // similarity-search radius (samples)
    this.corrStep = 4;         // decimate the correlation for speed
    this.win = new Float32Array(this.Fs);
    for (let i = 0; i < this.Fs; i++) this.win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (this.Fs - 1));
    this.port.onmessage = (e) => this._load(e.data);
  }
  _load(d) {
    this.src = d.channels;             // array of Float32Array (transferred)
    this.N = d.length;
    this.ch = this.src.length;
    this.mono = new Float32Array(this.N);   // mixdown drives the grain alignment
    for (let c = 0; c < this.ch; c++) { const s = this.src[c]; for (let i = 0; i < this.N; i++) this.mono[i] += s[i]; }
    if (this.ch > 1) for (let i = 0; i < this.N; i++) this.mono[i] /= this.ch;
    this.tail = [];                    // per-channel overlap carryover (Hs samples)
    this.ring = [];                    // per-channel output FIFO
    this.ringSize = this.Fs * 4;
    for (let c = 0; c < this.ch; c++) { this.tail.push(new Float32Array(this.Hs)); this.ring.push(new Float32Array(this.ringSize)); }
    this.rHead = 0; this.rTail = 0; this.rCount = 0;
    this.inPos = 0; this.refBase = 0; this.first = true;
    this.ready = true;
  }
  _wrap(i) { const N = this.N; i %= N; return i < 0 ? i + N : i; }
  _grain(speed) {
    const { Fs, Hs, tol, corrStep, win, mono } = this;
    const base = Math.round(this.inPos);
    let delta = 0;
    if (!this.first) {                 // align this grain to the previous grain's natural continuation
      let best = -Infinity;
      for (let d = -tol; d <= tol; d++) {
        let acc = 0;
        for (let i = 0; i < Hs; i += corrStep) acc += mono[this._wrap(base + d + i)] * mono[this._wrap(this.refBase + i)];
        if (acc > best) { best = acc; delta = d; }
      }
    }
    const g = base + delta;
    for (let c = 0; c < this.ch; c++) {
      const s = this.src[c], tail = this.tail[c], ring = this.ring[c];
      let w = this.rTail;
      for (let i = 0; i < Hs; i++) {                    // first half: finalize with prior tail, push to FIFO
        ring[w] = tail[i] + s[this._wrap(g + i)] * win[i];
        w = (w + 1) % this.ringSize;
      }
      for (let i = 0; i < Hs; i++) tail[i] = s[this._wrap(g + Hs + i)] * win[Hs + i];   // second half: new tail
    }
    this.rTail = (this.rTail + Hs) % this.ringSize;
    this.rCount += Hs;
    this.refBase = g + Hs;
    this.inPos += Hs * speed;
    if (this.inPos >= this.N) this.inPos -= this.N;     // keep the analysis pointer bounded (loop)
    this.first = false;
  }
  process(_inputs, outputs, params) {
    const out = outputs[0];
    if (!this.ready) { for (let c = 0; c < out.length; c++) out[c].fill(0); return true; }
    const speed = params.speed[0];
    const need = out[0].length;                         // render quantum (128)
    while (this.rCount < need) this._grain(speed);
    for (let c = 0; c < out.length; c++) {
      const dst = out[c], ring = this.ring[Math.min(c, this.ch - 1)];
      let r = this.rHead;
      for (let i = 0; i < need; i++) { dst[i] = ring[r]; r = (r + 1) % this.ringSize; }
    }
    this.rHead = (this.rHead + need) % this.ringSize;
    this.rCount -= need;
    return true;
  }
}
registerProcessor('tempo-stretch', TempoProcessor);
