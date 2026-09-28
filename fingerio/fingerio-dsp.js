// ============================================================
//  FingerIO shared DSP
//
//  The waveform and the per-channel CIR pipeline, with no DOM and no
//  globals of its own. fingerio/ drives one channel of this; the 2D
//  page drives two. Keeping it in one place is what stops the two
//  pages from drifting into subtly different sonar.
//
//  A channel owns a ring buffer, the symbol-boundary search and the
//  clutter estimate. It knows nothing about geometry: it reports the
//  delay of the moving echo in samples, and the caller turns that into
//  a range (1D) or into a path length for an ellipse (2D).
// ============================================================
(function (global) {
  'use strict';

  const SOUND_C = 343;              // m/s
  const ACTIVE_BIN_FLOOR = 1e-6;

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

  // The sync reference is the symbol zero-padded to 2N, transformed
  // once. Cached on the symbol so two channels sharing a waveform do
  // not each pay for it.
  function syncRef(symbol, N) {
    if (symbol._syncN === N) return symbol;
    const L = 2 * N;
    const re = new Float64Array(L);
    const im = new Float64Array(L);
    for (let n = 0; n < N; n++) re[n] = symbol.x[n];
    fft(re, im, false);
    symbol._syncRe = re;
    symbol._syncIm = im;
    symbol._syncN = N;
    return symbol;
  }

  // ============================================================
  //  One microphone's worth of pipeline.
  //
  //  push() takes however many samples arrived, writes them into the
  //  ring, re-finds the symbol boundary when due, and runs every
  //  aligned frame whose last sample has landed. onFrame(ch) fires
  //  once per recovered CIR.
  // ============================================================
  function makeChannel(opts) {
    const N = opts.N;
    const fs = opts.fs;
    const symbol = syncRef(opts.symbol, N);
    const ringSize = 1 << Math.max(15, Math.ceil(Math.log2(N * 16)));

    const ch = {
      N: N, fs: fs, symbol: symbol,
      label: opts.label || '',
      bgTau: opts.bgTau != null ? opts.bgTau : 2.0,
      // Motion smaller than this fraction of the direct path is noise,
      // and nothing is reported at all.
      minMoveRel: opts.minMoveRel != null ? opts.minMoveRel : 0.02,
      // Leading edge: the first bin reaching this fraction of the
      // frame's motion peak is taken as the fingertip.
      leadFrac: opts.leadFrac != null ? opts.leadFrac : 0.5,
      // Motion is this frame's profile minus the one this many frames
      // back, which is the paper's "subtraction between consecutive
      // echo profiles". A slow clutter EMA cannot do this job: it
      // accumulates energy where the finger recently was, which eats
      // the near side of the current echo and drags the leading edge
      // late in proportion to how fast the finger is moving.
      diffLag: opts.diffLag != null ? opts.diffLag : 0,
      // How the arrival time of a pulse is fixed. 'peak' interpolates
      // the maximum, which is unbiased for an isolated echo; 'lead'
      // takes the leading edge, which resolves a fingertip from the
      // hand behind it but is noisier and only unbiased because the
      // direct path is measured the same way.
      edgeMode: opts.edgeMode || 'peak',
      syncPeriodS: opts.syncPeriodS != null ? opts.syncPeriodS : 0.5,
      // The direct path cannot be further than the device is long.
      maxDirectM: opts.maxDirectM != null ? opts.maxDirectM : 0.4,
      guardBins: opts.guardBins != null ? opts.guardBins : 2,
      // Frames start this many samples BEFORE the strongest arrival.
      // Aligning the direct path onto bin 0 would push its own leading
      // edge around the circular CIR to the far end, where it cannot
      // be measured -- and the leading edge is the time origin.
      syncOffset: opts.syncOffset != null ? opts.syncOffset : 24,

      ring: new Float32Array(ringSize),
      ringMask: ringSize - 1,
      ringWrite: 0,
      nextFrame: 0,
      lastSyncAt: -1e9,
      resyncs: 0,

      h: new Float32Array(N),         // |h[n]| this frame
      hRe: new Float64Array(N),       // complex CIR, kept for phase work
      hIm: new Float64Array(N),
      bg: new Float32Array(N),        // slow clutter estimate
      motion: new Float32Array(N),    // max(0, |h| - |h| a few frames ago)
      bgReady: false,

      hPeak: 0,
      directBin: -1, directMag: 0,
      movePeak: 0, movePeakBin: -1,
      directFrac: -1,                 // leading edge of the direct path
      echoBin: -1,                    // first crossing past the direct path
      echoFrac: -1,                   // that crossing, to sub-sample precision
      delay: -1,                      // echoFrac - directFrac, in samples
      frames: 0
    };

    // Ring of recent magnitude profiles, so the frame from diffLag
    // frames back is always one slot away.
    const histLen = Math.max(1, ch.diffLag);   // 1 means "use the clutter estimate"
    const hist = [];
    for (let i = 0; i < histLen; i++) hist.push(new Float32Array(N));
    let histIdx = 0, histFilled = 0;

    const wre = new Float64Array(N);
    const wim = new Float64Array(N);
    const sre = new Float64Array(2 * N);
    const sim = new Float64Array(2 * N);
    const onFrame = opts.onFrame || function () {};

    // ----- symbol-boundary search -------------------------------
    // Correlate the last 2N samples against the known symbol; the lag
    // of the peak is where an aligned period starts.
    function findSync(yEnd) {
      const L = 2 * N;
      const ring = ch.ring, mask = ch.ringMask;
      const start = yEnd - L;
      for (let n = 0; n < L; n++) {
        sre[n] = ring[(start + n) & mask];
        sim[n] = 0;
      }
      fft(sre, sim, false);
      const Xre = symbol._syncRe, Xim = symbol._syncIm;
      for (let k = 0; k < L; k++) {
        const a = sre[k], b = sim[k];
        const c = Xre[k], d = -Xim[k];
        sre[k] = a * c - b * d;
        sim[k] = a * d + b * c;
      }
      fft(sre, sim, true);
      let best = -1, bestVal = -1;
      for (let n = 0; n < N; n++) {
        const m = sre[n] * sre[n] + sim[n] * sim[n];
        if (m > bestVal) { bestVal = m; best = n; }
      }
      return start + best;
    }

    // ----- one aligned frame ------------------------------------
    function processFrame(yStart) {
      const ring = ch.ring, mask = ch.ringMask;
      for (let n = 0; n < N; n++) {
        wre[n] = ring[(yStart + n) & mask];
        wim[n] = 0;
      }
      fft(wre, wim, false);
      // Divide by X(k) on active bins. |X(k)| = 1, so this is a
      // multiply by the conjugate. Inactive bins carry no information.
      const Xre = symbol.Xre, Xim = symbol.Xim, active = symbol.active;
      for (let k = 0; k < N; k++) {
        if (active[k]) {
          const a = wre[k], b = wim[k];
          const c = Xre[k], d = Xim[k];
          wre[k] = a * c + b * d;
          wim[k] = b * c - a * d;
        } else {
          wre[k] = 0; wim[k] = 0;
        }
      }
      fft(wre, wim, true);

      const h = ch.h, hRe = ch.hRe, hIm = ch.hIm;
      let hPeak = ACTIVE_BIN_FLOOR;
      for (let n = 0; n < N; n++) {
        hRe[n] = wre[n];
        hIm[n] = wim[n];
        const m = Math.hypot(wre[n], wim[n]);
        h[n] = m;
        if (m > hPeak) hPeak = m;
      }
      ch.hPeak = hPeak;

      // Clutter EMA over the magnitude.
      const dt = N / fs;
      const alpha = 1 - Math.exp(-dt / Math.max(0.05, ch.bgTau));
      const bg = ch.bg;
      if (!ch.bgReady) {
        bg.set(h);
        ch.bgReady = true;
      } else {
        for (let n = 0; n < N; n++) bg[n] += alpha * (h[n] - bg[n]);
      }

      // The direct path is the loudest thing inside one device length,
      // and it is the time origin: every echo delay is measured from
      // it, which is what makes the path length self-calibrating.
      const maxDirect = Math.min(N >> 1, Math.max(1, Math.ceil(ch.maxDirectM * fs / SOUND_C)));
      let dBin = -1, dMag = 0;
      for (let n = 0; n < maxDirect; n++) {
        if (bg[n] > dMag) { dMag = bg[n]; dBin = n; }
      }
      ch.directBin = dBin;
      ch.directMag = dMag;
      ch.directFrac = dBin < 0 ? -1
        : (ch.edgeMode === 'lead' ? crossing(bg, 0, dBin, ch.leadFrac * dMag)
                                  : parabolic(bg, dBin, N >> 1));

      // Motion is what changed since a few frames ago. The direct path
      // and the bins right behind it are skipped: speaker-to-mic
      // coupling dominates there and swamps anything real.
      const motion = ch.motion;
      const half = N >> 1;
      const first = (dBin >= 0 ? dBin : 0) + ch.guardBins;
      // diffLag = 0 measures against the slow clutter estimate instead
      // of a recent frame. Differencing two overlapping copies of the
      // same 12-sample-wide pulse drags the result late in proportion
      // to how fast the reflector is moving; a clutter estimate slow
      // enough that a moving finger never enters it does not.
      const past = hist[histIdx];
      const useDiff = histLen > 1 && histFilled >= histLen;
      let movePeak = 0, movePeakBin = -1;
      for (let n = 0; n < half; n++) {
        let v = histLen > 1 ? (useDiff ? h[n] - past[n] : 0) : h[n] - bg[n];
        if (v < 0) v = 0;
        motion[n] = v;
        if (n >= first && v > movePeak) { movePeak = v; movePeakBin = n; }
      }
      // Overwrite the slot we just consumed; it becomes the frame that
      // diffLag frames from now will compare against.
      past.set(h);
      histIdx = (histIdx + 1) % histLen;
      if (histFilled < histLen) histFilled++;

      // The fingertip is the NEAREST thing that moved, not the
      // loudest: the hand behind it moves along the same path and is
      // usually the stronger reflector. So take the leading edge --
      // the first bin reaching a fraction of this frame's own peak.
      //
      // The paper thresholds at 15% of the direct-path amplitude, but
      // a finger echo sits far below that; scaling to the frame's own
      // motion peak keeps the same "first crossing" logic without
      // depending on how loud the direct path happens to be.
      // Measured the same way as the direct path, so the pulse shape's
      // rise time -- about six samples at this bandwidth -- cancels in
      // the difference instead of biasing every distance low.
      let echoBin = -1, echoFrac = -1;
      if (movePeak > ch.minMoveRel * (dMag > 0 ? dMag : hPeak)) {
        if (ch.edgeMode === 'lead') {
          const lead = ch.leadFrac * movePeak;
          for (let n = first; n < half; n++) {
            if (motion[n] >= lead) { echoBin = n; break; }
          }
          if (echoBin >= 0) echoFrac = crossing(motion, first, echoBin, lead);
        } else {
          echoBin = movePeakBin;
          if (echoBin >= 0) echoFrac = parabolic(motion, echoBin, half);
        }
      }
      ch.movePeak = movePeak;
      ch.movePeakBin = movePeakBin;
      ch.echoBin = echoBin;
      ch.echoFrac = echoFrac;
      // Delay of the finger echo past the direct path, in samples.
      ch.delay = (echoFrac >= 0 && ch.directFrac >= 0) ? echoFrac - ch.directFrac : -1;
      ch.frames++;
      onFrame(ch);
    }

    // Sub-sample peak position, by fitting a parabola through the peak
    // bin and its two neighbours. Exact for a symmetric pulse, which is
    // what an isolated echo looks like at this bandwidth.
    function parabolic(a, i, n) {
      if (i <= 0 || i >= n - 1) return i;
      const l = a[i - 1], c = a[i], r = a[i + 1];
      const denom = l - 2 * c + r;
      if (Math.abs(denom) < 1e-12) return i;
      const d = 0.5 * (l - r) / denom;
      return (d > -1 && d < 1) ? i + d : i;
    }

    // Where a rising edge first reaches `level`, to sub-sample
    // precision, by interpolating between the last bin below it and
    // the first bin at or above it. One whole bin is 7.1 mm of path at
    // 48 kHz, so the fraction is worth recovering.
    function crossing(a, lo, hi, level) {
      for (let n = lo; n <= hi; n++) {
        if (a[n] >= level) {
          if (n === lo) return n;
          const prev = a[n - 1], cur = a[n];
          const span = cur - prev;
          return span > 1e-12 ? (n - 1) + (level - prev) / span : n;
        }
      }
      return hi;
    }

    // A backlog handed over in one lump can be longer than the ring, in
    // which case the oldest samples would be overwritten before the
    // frame grid reached them. Slice so the grid always stays inside.
    const MAX_CHUNK = ringSize >> 2;
    ch.push = function (input, n) {
      for (let off = 0; off < n; off += MAX_CHUNK) {
        const len = Math.min(MAX_CHUNK, n - off);
        pushChunk(off === 0 && len === n ? input : input.subarray(off, off + len), len);
      }
    };

    function pushChunk(input, n) {
      const ring = ch.ring, mask = ch.ringMask;
      for (let i = 0; i < n; i++) ring[(ch.ringWrite + i) & mask] = input[i];
      ch.ringWrite += n;

      if (ch.ringWrite - ch.lastSyncAt > fs * ch.syncPeriodS && ch.ringWrite > 2 * N) {
        const lag = findSync(ch.ringWrite) - ch.syncOffset;
        if (lag >= 0) {
          const periods = Math.ceil((ch.ringWrite - lag) / N);
          ch.nextFrame = lag + periods * N;
          ch.lastSyncAt = ch.ringWrite;
          ch.resyncs++;
        }
      }

      while (ch.nextFrame + N <= ch.ringWrite
             && ch.nextFrame > 0
             && ch.ringWrite - ch.nextFrame < ch.ring.length - N) {
        const frame = ch.nextFrame;
        ch.nextFrame += N;
        processFrame(frame);
      }
    }

    // A hole in the stream shifts every later sample, so the frame
    // grid no longer lines up with the transmitter. Find it again.
    ch.markGap = function () { ch.lastSyncAt = -1e9; };

    ch.reset = function () {
      ch.ringWrite = 0; ch.nextFrame = 0; ch.lastSyncAt = -1e9;
      ch.bgReady = false; ch.frames = 0; ch.resyncs = 0;
      ch.hPeak = 0; ch.directBin = -1; ch.directMag = 0;
      ch.movePeak = 0; ch.movePeakBin = -1; ch.echoBin = -1; ch.echoFrac = -1;
      ch.directFrac = -1; ch.delay = -1;
      histIdx = 0; histFilled = 0;
      for (const frame of hist) frame.fill(0);
      ch.ring.fill(0); ch.h.fill(0); ch.bg.fill(0); ch.motion.fill(0);
    };

    return ch;
  }

  // ============================================================
  //  Two ellipses, one shared focus.
  //
  //  The speaker sits at the origin and both microphones lie on the
  //  device's long axis at signed offsets m1, m2 (metres). A finger at
  //  F = (x, y) that produced a total path length d = |F| + |F - M|
  //  satisfies, after expanding |F - M|^2 = (d - r)^2 with r = |F|:
  //
  //      y * m  -  r * d  =  (m^2 - d^2) / 2
  //
  //  which is LINEAR in the unknowns y and r. Two microphones give a
  //  2x2 solve, and x follows from r^2 = x^2 + y^2. The sign of x is
  //  genuinely unrecoverable — the geometry is mirror-symmetric about
  //  the device axis — so the caller picks a side.
  // ============================================================
  function solve2D(m1, d1, m2, d2) {
    // Reject anything that violates the triangle inequality before it
    // turns into a confidently wrong coordinate.
    if (!(d1 >= Math.abs(m1)) || !(d2 >= Math.abs(m2))) return null;

    const b1 = (m1 * m1 - d1 * d1) / 2;
    const b2 = (m2 * m2 - d2 * d2) / 2;
    const det = m1 * (-d2) - (-d1) * m2;      // m1*(-d2) + d1*m2
    if (Math.abs(det) < 1e-9) return null;    // mics too close to separate

    const y = (b1 * (-d2) - (-d1) * b2) / det;
    const r = (m1 * b2 - b1 * m2) / det;
    if (!(r > 0)) return null;

    const x2 = r * r - y * y;
    if (x2 < 0) return null;
    return { x: Math.sqrt(x2), y: y, r: r };
  }

  global.FingerDSP = {
    SOUND_C: SOUND_C,
    fft: fft,
    buildSymbol: buildSymbol,
    makeChannel: makeChannel,
    solve2D: solve2D
  };
})(typeof window !== 'undefined' ? window : globalThis);
