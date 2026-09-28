// End to end: a simulated phone emits FIO2 packets for a finger moving
// along a known path; the page's parse + solve chain must follow it.
require('/Applications/XAMPP/xamppfiles/htdocs/sensingstudio/fingerio/fingerio-dsp.js');
const D = globalThis.FingerDSP, C = D.SOUND_C;
const fs = 48000, N = 256;
const sym = D.buildSymbol(N, fs, 18000, 22000);

const m1 = 0.010, m2 = 0.145;             // mic offsets from the speaker
const b1 = Math.abs(m1), b2 = Math.abs(m2);

// --- the fake phone ------------------------------------------------
let seq = 0;
function packet(nFrames, paths) {
  const buf = new ArrayBuffer(16 + 2 * nFrames * 4);
  const view = new DataView(buf);
  view.setUint8(0, 0x46); view.setUint8(1, 0x49); view.setUint8(2, 0x4F); view.setUint8(3, 0x32);
  view.setUint32(4, seq, true);
  view.setUint32(8, nFrames, true);
  view.setUint16(12, 2, true);
  for (let c = 0; c < 2; c++) {
    const out = new Float32Array(buf, 16 + c * nFrames * 4, nFrames);
    for (let i = 0; i < nFrames; i++) {
      const n = seq + i;
      let v = 0;
      for (const p of paths[c]) v += p.a * sym.x[(((n - p.d) % N) + N) % N];
      out[i] = v + (Math.random() - 0.5) * 0.004;
    }
  }
  seq = (seq + nFrames) >>> 0;
  return buf;
}

// --- the page ------------------------------------------------------
const chans = [0, 1].map(i => D.makeChannel({ N, fs, symbol: sym, label: 'ch' + i }));
let expectSeq = null, gaps = 0;

function onAudio(buf) {
  const view = new DataView(buf);
  if (view.getUint32(0, false) !== 0x46494F32) throw new Error('bad magic');
  const s = view.getUint32(4, true);
  const nFrames = view.getUint32(8, true);
  const ch = view.getUint16(12, true);
  if (expectSeq !== null && s !== expectSeq) { gaps++; for (const c of chans) c.markGap(); }
  expectSeq = (s + nFrames) >>> 0;
  for (let c = 0; c < Math.min(ch, chans.length); c++) {
    chans[c].push(new Float32Array(buf, 16 + c * nFrames * 4, nFrames), nFrames);
  }
  if (chans[0].delay < 0 || chans[1].delay < 0) return null;
  const d1 = b1 + chans[0].delay * C / fs;
  const d2 = b2 + chans[1].delay * C / fs;
  return D.solve2D(m1, d1, m2, d2);
}

// --- geometry helpers ----------------------------------------------
const D1 = Math.round(b1 * fs / C), D2 = Math.round(b2 * fs / C);
// The simulator can only place echoes on whole samples, so the best any
// estimator could do is the position those rounded delays imply. Compare
// against that, or the simulator's own quantisation gets blamed on the
// pipeline -- at this geometry one sample is already 10-16 mm.
function quantisedTruth(pt) {
  const r = Math.hypot(pt.x, pt.y);
  const e1 = Math.round((r + Math.hypot(pt.x, pt.y - m1)) * fs / C);
  const e2 = Math.round((r + Math.hypot(pt.x, pt.y - m2)) * fs / C);
  return D.solve2D(m1, b1 + (e1 - D1) * C / fs, m2, b2 + (e2 - D2) * C / fs);
}

function pathsFor(pt) {
  const quiet = [[{ a: 1.0, d: D1 }, { a: 0.3, d: 90 }],
                 [{ a: 1.0, d: D2 }, { a: 0.3, d: 96 }]];
  if (!pt) return quiet;
  const r = Math.hypot(pt.x, pt.y);
  const t1 = r + Math.hypot(pt.x, pt.y - m1);
  const t2 = r + Math.hypot(pt.x, pt.y - m2);
  return [quiet[0].concat([{ a: 0.07, d: Math.round(t1 * fs / C) }]),
          quiet[1].concat([{ a: 0.07, d: Math.round(t2 * fs / C) }])];
}

// Two seconds of empty room so the clutter estimate settles.
for (let i = 0; i < 94; i++) onAudio(packet(1024, pathsFor(null)));
console.log('warm-up frames per channel:', chans[0].frames, '| false fixes:',
            chans[0].echoBin >= 0 || chans[1].echoBin >= 0 ? 'yes' : 'none');

// Then a finger drawing a diagonal stroke.
const errs = [], raw = [];
const STEPS = 60;
for (let k = 0; k < STEPS; k++) {
  const u = k / (STEPS - 1);
  const truth = { x: 0.06 + 0.10 * u, y: 0.02 + 0.14 * u };
  const f = onAudio(packet(1024, pathsFor(truth)));
  if (k < 5) continue;                       // let the echo establish
  if (!f) { errs.push(null); continue; }
  const q = quantisedTruth(truth);
  errs.push(Math.hypot(f.x - q.x, f.y - q.y));
  raw.push(Math.hypot(f.x - truth.x, f.y - truth.y));
}

const got = errs.filter(e => e !== null);
const missed = errs.length - got.length;
got.sort((a, b) => a - b);
const median = got[Math.floor(got.length / 2)];
const p90 = got[Math.floor(got.length * 0.9)];
console.log('fixes:', got.length, '| no-fix:', missed, '| gaps:', gaps);
const rawSorted = raw.slice().sort((a, b) => a - b);
console.log('vs quantised truth   median ' + (median * 1000).toFixed(1) + ' mm');
console.log('vs continuous truth  median ' + (rawSorted[Math.floor(rawSorted.length / 2)] * 1000).toFixed(1) + ' mm  (includes the simulator\'s own rounding)');
console.log('p90 error:    ' + (p90 * 1000).toFixed(1) + ' mm');
const ok = got.length > errs.length * 0.9 && median < 0.015;
console.log(ok ? '\nEND TO END OK' : '\nEND TO END WEAK');
process.exit(ok ? 0 : 1);
