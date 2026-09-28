// Headless check of the shared DSP: the ellipse solver in isolation,
// then the full channel pipeline on a synthesised two-mic recording.
require('/Applications/XAMPP/xamppfiles/htdocs/sensingstudio/fingerio/fingerio-dsp.js');
const D = globalThis.FingerDSP;
const C = D.SOUND_C, fs = 48000, N = 256;
let fails = 0;
const ok = (name, cond, detail) => {
  console.log((cond ? 'PASS  ' : 'FAIL  ') + name + (detail ? '   ' + detail : ''));
  if (!cond) fails++;
};

// ---- A. solve2D inverts the geometry exactly -------------------
const m1 = 0.010, m2 = 0.145;                 // mic offsets from speaker
function pathLengths(x, y) {
  const r = Math.hypot(x, y);
  return [r + Math.hypot(x, y - m1), r + Math.hypot(x, y - m2)];
}
for (const [x, y] of [[0.10, 0.05], [0.05, 0.20], [0.25, -0.08], [0.02, 0.07]]) {
  const [d1, d2] = pathLengths(x, y);
  const f = D.solve2D(m1, d1, m2, d2);
  const err = f ? Math.hypot(f.x - x, f.y - y) : Infinity;
  ok(`solve2D recovers (${x}, ${y})`, err < 1e-9, `err=${err.toExponential(2)} m`);
}
ok('solve2D rejects an impossible path', D.solve2D(m1, 0.001, m2, 0.9) === null);
ok('solve2D rejects co-located mics', D.solve2D(0.05, 0.3, 0.05, 0.3) === null);

// ---- B. the channel recovers simulated echo delays --------------
const sym = D.buildSymbol(N, fs, 18000, 22000);

// Received signal is a circular convolution: the symbol repeats
// forever, so each path is just a shifted copy of one period.
function synth(paths, nSamples) {
  const out = new Float32Array(nSamples);
  for (let n = 0; n < nSamples; n++) {
    let v = 0;
    for (const p of paths) v += p.a * sym.x[(((n - p.d) % N) + N) % N];
    out[n] = v + (Math.random() - 0.5) * 0.002;
  }
  return out;
}

const DIRECT1 = 1, DIRECT2 = 20;     // speaker->mic direct arrivals
const ECHO1 = 31, ECHO2 = 35;        // speaker->finger->mic arrivals
const clutter = { a: 0.30, d: 64 };  // a static wall, must not be tracked

// A strong echo isolates the geometry from the estimator: at this
// amplitude the magnitude peak IS the arrival, so any error here is a
// bug in the delay maths rather than a limit of the sonar. The weak
// case below measures what a real finger actually costs.
const ECHO_AMP = 0.8;

const results = {};
for (const [label, direct, echo] of [['mic1', DIRECT1, ECHO1], ['mic2', DIRECT2, ECHO2]]) {
  const ch = D.makeChannel({ N, fs, symbol: sym, label });
  const quiet = [{ a: 1.0, d: direct }, clutter];
  const withFinger = quiet.concat([{ a: ECHO_AMP, d: echo }]);
  // Two seconds with no finger lets the clutter estimate converge,
  // then the finger appears and must show up as motion.
  ch.push(synth(quiet, fs * 2), fs * 2);
  const bgFrames = ch.frames;
  const clutterTracked = ch.echoBin;
  // Only a few frames: motion is now this profile minus one from
  // diffLag frames back, so a finger that appears and then holds still
  // becomes invisible again -- which is what an active sonar does.
  const buf = synth(withFinger, 1024);
  ch.push(buf, buf.length);
  results[label] = { delta: ch.delay, ch };
  ok(`${label} converges on the clutter (no false finger)`,
     clutterTracked < 0, `echoBin=${clutterTracked}, frames=${bgFrames}`);
  ok(`${label} finds the finger echo`, ch.echoBin >= 0, `echoBin=${ch.echoBin}`);
  const expected = echo - direct;
  const got = ch.delay;
  ok(`${label} delay = ${expected} samples past direct`,
     Math.abs(got - expected) < 1.0, `got=${got.toFixed(2)}`);
}

// ---- C. end to end: delays -> path lengths -> 2D position -------
const d1 = m1 + results.mic1.delta * C / fs;
const d2 = m2 + results.mic2.delta * C / fs;
const f = D.solve2D(m1, d1, m2, d2);
if (!f) { ok('end-to-end position', false, 'solver returned null'); }
else {
  // Ground truth implied by the simulated integer delays.
  const t1 = m1 + (ECHO1 - DIRECT1) * C / fs;
  const t2 = m2 + (ECHO2 - DIRECT2) * C / fs;
  const truth = D.solve2D(m1, t1, m2, t2);
  const err = Math.hypot(f.x - truth.x, f.y - truth.y);
  // Sub-sample delay error of half a sample is ~3.5 mm of path length,
  // and the ellipse geometry amplifies that. The paper reports 8 mm
  // average 2D accuracy on real hardware, so anything inside 15 mm here
  // means the pipeline is sound rather than merely lucky.
  ok('end-to-end position within 15 mm of the simulated truth', err < 0.015,
     `got=(${(f.x*100).toFixed(1)}, ${(f.y*100).toFixed(1)}) cm, ` +
     `truth=(${(truth.x*100).toFixed(1)}, ${(truth.y*100).toFixed(1)}) cm, err=${(err*1000).toFixed(1)} mm`);
}

// ---- D. what a realistic echo amplitude costs -------------------
// Only 22 subcarriers are active, so the CIR is a ~20 kHz carrier under
// a 12-sample envelope. Magnitude peak-picking on that is pulled by the
// direct path's sidelobes once the echo is weak, and no amount of
// clutter modelling fixes it -- the paper's step 3 fits the phase slope
// across subcarriers instead, which is not implemented here.
console.log('\namplitude sensitivity (mic1, true delay 30):');
for (const amp of [0.06, 0.2, 0.8, 3.0]) {
  const ch = D.makeChannel({ N, fs, symbol: sym });
  const quiet = [{ a: 1.0, d: DIRECT1 }, clutter];
  ch.push(synth(quiet, fs * 2), fs * 2);
  const buf = synth(quiet.concat([{ a: amp, d: ECHO1 }]), 1024);
  ch.push(buf, buf.length);
  const err = ch.delay - (ECHO1 - DIRECT1);
  console.log(`  echo amplitude ${String(amp).padStart(4)}  ->  delay ${ch.delay.toFixed(2).padStart(6)}` +
              `  (${err >= 0 ? '+' : ''}${err.toFixed(2)} samples, ${(err * 100 * C / fs).toFixed(1)} cm of path)`);
}

console.log(fails ? `\n${fails} failure(s)` : '\nall good');
process.exit(fails ? 1 : 0);
