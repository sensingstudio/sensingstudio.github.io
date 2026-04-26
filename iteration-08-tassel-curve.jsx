// Iteration 08 — fix the tassel's 90° corner
// The tassel cord now drapes naturally:
//   - Starts at the button (board center)
//   - Curves smoothly over the right corner of the diamond
//   - Hangs down with a slight catenary (pulled by gravity)
// Cap silhouette + position from iteration 07 are unchanged.

const RED = 'rgb(188,18,42)';
const BLACK = '#111';
const GOLD = '#E8B923';

function MarkBullseye({ size=180 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-label="Sensing School">
      <circle cx="50" cy="50" r="46" fill="none" stroke={RED} strokeWidth="2" opacity="0.55"/>
      <circle cx="50" cy="50" r="36" fill="none" stroke={RED} strokeWidth="2.5" opacity="0.7"/>
      <circle cx="50" cy="50" r="26" fill="none" stroke={RED} strokeWidth="3"/>
      <circle cx="50" cy="50" r="14" fill="none" stroke={RED} strokeWidth="3.5"/>
      <circle cx="50" cy="50" r="6"  fill={RED}/>
    </svg>
  );
}

function TallGradCap({ width=120, capColor=BLACK, tasselColor=GOLD, tasselLen=70, tilt=12 }) {
  const totalH = 58 + tasselLen + 16;
  // Tassel path notes:
  //   - leaves the button at (50,20) heading right
  //   - drapes over the right edge — control point pulls it slightly down
  //     and BEYOND the corner so it bends around it (no hard 90°)
  //   - then continues into a gentle catenary downward
  const cordEndY = 20 + tasselLen;
  return (
    <svg
      width={width}
      height={width * (totalH/100)}
      viewBox={`0 0 100 ${totalH}`}
      style={{ overflow:'visible', display:'block', transform:`rotate(${tilt}deg)`, transformOrigin:'50px 30px' }}
    >
      {/* TALLER crown / under-cap */}
      <path d="M 24 20 L 76 20 L 70 56 Q 50 64 30 56 Z" fill={capColor}/>
      {/* mortarboard diamond */}
      <path d="M 50 2 L 98 20 L 50 38 L 2 20 Z" fill={capColor}/>
      {/* edge highlight */}
      <path d="M 50 38 L 98 20 L 96 21.6 L 50 39.6 Z" fill="#000" opacity="0.18"/>
      {/* crown shadow band */}
      <path d="M 24 20 L 76 20 L 75 24 L 25 24 Z" fill="#000" opacity="0.18"/>
      {/* button */}
      <circle cx="50" cy="20" r="2.6" fill={tasselColor}/>
      {/* tassel cord — smooth curve, no 90° corner */}
      <path
        d={`
          M 50 20
          C 75 20, 92 22, 97 28
          C 99 36, 96 ${cordEndY*0.55}, 95 ${cordEndY}
        `}
        fill="none" stroke={tasselColor} strokeWidth="3" strokeLinecap="round"
      />
      {/* tassel head */}
      <g transform={`translate(95 ${cordEndY})`}>
        <circle cx="0" cy="-1" r="3.4" fill={tasselColor}/>
        <line x1="-3" y1="0" x2="-5" y2="13" stroke={tasselColor} strokeWidth="2.4" strokeLinecap="round"/>
        <line x1="-1" y1="1" x2="-2" y2="14" stroke={tasselColor} strokeWidth="2.4" strokeLinecap="round"/>
        <line x1="1"  y1="1" x2="2"  y2="14" stroke={tasselColor} strokeWidth="2.4" strokeLinecap="round"/>
        <line x1="3"  y1="0" x2="5"  y2="13" stroke={tasselColor} strokeWidth="2.4" strokeLinecap="round"/>
      </g>
    </svg>
  );
}

function Lockup({
  size=86, capWidth=130, tasselLen=80, tilt=12,
  tasselColor=GOLD, capColor=BLACK, markSize=180
}) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:36, padding:'40px 50px', background:'#fff', minHeight:380 }}>
      <MarkBullseye size={markSize}/>
      <div>
        <div style={{ position:'relative', display:'inline-block' }}>
          <div style={{
            fontFamily:'Roboto, sans-serif', fontWeight:900, fontSize:size,
            lineHeight:0.95, letterSpacing:'-1px', color: BLACK
          }}>SENSING</div>
          <div style={{
            position:'absolute', top: -capWidth*0.55, right: -capWidth*0.55, pointerEvents:'none'
          }}>
            <TallGradCap width={capWidth} tasselLen={tasselLen} tilt={tilt}
              tasselColor={tasselColor} capColor={capColor}/>
          </div>
        </div>
        <div style={{
          fontFamily:'Roboto, sans-serif', fontWeight:900, fontSize:size,
          lineHeight:0.95, letterSpacing:'-1px', color: RED, marginTop:6
        }}>SCHOOL</div>
        <div style={{ height:1, background:'#dee2e6', margin:'18px 0 10px' }}/>
        <div style={{
          fontFamily:'Roboto, sans-serif', fontWeight:400, fontSize:18,
          letterSpacing:'2px', color:'#5b6068', textTransform:'uppercase'
        }}>Carnegie Mellon University</div>
      </div>
    </div>
  );
}

const V1 = () => <Lockup/>;
const V2 = () => <Lockup tilt={18} tasselLen={92}/>;
const V3 = () => <Lockup tilt={6} capWidth={120} tasselLen={70}/>;
const V4 = () => <Lockup size={56} capWidth={84} tasselLen={54} tilt={12} markSize={120}/>;

Object.assign(window, { MarkBullseye, TallGradCap, V1, V2, V3, V4 });
