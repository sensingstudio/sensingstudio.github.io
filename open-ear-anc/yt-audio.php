<?php
/**
 * yt-audio.php — extract the audio track from a YouTube link for the ANC demo.
 *
 * The front end posts/gets a YouTube URL; we use yt-dlp to pull the audio,
 * transcode it to mp3 (cached by video id), and return JSON describing the
 * cached file plus title/uploader/thumbnail. The browser then fetches the
 * file and decodes it into the Web Audio graph exactly like the bundled clip.
 *
 *   GET/POST  url=<youtube link>   -> {ok, id, title, uploader, audio, thumb}
 *
 * Requires yt-dlp + ffmpeg on the server (this box: /opt/homebrew/bin).
 */

header('Content-Type: application/json');

// ---- config -------------------------------------------------------------
$BIN_DIRS   = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin'];
$CACHE_DIR  = __DIR__ . '/yt-cache';
$MAX_SECONDS = 1200;          // skip anything longer than 20 min
$MAX_AGE     = 7 * 24 * 3600; // prune cached files older than a week

// ---- helpers ------------------------------------------------------------
function fail($msg, $code = 400) {
  http_response_code($code);
  echo json_encode(['ok' => false, 'error' => $msg]);
  exit;
}

function find_bin($name, $dirs) {
  foreach ($dirs as $d) {
    $p = "$d/$name";
    if (is_executable($p)) return $p;
  }
  return null;
}

// ---- locate tools -------------------------------------------------------
$ytdlp  = find_bin('yt-dlp', $BIN_DIRS);
$ffmpeg = find_bin('ffmpeg', $BIN_DIRS);
if (!$ytdlp)  fail('yt-dlp not found on server', 500);
if (!$ffmpeg) fail('ffmpeg not found on server', 500);
$ffdir = dirname($ffmpeg);

// ---- validate input -----------------------------------------------------
$url = $_GET['url'] ?? $_POST['url'] ?? '';
$url = trim($url);
if ($url === '') fail('missing url');
if (!preg_match('#^https?://(www\.|m\.|music\.)?(youtube\.com/|youtu\.be/)#i', $url)) {
  fail('not a YouTube URL');
}

if (!is_dir($CACHE_DIR)) @mkdir($CACHE_DIR, 0775, true);
if (!is_dir($CACHE_DIR)) fail('cache dir unavailable', 500);

// best-effort prune of stale cache entries
foreach (glob("$CACHE_DIR/*") as $f) {
  if (is_file($f) && (time() - filemtime($f)) > $MAX_AGE) @unlink($f);
}

// ---- fetch metadata (id, title, uploader, duration) ---------------------
$meta_cmd = escapeshellarg($ytdlp)
  . ' --no-playlist --no-warnings --skip-download'
  . ' --print "%(id)s\t%(title)s\t%(uploader)s\t%(duration)s"'
  . ' ' . escapeshellarg($url) . ' 2>&1';
$meta_out = shell_exec("PATH=$ffdir:\$PATH " . $meta_cmd);
$line = trim(is_string($meta_out) ? strtok($meta_out, "\n") : '');
$parts = explode("\t", $line);
if (count($parts) < 1 || !preg_match('/^[A-Za-z0-9_-]{6,15}$/', $parts[0])) {
  fail('could not read video metadata: ' . substr(trim((string)$meta_out), 0, 300), 502);
}
$id       = $parts[0];
$title    = $parts[1] ?? $id;
$uploader = $parts[2] ?? '';
$duration = isset($parts[3]) ? (int)$parts[3] : 0;
if ($MAX_SECONDS && $duration > $MAX_SECONDS) {
  fail('track too long (' . $duration . 's; limit ' . $MAX_SECONDS . 's)');
}

$rel  = 'yt-cache/' . $id . '.mp3';
$file = $CACHE_DIR . '/' . $id . '.mp3';

// ---- download + transcode if not cached ---------------------------------
if (!file_exists($file)) {
  $out_tmpl = $CACHE_DIR . '/%(id)s.%(ext)s';
  $dl_cmd = escapeshellarg($ytdlp)
    . ' --no-playlist --no-warnings -f bestaudio/best'
    . ' -x --audio-format mp3 --audio-quality 5'
    . ' --ffmpeg-location ' . escapeshellarg($ffdir)
    . ' -o ' . escapeshellarg($out_tmpl)
    . ' ' . escapeshellarg($url) . ' 2>&1';
  $dl_out = shell_exec("PATH=$ffdir:\$PATH " . $dl_cmd);
  if (!file_exists($file)) {
    fail('extraction failed: ' . substr(trim((string)$dl_out), 0, 400), 502);
  }
}

echo json_encode([
  'ok'       => true,
  'id'       => $id,
  'title'    => $title,
  'uploader' => $uploader,
  'duration' => $duration,
  'audio'    => $rel,
  'thumb'    => 'https://i.ytimg.com/vi/' . $id . '/hqdefault.jpg',
]);
