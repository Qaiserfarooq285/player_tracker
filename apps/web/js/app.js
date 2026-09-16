/**
 * AI Football Player Tracking & Analytics Engine - Frontend Logic
 */

let activeJobId = null;
let pollingTimer = null;
let currentResults = null;
let pitchMode = 'trajectory';

document.addEventListener('DOMContentLoaded', async () => {
  initLoginGate();
  await ensureAuthenticated();
  initApp();
  initUploadZone();
  initCanvasPitch();
});

// ---- Access gate (hosted deployment, docs/DEPLOY.md) ----
// The server only enforces a password when PV_ACCESS_PASSWORD is set; locally the overlay never
// appears. Any later 401 (session expired after a redeploy) re-opens it via `fetch` below.
let _loginResolvers = [];
function showLoginOverlay() {
  document.getElementById('login-overlay')?.classList.remove('hidden');
  document.getElementById('login-password')?.focus();
  return new Promise(resolve => _loginResolvers.push(resolve));
}
function initLoginGate() {
  const form = document.getElementById('login-form');
  if (!form) return;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const pw = document.getElementById('login-password').value;
    const err = document.getElementById('login-error');
    err.classList.add('hidden');
    const res = await _rawFetch('/api/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: pw }),
    });
    if (res.ok) {
      document.getElementById('login-overlay').classList.add('hidden');
      document.getElementById('login-password').value = '';
      _loginResolvers.splice(0).forEach(r => r());
    } else {
      err.textContent = res.status === 401 ? 'Wrong password.' : `Login failed (${res.status}).`;
      err.classList.remove('hidden');
    }
  });
}
async function ensureAuthenticated() {
  try {
    const res = await _rawFetch('/api/auth/status');
    const data = await res.json();
    if (data.required && !data.authenticated) await showLoginOverlay();
  } catch (err) {
    console.warn('auth status check failed:', err);
  }
}
// Every API call goes through window.fetch; a 401 means "log in, then retry the same request".
const _rawFetch = window.fetch.bind(window);
window.fetch = async function (input, init) {
  const res = await _rawFetch(input, init);
  if (res.status !== 401) return res;
  const url = typeof input === 'string' ? input : (input && input.url) || '';
  if (url.includes('/api/login') || url.includes('/api/auth/status')) return res;
  await showLoginOverlay();
  return _rawFetch(input, init);
};

// "streamed-gathering-treehouse" plan Fix D, 2026-09-15: the player currently shown in the
// results dashboard -- starts as whatever `primary_player` the backend resolved (the run's real
// target, see `_resolve_primary_player_jersey` in `apps/api/main.py`) and moves to whichever
// player the owner picks via `renderPlayerSwitcher`'s chips on a multi-player run. `playReel`/
// `exportStatCard` read THIS, not `currentResults.primary_player` directly, so a switcher click
// actually changes what those actions operate on.
let activePlayer = null;

let selectedClickPoint = null;
// "streamed-gathering-treehouse" plan Stage 1: the frame-players picker used to overwrite a SINGLE
// `selectedFramePlayer` scalar on every click, so a second click in the same take (the "player left
// frame and came back with a new track id" case) silently discarded the first. `targetAnchors` is
// now an array that APPENDS on every picker click -- each entry names an EXACT
// `(take_id, raw_track_id)` pair, needing no coordinate-to-box matching on the server at all, and
// takes priority over `selectedClickPoint` (the raw coordinate-click fallback, kept working
// unchanged for backwards compatibility -- see `startPipelineProcessing`).
let targetAnchors = []; // [{ take_id, raw_track_id, chain_id, t }, ...]

function initApp() {
  checkHealth();
  loadVideos();

  document.getElementById('btn-process').addEventListener('click', startPipelineProcessing);
  document.getElementById('video-select').addEventListener('change', updatePreviewVideo);
  initClickToTrack();
  initFramePlayersPicker();
}

function updatePreviewVideo() {
  const videoName = document.getElementById('video-select').value;
  const preview = document.getElementById('preview-video');
  if (videoName && preview) {
    // Handle path encoding for space characters
    const encodedName = encodeURIComponent(videoName).replace(/%2F/g, '/');
    preview.src = `/media/${encodedName}`;
    preview.load();
    selectedClickPoint = null;
    targetAnchors = [];
    document.getElementById('click-marker')?.classList.add('hidden');
    document.getElementById('click-hint').innerHTML = `<i class="fa-solid fa-info-circle"></i> Pause video at any frame and click directly on the target player to pin a tracking candidate.`;
    document.getElementById('frame-players-wrapper')?.classList.add('hidden');
    document.getElementById('frame-players-hint')?.classList.add('hidden');
    document.getElementById('track-id-input').value = '';
    renderAnchorChips();
  }
}

function setJersey(num) {
  document.getElementById('target-jersey-input').value = num;
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  if (event && event.target) event.target.classList.add('active');
  document.getElementById('identity-target-badge').textContent = `Target Jersey #${num}`;
  const lbl = document.getElementById('click-marker-label');
  if (lbl) lbl.textContent = `#${num} | TARGET`;
}

function initClickToTrack() {
  const wrapper = document.getElementById('click-preview-wrapper');
  const video = document.getElementById('preview-video');
  const marker = document.getElementById('click-marker');
  const hint = document.getElementById('click-hint');

  if (!wrapper || !video) return;

  wrapper.addEventListener('click', (e) => {
    // Exclude clicks on video native controls bar at bottom (bottom 40px)
    const rect = video.getBoundingClientRect();
    const clickYRel = e.clientY - rect.top;
    if (clickYRel > rect.height - 40) return;

    // Pause video at click frame -- this holds regardless of the branch below: it is how the
    // owner's own stated workflow ("click the player, seek forward, click again, seek again...")
    // lands on an exact frame, whether that click is about to pin a raw-coordinate candidate or
    // (once anchors already exist, see the early return just below) simply line up the moment
    // they're about to add via "Load Players at This Moment".
    video.pause();

    // "streamed-gathering-treehouse" plan Fix C, 2026-09-15 (recheck finding C): once one or more
    // picker anchors already exist, a click anywhere on the video preview is PLAYBACK
    // interaction -- seeking to a new moment -- not a fresh targeting command. The bug this fixes:
    // this handler used to unconditionally run `targetAnchors = [];` below, so the very act of
    // clicking the video to move to the next timestamp silently discarded every anchor the picker
    // had accumulated -- only one anchor ever survived to submission, defeating the owner's
    // explicit multi-click-across-timestamps workflow. Once there is at least one chip, this
    // branch does NOT touch `targetAnchors` at all (no marker, no wipe) -- the chips' own ✕
    // (`removeAnchor`) is the only way to remove one, and "Load Players at This Moment" is the
    // only way to add another at the newly-sought timestamp.
    if (targetAnchors.length > 0) {
      if (hint) {
        hint.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${targetAnchors.length} anchor(s) pinned already -- this click just paused/sought the video. Use <em>"Load Players at This Moment"</em> to add another anchor here, or remove one with its chip's ✕ below.`;
      }
      return;
    }

    const xPct = (e.clientX - rect.left) / rect.width;
    const yPct = (e.clientY - rect.top) / rect.height;
    const t = video.currentTime || 0.0;
    // "streamed-gathering-treehouse" plan Stage E, 2026-09-04: a click is a CANDIDATE for the
    // persistent target identity, not a command that locks it in -- the server verifies it
    // (kit colour / jersey / height) and may REJECT it, so this UI must never fabricate a jersey
    // number (the old `|| '10'` fallback silently claimed "#10" for every click when the jersey
    // field was left blank) or claim the target is "Locked" before that verification happens.
    const jerseyValue = document.getElementById('target-jersey-input').value.trim();

    // A raw coordinate click stays a single, EXCLUSIVE candidate (backwards compatible with the
    // pre-Stage-1 behaviour) -- reachable ONLY while the chip list is empty (the early return
    // above), so there is nothing left here to wipe; once a picker anchor exists this branch
    // never runs again for a plain video click.
    selectedClickPoint = { x: xPct, y: yPct, t: t };
    document.getElementById('track-id-input').value = '';
    document.getElementById('frame-players-wrapper')?.classList.add('hidden');

    if (marker) {
      marker.style.left = `${xPct * 100}%`;
      marker.style.top = `${yPct * 100}%`;
      marker.classList.remove('hidden');
      const lbl = document.getElementById('click-marker-label');
      if (lbl) lbl.textContent = jerseyValue ? `#${jerseyValue}? CANDIDATE` : 'CANDIDATE';
    }

    if (hint) {
      hint.innerHTML = `<i class="fa-solid fa-crosshair" style="color:#ef4444;"></i> <strong style="color:#ef4444;">Candidate pinned @ ${t.toFixed(1)}s!</strong> The server verifies this click against the tracked target (and may reject it) before locking anything in. Click <em>"Run Player Tracking & Analytics"</em> below.`;
    }
  });
}

/**
 * Click-to-track player picker (CLAUDE.md Golden Rule 4: a human-provided identity is the
 * strongest evidence this pipeline has). Fetches `/api/frame_players` for the video's CURRENT
 * paused time and renders one clickable box per detected player, drawn on the SNAPSHOT the
 * endpoint returns rather than on the <video> element -- the snapshot's own pixel dimensions
 * match the returned box coordinates exactly (no `object-fit: contain` letterbox math needed,
 * which the raw click-on-video path above has to approximate and can get wrong on a video whose
 * aspect ratio differs from the fixed-height preview box).
 */
function initFramePlayersPicker() {
  const btn = document.getElementById('btn-load-players');
  if (!btn) return;
  btn.addEventListener('click', loadFramePlayers);
}

async function loadFramePlayers() {
  const videoName = document.getElementById('video-select').value;
  const preview = document.getElementById('preview-video');
  const hint = document.getElementById('frame-players-hint');
  const btn = document.getElementById('btn-load-players');
  if (!videoName || !preview) {
    alert('Select a video first.');
    return;
  }

  preview.pause();
  const t = preview.currentTime || 0.0;

  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Detecting players...`;
  hint.classList.remove('hidden');
  hint.textContent = `Loading players at t=${t.toFixed(1)}s -- this runs detection+tracking once per video (cached after the first time), so it may take a while the first time.`;

  try {
    const url = `/api/frame_players?video=${encodeURIComponent(videoName)}&t=${t}`;
    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderFramePlayers(data, videoName);
    hint.textContent = data.players.length
      ? `${data.players.length} player(s) detected at t=${data.t.toFixed(1)}s (take ${data.take_id}). Click the one you want to track.`
      : `No players detected at t=${data.t.toFixed(1)}s -- try a different moment.`;
  } catch (err) {
    hint.textContent = `Could not load players: ${err.message}`;
    logTerminal(`frame_players error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-users-viewfinder"></i> Load Players at This Moment`;
  }
}

function renderFramePlayers(data, videoName) {
  const wrapper = document.getElementById('frame-players-wrapper');
  const img = document.getElementById('frame-players-image');
  const boxesLayer = document.getElementById('frame-players-boxes');
  if (!wrapper || !img || !boxesLayer) return;

  img.src = `data:image/jpeg;base64,${data.frame_jpeg_base64}`;
  boxesLayer.innerHTML = '';
  wrapper.classList.remove('hidden');

  // The overlay layer must exactly cover the <img>, which is `display:block; width:100%;
  // height:auto` (no letterboxing, see the CSS comment) -- an absolutely positioned sibling with
  // inset:0 tracks the image's own rendered box regardless of its natural aspect ratio.
  boxesLayer.style.position = 'absolute';
  boxesLayer.style.inset = '0';

  data.players.forEach((p) => {
    const box = document.createElement('div');
    box.className = 'player-box';
    box.style.left = `${p.bbox_norm.x1 * 100}%`;
    box.style.top = `${p.bbox_norm.y1 * 100}%`;
    box.style.width = `${(p.bbox_norm.x2 - p.bbox_norm.x1) * 100}%`;
    box.style.height = `${(p.bbox_norm.y2 - p.bbox_norm.y1) * 100}%`;

    const label = document.createElement('span');
    label.className = 'player-box-label';
    const hint = p.jersey_hint ? ` #${p.jersey_hint.digits}?` : '';
    label.textContent = `ID ${p.chain_id}${hint}`;
    box.appendChild(label);

    box.addEventListener('click', () => {
      // "streamed-gathering-treehouse" plan Stage 1: APPEND, never overwrite -- a player who left
      // frame and came back gets a NEW track id per reappearance, so this needs to be a growing
      // set of anchors, not a single scalar (the pre-existing bug: a second click in the same take
      // used to silently discard the first).
      box.classList.add('selected');
      targetAnchors.push({
        take_id: data.take_id,
        track_id: p.raw_track_id,
        chain_id: p.chain_id,
        t: data.t,
      });
      selectedClickPoint = null; // a picker selection supersedes the raw coordinate-click fallback
      document.getElementById('click-marker')?.classList.add('hidden');
      renderAnchorChips();
      document.getElementById('click-hint').innerHTML =
        `<i class="fa-solid fa-crosshair" style="color:#10b981;"></i> <strong>Player (chain ID ${p.chain_id}) pinned @ t=${data.t.toFixed(1)}s!</strong> ${targetAnchors.length > 1 ? `${targetAnchors.length} anchors pinned so far -- click more (e.g. after the player re-enters frame) or ` : ''}Click <em>"Run Player Tracking & Analytics"</em> below.`;
    });

    boxesLayer.appendChild(box);
  });
}

/**
 * "streamed-gathering-treehouse" plan Stage 1: render one removable chip per accumulated click
 * anchor ("take 0 · track 44 · 0:48 ✕") and keep `#track-id-input` in sync as a human-readable,
 * still hand-editable, `take:track` mirror of the same set -- typing into that field directly
 * continues to work exactly as before (this function only ever WRITES to it in response to a
 * picker click/chip removal, it never reads it, so nothing here fights a manual edit made in
 * between two clicks).
 */
function renderAnchorChips() {
  const container = document.getElementById('target-anchor-chips');
  if (!container) return;
  container.innerHTML = '';

  if (targetAnchors.length === 0) {
    container.classList.add('hidden');
    return;
  }
  container.classList.remove('hidden');

  targetAnchors.forEach((anchor, idx) => {
    const chip = document.createElement('span');
    chip.style.cssText = 'display:inline-flex; align-items:center; gap:6px; background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.4); border-radius:14px; padding:4px 6px 4px 12px; font-size:12px; font-weight:600; color:#10b981; font-family:"JetBrains Mono", monospace;';
    const tLabel = typeof anchor.t === 'number' ? `${anchor.t.toFixed(1)}s` : '?';
    chip.innerHTML = `<span>take ${anchor.take_id} &middot; track ${anchor.track_id} &middot; ${tLabel}</span>`;

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.setAttribute('aria-label', `Remove anchor take ${anchor.take_id} track ${anchor.track_id}`);
    removeBtn.style.cssText = 'background:none; border:none; color:inherit; cursor:pointer; font-weight:800; padding:0 2px; line-height:1;';
    removeBtn.textContent = '✕';
    removeBtn.addEventListener('click', () => removeAnchor(idx));
    chip.appendChild(removeBtn);

    container.appendChild(chip);
  });

  document.getElementById('track-id-input').value = targetAnchors
    .map((a) => `${a.take_id}:${a.track_id}`)
    .join(',');
}

function removeAnchor(index) {
  targetAnchors.splice(index, 1);
  renderAnchorChips();
  document.getElementById('click-hint').innerHTML = targetAnchors.length
    ? `<i class="fa-solid fa-crosshair" style="color:#10b981;"></i> <strong>${targetAnchors.length} anchor(s) pinned.</strong> Click <em>"Run Player Tracking & Analytics"</em> below.`
    : `<i class="fa-solid fa-info-circle"></i> Pause the video at any frame and click directly on the target player to pin the tracking target -- shown at full size for precise clicking.`;
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.gpu_available) {
      document.getElementById('gpu-status').innerHTML = `<i class="fa-solid fa-microchip"></i> ${data.gpu_name} (CUDA)`;
    } else {
      document.getElementById('gpu-status').innerHTML = `<i class="fa-solid fa-cpu"></i> CPU Mode`;
    }
  } catch (err) {
    console.warn('Backend health check failed:', err);
  }
}

async function loadVideos() {
  try {
    const res = await fetch('/api/videos');
    const data = await res.json();
    const select = document.getElementById('video-select');
    if (data.input_videos && data.input_videos.length > 0) {
      select.innerHTML = '';
      data.input_videos.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = `${v.name} (${v.size_mb} MB)`;
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn('Error loading video list:', err);
  }
}

function setJersey(num) {
  document.getElementById('target-jersey-input').value = num;
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('identity-target-badge').textContent = `Target Jersey #${num}`;
}

function toggleCollapsible(id) {
  const elem = document.getElementById(id);
  elem.classList.toggle('hidden');
}

function initUploadZone() {
  const zone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');

  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.style.borderColor = '#10b981';
  });
  zone.addEventListener('dragleave', () => zone.style.borderColor = '');
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.style.borderColor = '';
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileUpload(fileInput.files[0]);
    }
  });
}

// Chunk size for uploads. Hosted deployments sit behind Cloudflare, whose free plan rejects any
// single request body over 100 MB -- a 4K clip is routinely several hundred MB -- so the browser
// slices the file and the server (`/api/upload/chunk`) reassembles it. Well under the cap on
// purpose: form-data framing adds a little, and a smaller slice retries cheaper on a flaky link.
const UPLOAD_CHUNK_BYTES = 24 * 1024 * 1024;

async function uploadInChunks(file) {
  const uploadId = (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`)
    .replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64);
  const total = Math.max(1, Math.ceil(file.size / UPLOAD_CHUNK_BYTES));
  let data = null;
  for (let index = 0; index < total; index++) {
    const start = index * UPLOAD_CHUNK_BYTES;
    const formData = new FormData();
    formData.append('upload_id', uploadId);
    formData.append('index', String(index));
    formData.append('total', String(total));
    formData.append('filename', file.name);
    formData.append('chunk', file.slice(start, start + UPLOAD_CHUNK_BYTES), file.name);
    const res = await fetch('/api/upload/chunk', { method: 'POST', body: formData });
    data = await res.json();
    if (!res.ok) throw new Error(data.detail || `chunk ${index + 1}/${total} failed (${res.status})`);
    if (total > 1 && (index + 1) % 4 === 0 && index + 1 < total) {
      logTerminal(`Uploading... ${Math.round(((index + 1) / total) * 100)}%`, 'info');
    }
  }
  return data;
}

async function handleFileUpload(file) {
  logTerminal(`Uploading file '${file.name}' (${(file.size / 1024 / 1024).toFixed(2)} MB)...`, 'info');

  try {
    const data = await uploadInChunks(file);
    if (data.status === 'success') {
      logTerminal(`File uploaded successfully: ${data.filename}`, 'success');
      await loadVideos();
      const select = document.getElementById('video-select');
      let exists = Array.from(select.options).some(opt => opt.value === data.filename);
      if (!exists) {
        const opt = document.createElement('option');
        opt.value = data.filename;
        opt.textContent = `${data.filename} (${data.size_mb} MB)`;
        select.appendChild(opt);
      }
      select.value = data.filename;
      updatePreviewVideo();
    } else {
      logTerminal(`Upload failed: ${data.detail || 'Unknown error'}`, 'error');
    }
  } catch (err) {
    logTerminal(`Upload error: ${err.message}`, 'error');
  }
}

async function startPipelineProcessing() {
  const videoName = document.getElementById('video-select').value;
  // "streamed-gathering-treehouse" plan Stage E, 2026-09-04: this used to be
  // `parseInt(...) || 10`, which silently sent jersey #10 to the API whenever the field was left
  // blank OR contained a non-numeric value -- the confirmed real cause of
  // `work/chelsea_burnley_target2/selection.json` carrying `target_jersey: 10` for a clip whose
  // own filename says `target2`. A blank/invalid field now means "no jersey number given"
  // (`null`), matching `ProcessRequest.target_jersey`'s own corrected default -- identity comes
  // from the verified click/profile (Stage E), not from a number nobody actually entered.
  const jerseyRaw = document.getElementById('target-jersey-input').value.trim();
  const parsedJersey = jerseyRaw ? parseInt(jerseyRaw, 10) : NaN;
  const targetJersey = Number.isNaN(parsedJersey) ? null : parsedJersey;
  const trackId = document.getElementById('track-id-input').value.trim();
  const manualAnnotations = document.getElementById('manual-annotations-input').value;
  // Plan Stage 2 ("streamed-gathering-treehouse", 2026-09-14): optional, like the jersey field --
  // `null`, not an empty string, when left blank, matching `ProcessRequest.touch_times`'s own
  // optional default.
  const touchTimesRaw = document.getElementById('touch-times-input')?.value.trim();

  if (!videoName) {
    alert('Please select or upload a video clip first.');
    return;
  }

  logTerminal(
    targetJersey !== null
      ? `Starting analysis for '${videoName}' with target jersey #${targetJersey}...`
      : `Starting analysis for '${videoName}' (no jersey number given -- identity comes from the click/profile)...`,
    'info'
  );
  document.getElementById('btn-process').disabled = true;
  document.getElementById('btn-process').innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Video...`;

  try {
    const payload = {
      video_name: videoName,
      target_jersey: targetJersey,
      // `#track-id-input` is read fresh here (not from `targetAnchors` directly) so a manual edit
      // made after the last picker click still wins, exactly as it always has.
      track_id: trackId || null,
      manual_annotations: manualAnnotations,
      touch_times: touchTimesRaw || null,
    };

    // "streamed-gathering-treehouse" plan Stage 1: every accumulated picker-click anchor rides
    // along as its own structured `target_clicks` entry -- the AUTHORITATIVE multi-anchor channel
    // (the server merges it with `track_id` above, so sending both is safe: real duplicates are
    // deduped server-side, see `normalize_manual_overrides`). `null`, not `[]`, when nothing was
    // clicked, so the server can tell "no clicks" apart from "clicks that all got removed".
    if (targetAnchors.length > 0) {
      payload.target_clicks = targetAnchors.map((a) => ({
        take_id: a.take_id,
        track_id: a.track_id,
        t: a.t,
      }));
      logTerminal(`Submitting ${targetAnchors.length} click anchor(s) across ${new Set(targetAnchors.map((a) => a.take_id)).size} take(s).`, 'info');
    }

    if (selectedClickPoint) {
      payload.click_x = selectedClickPoint.x;
      payload.click_y = selectedClickPoint.y;
      payload.click_t = selectedClickPoint.t;
      logTerminal(`Pinning click target at (${(selectedClickPoint.x*100).toFixed(1)}%, ${(selectedClickPoint.y*100).toFixed(1)}%) @ t=${selectedClickPoint.t.toFixed(1)}s`, 'info');
    }

    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    activeJobId = data.job_id;
    startJobPolling(activeJobId);
  } catch (err) {
    logTerminal(`Error launching pipeline: ${err.message}`, 'error');
    resetProcessBtn();
  }
}

function startJobPolling(jobId) {
  if (pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(() => checkJobStatus(jobId), 1000);
}

async function checkJobStatus(jobId) {
  try {
    const res = await fetch(`/api/status/${jobId}`);
    const job = await res.json();

    updateProgressBar(job.progress, job.stage);
    document.getElementById('job-status-badge').textContent = job.status.toUpperCase();

    // Highlight flow diagram node based on stage
    highlightFlowStage(job.stage);

    if (job.logs && job.logs.length > 0) {
      const lastLog = job.logs[job.logs.length - 1];
      logTerminal(lastLog, job.status === 'failed' ? 'error' : 'info');
    }

    if (job.status === 'completed') {
      clearInterval(pollingTimer);
      logTerminal('Analysis completed! Fetching results dashboard...', 'success');
      resetProcessBtn();
      loadResults(job.slug || jobId);
    } else if (job.status === 'failed') {
      clearInterval(pollingTimer);
      logTerminal(`Job failed: ${job.error}`, 'error');
      resetProcessBtn();
    }
  } catch (err) {
    console.warn('Status poll error:', err);
  }
}

function highlightFlowStage(stageName) {
  const nodes = ['node-detect', 'node-track', 'node-ocr', 'node-identity', 'node-analytics', 'node-stats'];
  nodes.forEach(n => document.getElementById(n)?.classList.remove('active'));

  if (!stageName) return;
  const s = stageName.toLowerCase();

  if (s.includes('detect')) document.getElementById('node-detect')?.classList.add('active');
  else if (s.includes('track')) document.getElementById('node-track')?.classList.add('active');
  else if (s.includes('ocr') || s.includes('jersey')) document.getElementById('node-ocr')?.classList.add('active');
  else if (s.includes('identity')) document.getElementById('node-identity')?.classList.add('active');
  else if (s.includes('event') || s.includes('movement')) document.getElementById('node-analytics')?.classList.add('active');
  else if (s.includes('render') || s.includes('stat')) document.getElementById('node-stats')?.classList.add('active');
}

function updateProgressBar(percent, stageText) {
  document.getElementById('progress-fill').style.width = `${percent}%`;
  document.getElementById('progress-percentage').textContent = `${percent}%`;
  document.getElementById('current-stage-name').textContent = stageText || 'Processing...';
}

function resetProcessBtn() {
  const btn = document.getElementById('btn-process');
  btn.disabled = false;
  btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Player Tracking & Analytics`;
}

function logTerminal(msg, type = 'info') {
  const logs = document.getElementById('terminal-logs');
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  const now = new Date().toLocaleTimeString();
  entry.innerHTML = `<span class="log-time">[${now}]</span> ${msg}`;
  logs.appendChild(entry);
  logs.scrollTop = logs.scrollHeight;
}

async function loadResults(slug) {
  try {
    const res = await fetch(`/api/results/${slug}`);
    if (!res.ok) {
      // Try fallback slug
      const slugFallback = slug.replace(/\.mp4$/, '').replace(/\s+/g, '_');
      const res2 = await fetch(`/api/results/${slugFallback}`);
      if (res2.ok) {
        currentResults = await res2.json();
      } else {
        throw new Error('Results endpoint returned 404');
      }
    } else {
      currentResults = await res.json();
    }

    renderDashboard(currentResults);
  } catch (err) {
    logTerminal(`Could not load results for ${slug}: ${err.message}`, 'error');
  }
}

function renderDashboard(data) {
  const player = data.primary_player || data.players[0];
  activePlayer = player;

  document.getElementById('dash-video-name').textContent = data.slug;
  renderPlayerSwitcher(data);
  renderPlayerStats(player);

  // Video Source
  const videoPlayer = document.getElementById('annotated-video-player');
  if (data.annotated_video_url) {
    videoPlayer.src = data.annotated_video_url;
    videoPlayer.load();
  }

  // Render 2D Tactical Pitch Canvas
  drawTacticalPitch(data.pitch_trajectory || []);

  // Show Dashboard -- it starts `hidden` in index.html so a fresh page never shows a stale
  // annotated video + placeholder stat values as if they were a real result (Golden Rule 5:
  // nothing on screen should look like a finished analysis until one actually finished).
  const dashboard = document.getElementById('results-dashboard');
  dashboard.classList.remove('hidden');
  dashboard.scrollIntoView({ behavior: 'smooth' });
}

/**
 * "streamed-gathering-treehouse" plan Fix D, 2026-09-15: fills in every per-player field of the
 * dashboard header/stat panel/timeline for ONE player -- factored out of `renderDashboard` so
 * `renderPlayerSwitcher`'s chip clicks can re-run exactly this without re-fetching or re-drawing
 * the (shared, not per-player) video/pitch canvas.
 */
function renderPlayerStats(player) {
  const jerseyNum = player.jersey_number || 11;

  // Header Summary
  document.getElementById('dash-jersey-num').textContent = jerseyNum;
  document.getElementById('statcard-jersey-tag').textContent = `#${jerseyNum}`;
  document.getElementById('dash-player-title').textContent = `TARGET PLAYER #${jerseyNum}`;
  document.getElementById('dash-identity-status').innerHTML = `<i class="fa-solid fa-shield-check"></i> Identity: ${player.identity_status}`;

  // Stat Card Metrics (#11: touches, passes, turnovers, sprints, goals, assists, shots, tackles, saves)
  document.getElementById('stat-goals').textContent = player.goals || 0;
  document.getElementById('stat-assists').textContent = player.assists || 0;
  document.getElementById('stat-touches').textContent = player.touches || 0;
  document.getElementById('stat-passes').textContent = player.passes || 0;
  document.getElementById('stat-sprints').textContent = player.sprints || 0;
  document.getElementById('stat-shots').textContent = player.shots || 0;
  document.getElementById('stat-tackles').textContent = player.tackles || 0;
  document.getElementById('stat-saves').textContent = player.saves || 0;
  document.getElementById('stat-turnovers').textContent = player.turnovers || 0;
  document.getElementById('stat-possession').textContent = player.possession_time || 'uncertain';
  document.getElementById('stat-distance').textContent = player.distance_covered || 'uncertain';

  // Render Timestamped Event Timeline
  renderEventsTable(player.events || []);
}

/**
 * "streamed-gathering-treehouse" plan Fix D, 2026-09-15: `/api/results/{slug}` now resolves one
 * real `primary_player` (the run's own actual target -- `target.json`/`selection.json`, never a
 * lexicographic accident), but a run can genuinely produce more than one known jersey number
 * (e.g. two separate verified takes, or a multi-anchor click run whose chain fragments across two
 * players). This renders one small chip per player so the owner can look at -- and download the
 * statcard for -- whichever one they want, instead of only ever seeing `primary_player`. Hidden
 * entirely on the common single-player case (an always-visible switcher with one disabled option
 * would just be clutter).
 */
function renderPlayerSwitcher(data) {
  const container = document.getElementById('player-switcher');
  if (!container) return;
  container.innerHTML = '';

  if (!data.players || data.players.length < 2) {
    container.classList.add('hidden');
    return;
  }
  container.classList.remove('hidden');

  data.players.forEach((p) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip' + (p === activePlayer ? ' active' : '');
    btn.textContent = `#${p.jersey_number}`;
    btn.title = `View player #${p.jersey_number}'s stats and highlights`;
    btn.addEventListener('click', () => {
      activePlayer = p;
      renderPlayerStats(p);
      renderPlayerSwitcher(data);
    });
    container.appendChild(btn);
  });
}

function renderEventsTable(events) {
  const tbody = document.getElementById('events-tbody');
  const markersList = document.getElementById('event-markers-list');

  tbody.innerHTML = '';
  markersList.innerHTML = '';

  document.getElementById('timeline-count-badge').textContent = `${events.length} Events`;

  if (events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: #9ca3af;">No specific events detected in clip</td></tr>`;
    return;
  }

  events.forEach(ev => {
    // Table Row
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><i class="fa-solid fa-clock"></i> ${ev.time}</td>
      <td><strong>${ev.event}</strong></td>
      <td><span class="badge">${(ev.confidence * 100).toFixed(0)}%</span></td>
      <td><button class="chip" onclick="seekVideo('${ev.time}')"><i class="fa-solid fa-play"></i> Seek</button></td>
    `;
    tbody.appendChild(tr);

    // Video Timeline Marker
    const marker = document.createElement('button');
    marker.className = 'chip';
    marker.innerHTML = `<i class="fa-solid fa-bookmark"></i> ${ev.event} @ ${ev.time}`;
    marker.onclick = () => seekVideo(ev.time);
    markersList.appendChild(marker);
  });
}

function seekVideo(timeStr) {
  const player = document.getElementById('annotated-video-player');
  const parts = timeStr.split(':');
  let seconds = 0;
  if (parts.length === 2) {
    seconds = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
  } else if (parts.length === 3) {
    seconds = parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
  } else {
    seconds = parseFloat(timeStr);
  }

  player.currentTime = seconds;
  player.play();
}

function initCanvasPitch() {
  const canvas = document.getElementById('pitch-canvas');
  if (!canvas) return;
  drawSoccerPitchField(canvas);
}

function togglePitchMode(mode) {
  pitchMode = mode;
  document.querySelectorAll('.pitch-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  if (currentResults && currentResults.pitch_trajectory) {
    drawTacticalPitch(currentResults.pitch_trajectory);
  }
}

function drawSoccerPitchField(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Background Grass
  ctx.fillStyle = '#04140b';
  ctx.fillRect(0, 0, w, h);

  // Pitch Lines Style
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 2;

  // Outer Boundary
  ctx.strokeRect(20, 20, w - 40, h - 40);

  // Center Line & Circle
  ctx.beginPath();
  ctx.moveTo(w / 2, 20);
  ctx.lineTo(w / 2, h - 20);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(w / 2, h / 2, 45, 0, Math.PI * 2);
  ctx.stroke();

  // Penalty Boxes (Left & Right)
  ctx.strokeRect(20, (h - 140) / 2, 80, 140);
  ctx.strokeRect(w - 100, (h - 140) / 2, 80, 140);

  // Goal Mouth Boxes
  ctx.strokeRect(20, (h - 60) / 2, 30, 60);
  ctx.strokeRect(w - 50, (h - 60) / 2, 30, 60);
}

function drawTacticalPitch(points) {
  const canvas = document.getElementById('pitch-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  drawSoccerPitchField(canvas);

  if (!points || points.length === 0) return;

  if (pitchMode === 'trajectory') {
    // Draw continuous path trajectory line
    ctx.beginPath();
    ctx.lineWidth = 3;

    points.forEach((pt, idx) => {
      const px = 20 + (pt.x / 100) * (w - 40);
      const py = 20 + (pt.y / 100) * (h - 40);

      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });

    ctx.strokeStyle = '#10b981';
    ctx.stroke();

    // Draw individual action points
    points.forEach((pt, idx) => {
      const px = 20 + (pt.x / 100) * (w - 40);
      const py = 20 + (pt.y / 100) * (h - 40);

      ctx.beginPath();
      if (pt.action === 'Sprint') {
        ctx.fillStyle = '#f59e0b';
        ctx.arc(px, py, 5, 0, Math.PI * 2);
      } else if (pt.action === 'Pass' || pt.action === 'Shot' || pt.action === 'Goal') {
        ctx.fillStyle = '#06b6d4';
        ctx.arc(px, py, 7, 0, Math.PI * 2);
      } else {
        ctx.fillStyle = '#10b981';
        ctx.arc(px, py, 3, 0, Math.PI * 2);
      }
      ctx.fill();
    });

  } else {
    // Heatmap mode
    points.forEach(pt => {
      const px = 20 + (pt.x / 100) * (w - 40);
      const py = 20 + (pt.y / 100) * (h - 40);

      const radGrd = ctx.createRadialGradient(px, py, 2, px, py, 35);
      radGrd.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
      radGrd.addColorStop(0.6, 'rgba(245, 158, 11, 0.2)');
      radGrd.addColorStop(1, 'rgba(0, 0, 0, 0)');

      ctx.fillStyle = radGrd;
      ctx.fillRect(px - 35, py - 35, 70, 70);
    });
  }
}

function playReel(category) {
  // Plan Fix D, 2026-09-15: reads `activePlayer` (whichever player the switcher currently has
  // selected), not always `currentResults.primary_player` -- so on a multi-player run this plays
  // the reel for the player actually on screen, not always the backend-resolved primary one.
  if (!activePlayer) {
    alert('Please process or select a match video first.');
    return;
  }
  const highlights = activePlayer.highlights || {};
  if (highlights[category]) {
    const videoPlayer = document.getElementById('annotated-video-player');
    videoPlayer.src = highlights[category];
    videoPlayer.play();
    logTerminal(`Playing ${category} highlight reel`, 'info');
  } else {
    alert(`No ${category} highlight clips generated for this target player.`);
  }
}

function exportStatCard() {
  // Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14): this used to re-type an
  // approximation of the statcard client-side -- it omitted Dribbles entirely, rescaled
  // confidences, and (because the old `_safe_int` coercion turned the non-numeric
  // "not available (...)" string into a bare `0`) printed "Goals: 0" where the real statcard.md
  // carries the full "not available" explanation. That overclaimed certainty relative to the real
  // file (a Golden Rule 5 violation), so this now just downloads the real, server-rendered
  // `statcard.pdf` (`src.pipeline.statcard_pdf.render_statcard_pdf`) -- the SAME values
  // `statcard.md` uses, no second, drifting re-typing of the stats anywhere.
  // Plan Fix D, 2026-09-15: exports whichever player the switcher currently has active, not
  // always the backend-resolved `primary_player` -- see `renderPlayerSwitcher`.
  if (!currentResults || !activePlayer) {
    alert('No results loaded yet -- process a video first.');
    return;
  }
  const player = activePlayer;
  if (!player.statcard_pdf_url) {
    // Honest message rather than the old silent `return` (itself a minor Golden Rule 5 gap) --
    // e.g. `reportlab` wasn't installed on the server when this run happened.
    alert(
      'No downloadable stat-card PDF is available for this player (the PDF was not generated ' +
      'for this run). The full statcard.md is still viewable at ' +
      `/media/output/${currentResults.slug}/players/player_${player.jersey_number}/statcard.md`
    );
    return;
  }
  const a = document.createElement('a');
  a.href = player.statcard_pdf_url;
  a.download = `player_${player.jersey_number}_statcard.pdf`;
  a.click();
}

function downloadAnnotatedVideo() {
  if (!currentResults || !currentResults.annotated_video_url) return;
  const a = document.createElement('a');
  a.href = currentResults.annotated_video_url;
  a.download = `${currentResults.slug}_annotated.mp4`;
  a.click();
}
