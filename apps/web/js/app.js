/**
 * AI Football Player Tracking & Analytics Engine - Frontend Logic
 */

let activeJobId = null;
let pollingTimer = null;
let currentResults = null;
let pitchMode = 'trajectory';

document.addEventListener('DOMContentLoaded', () => {
  initApp();
  initUploadZone();
  initCanvasPitch();
});

let selectedClickPoint = null;
// Set by the frame-players picker (`/api/frame_players`) when the user clicks a detected box --
// takes priority over `selectedClickPoint` (the raw coordinate-click fallback) because it names an
// EXACT raw_track_id, needing no coordinate-to-box matching on the server at all.
let selectedFramePlayer = null; // { take_id, raw_track_id, chain_id }

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
    selectedFramePlayer = null;
    document.getElementById('click-marker')?.classList.add('hidden');
    document.getElementById('click-hint').innerHTML = `<i class="fa-solid fa-info-circle"></i> Pause video at any frame and click directly on player #10 to pin tracking target.`;
    document.getElementById('frame-players-wrapper')?.classList.add('hidden');
    document.getElementById('frame-players-hint')?.classList.add('hidden');
    document.getElementById('track-id-input').value = '';
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

    // Pause video at click frame
    video.pause();

    const xPct = (e.clientX - rect.left) / rect.width;
    const yPct = (e.clientY - rect.top) / rect.height;
    const t = video.currentTime || 0.0;
    const targetJersey = document.getElementById('target-jersey-input').value || '10';

    selectedClickPoint = { x: xPct, y: yPct, t: t };
    selectedFramePlayer = null;
    document.getElementById('track-id-input').value = '';
    document.getElementById('frame-players-wrapper')?.classList.add('hidden');

    if (marker) {
      marker.style.left = `${xPct * 100}%`;
      marker.style.top = `${yPct * 100}%`;
      marker.classList.remove('hidden');
      const lbl = document.getElementById('click-marker-label');
      if (lbl) lbl.textContent = `#${targetJersey} | TARGET`;
    }

    if (hint) {
      hint.innerHTML = `<i class="fa-solid fa-crosshair" style="color:#ef4444;"></i> <strong style="color:#ef4444;">Player #${targetJersey} Target Locked @ ${t.toFixed(1)}s!</strong> Click <em>"Run Player Tracking & Analytics"</em> below to trace player continuously in RED box.`;
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
      boxesLayer.querySelectorAll('.player-box.selected').forEach((el) => el.classList.remove('selected'));
      box.classList.add('selected');
      selectedFramePlayer = { take_id: data.take_id, raw_track_id: p.raw_track_id, chain_id: p.chain_id };
      selectedClickPoint = null; // a picker selection supersedes the raw coordinate-click fallback
      document.getElementById('click-marker')?.classList.add('hidden');
      document.getElementById('track-id-input').value = `${data.take_id}:${p.raw_track_id}`;
      document.getElementById('click-hint').innerHTML =
        `<i class="fa-solid fa-crosshair" style="color:#10b981;"></i> <strong>Player (chain ID ${p.chain_id}) pinned @ t=${data.t.toFixed(1)}s!</strong> Click <em>"Run Player Tracking & Analytics"</em> below.`;
    });

    boxesLayer.appendChild(box);
  });
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

async function handleFileUpload(file) {
  logTerminal(`Uploading file '${file.name}' (${(file.size / 1024 / 1024).toFixed(2)} MB)...`, 'info');
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
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
  const targetJersey = parseInt(document.getElementById('target-jersey-input').value) || 10;
  const trackId = document.getElementById('track-id-input').value.trim();
  const manualAnnotations = document.getElementById('manual-annotations-input').value;

  if (!videoName) {
    alert('Please select or upload a video clip first.');
    return;
  }

  logTerminal(`Starting analysis for '${videoName}' with target jersey #${targetJersey}...`, 'info');
  document.getElementById('btn-process').disabled = true;
  document.getElementById('btn-process').innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Video...`;

  try {
    const payload = {
      video_name: videoName,
      target_jersey: targetJersey,
      track_id: trackId || null,
      manual_annotations: manualAnnotations,
    };

    // Note: a frame-players picker selection doesn't need handling here -- its click handler
    // (`renderFramePlayers`) writes `take_id:raw_track_id` directly into `#track-id-input`, which
    // `trackId` above already reads fresh at submit time. Re-reading `selectedFramePlayer` here
    // too would risk overriding a manual edit to that field with a stale remembered value.
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
  const jerseyNum = player.jersey_number || 11;

  // Header Summary
  document.getElementById('dash-jersey-num').textContent = jerseyNum;
  document.getElementById('statcard-jersey-tag').textContent = `#${jerseyNum}`;
  document.getElementById('dash-player-title').textContent = `TARGET PLAYER #${jerseyNum}`;
  document.getElementById('dash-identity-status').innerHTML = `<i class="fa-solid fa-shield-check"></i> Identity: ${player.identity_status}`;
  document.getElementById('dash-video-name').textContent = data.slug;

  // Video Source
  const videoPlayer = document.getElementById('annotated-video-player');
  if (data.annotated_video_url) {
    videoPlayer.src = data.annotated_video_url;
    videoPlayer.load();
  }

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

  // Render 2D Tactical Pitch Canvas
  drawTacticalPitch(data.pitch_trajectory || []);

  // Show Dashboard
  document.getElementById('results-dashboard').scrollIntoView({ behavior: 'smooth' });
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
  if (!currentResults || !currentResults.primary_player) {
    alert('Please process or select a match video first.');
    return;
  }
  const highlights = currentResults.primary_player.highlights || {};
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
  if (!currentResults || !currentResults.primary_player) return;
  const player = currentResults.primary_player;

  let text = `# PLAYER STAT CARD - JERSEY #${player.jersey_number}\n\n`;
  text += `Identity Status: ${player.identity_status}\n`;
  text += `Touches: ${player.touches}\n`;
  text += `Passes (Same-Colour): ${player.passes}\n`;
  text += `Turnovers: ${player.turnovers}\n`;
  text += `Sprints/Runs: ${player.sprints}\n`;
  text += `Goals: ${player.goals}\n`;
  text += `Assists: ${player.assists}\n`;
  text += `Shots: ${player.shots}\n`;
  text += `Tackles: ${player.tackles}\n`;
  text += `Saves: ${player.saves}\n`;
  text += `Possession Time: ${player.possession_time}\n`;
  text += `Distance Covered: ${player.distance_covered}\n\n`;
  text += `## EVENT TIMELINE\n`;
  (player.events || []).forEach(e => {
    text += `- ${e.time} | ${e.event} | Conf: ${(e.confidence * 100).toFixed(0)}%\n`;
  });

  const blob = new Blob([text], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Player_${player.jersey_number}_StatCard.md`;
  a.click();
}

function downloadAnnotatedVideo() {
  if (!currentResults || !currentResults.annotated_video_url) return;
  const a = document.createElement('a');
  a.href = currentResults.annotated_video_url;
  a.download = `${currentResults.slug}_annotated.mp4`;
  a.click();
}
