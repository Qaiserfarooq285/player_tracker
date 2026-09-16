'use strict';
/**
 * Node-only behavioural harness for `apps/web/js/app.js`'s `initClickToTrack` click handler --
 * Plan Fix C ("streamed-gathering-treehouse" re-check, 2026-09-15): once one or more picker
 * anchors exist, a click on the video preview must be treated as playback interaction (seeking),
 * never a fresh targeting command that wipes `targetAnchors`.
 *
 * This repo has no `package.json`/JS test runner and no jsdom dependency (CLAUDE.md's own
 * dependency-license discipline, §7, has never had to consider a JS toolchain at all) -- rather
 * than add one for a single check, this loads and executes the REAL, UNMODIFIED `app.js` source
 * inside a minimal hand-built DOM stub (Node's built-in `vm` module) and drives it exactly like a
 * browser would: construct a fake click event, dispatch it to the listener `initClickToTrack`
 * itself attaches, and read back the resulting state/DOM side effects. Invoked from
 * `tests/test_click_anchor_js.py` via `subprocess` -- prints one JSON object to stdout.
 *
 * Node's `vm.runInContext` gives every call against the SAME `context` a shared top-level lexical
 * scope (the same way separate `<script>` tags on one HTML page share `let`/`const` globals) --
 * that is what lets the snippets below read/mutate app.js's own top-level `targetAnchors`/
 * `selectedClickPoint` `let` bindings directly, with no need for app.js to export anything.
 */
const fs = require('fs');
const vm = require('vm');

const appJsPath = process.argv[2];
const source = fs.readFileSync(appJsPath, 'utf8');

function makeClassList() {
  const set = new Set();
  return {
    add: (...names) => names.forEach((n) => set.add(n)),
    remove: (...names) => names.forEach((n) => set.delete(n)),
    contains: (n) => set.has(n),
    toggle: (n) => (set.has(n) ? set.delete(n) : set.add(n)),
  };
}

function makeElement() {
  const listeners = {};
  const el = {
    style: {},
    dataset: {},
    _value: '',
    _innerHTML: '',
    _textContent: '',
    children: [],
    get value() {
      return this._value;
    },
    set value(v) {
      this._value = v;
    },
    get innerHTML() {
      return this._innerHTML;
    },
    set innerHTML(v) {
      this._innerHTML = v;
    },
    get textContent() {
      return this._textContent;
    },
    set textContent(v) {
      this._textContent = v;
    },
    setAttribute() {},
    addEventListener(type, cb) {
      (listeners[type] = listeners[type] || []).push(cb);
    },
    dispatch(type, evt) {
      (listeners[type] || []).forEach((cb) => cb(evt));
    },
    getBoundingClientRect() {
      return { top: 0, left: 0, width: 800, height: 450 };
    },
    querySelector() {
      return null;
    },
    appendChild(child) {
      this.children.push(child);
    },
    pause() {
      this._paused = true;
    },
    get currentTime() {
      return this._currentTime || 0;
    },
    set currentTime(v) {
      this._currentTime = v;
    },
  };
  el.classList = makeClassList();
  return el;
}

// Elements `initClickToTrack`, `renderAnchorChips`, and `removeAnchor` touch by id -- matching
// `apps/web/index.html`'s real markup (`click-marker`/`frame-players-wrapper`/
// `target-anchor-chips` all start with the literal `class="hidden"`, reproduced below so a test
// that checks "did this become visible" is checking a REAL transition, not a default).
const elements = {};
[
  'click-preview-wrapper',
  'preview-video',
  'click-marker',
  'click-marker-label',
  'click-hint',
  'target-jersey-input',
  'track-id-input',
  'frame-players-wrapper',
  'target-anchor-chips',
].forEach((id) => {
  elements[id] = makeElement();
});
elements['click-marker'].classList.add('hidden');
elements['frame-players-wrapper'].classList.add('hidden');
elements['target-anchor-chips'].classList.add('hidden');

const documentStub = {
  getElementById: (id) => elements[id] || null,
  addEventListener() {},
  querySelectorAll: () => [],
  createElement: () => makeElement(),
};

// `fetch`: every browser has it, and app.js wraps `window.fetch` at load time for the hosted
// deployment's login gate (docs/DEPLOY.md). Never actually called by the click paths under test.
const sandbox = {
  document: documentStub,
  console,
  event: null,
  fetch: async () => ({ ok: true, status: 200, json: async () => ({}) }),
};
sandbox.window = sandbox;
const context = vm.createContext(sandbox);

vm.runInContext(source, context, { filename: appJsPath });
vm.runInContext('initClickToTrack();', context);

function click(clientX, clientY) {
  vm.runInContext(
    `document.getElementById('click-preview-wrapper').dispatch('click', ` +
      `{ clientX: ${clientX}, clientY: ${clientY} });`,
    context
  );
}

const results = {};

// Scenario 1 (backwards compatibility): chip list empty -> a raw click on the video still pins a
// single candidate, exactly like before Fix C.
click(100, 100);
results.emptyListPinsCandidate = vm.runInContext('selectedClickPoint !== null', context);
results.markerVisibleAfterFirstClick = vm.runInContext(
  "!document.getElementById('click-marker').classList.contains('hidden')",
  context
);

// Scenario 2 (the real bug): simulate the picker having accumulated two anchors -- pushing onto
// the exact same array `renderFramePlayers`'s own click handler pushes onto, never overwriting it
// -- then click the video preview again (the owner's own "seek to the next moment" action).
vm.runInContext(
  'targetAnchors.push({take_id:0, track_id:44, chain_id:1, t:5.0});' +
    'targetAnchors.push({take_id:0, track_id:51, chain_id:2, t:12.0});' +
    'renderAnchorChips();',
  context
);
results.anchorCountBeforeVideoClick = vm.runInContext('targetAnchors.length', context);
results.chipsVisibleBeforeVideoClick = vm.runInContext(
  "!document.getElementById('target-anchor-chips').classList.contains('hidden')",
  context
);

click(200, 200);

results.anchorCountAfterVideoClick = vm.runInContext('targetAnchors.length', context);
results.chipsStillVisibleAfterVideoClick = vm.runInContext(
  "!document.getElementById('target-anchor-chips').classList.contains('hidden')",
  context
);
// The chips array itself must be byte-identical, not just the same length.
results.anchorsUnchangedAfterVideoClick = vm.runInContext(
  'JSON.stringify(targetAnchors) === ' +
    'JSON.stringify([{take_id:0, track_id:44, chain_id:1, t:5.0}, ' +
    '{take_id:0, track_id:51, chain_id:2, t:12.0}])',
  context
);

process.stdout.write(JSON.stringify(results));
