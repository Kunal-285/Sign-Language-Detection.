

/* =============================================
   NEURAL NETWORK CANVAS
   ============================================= */
(function () {
    const canvas = document.getElementById('neuralCanvas');
    const ctx    = canvas.getContext('2d');
    const PRI    = '56, 189, 248';
    const SUC    = '34, 197, 94';
    let W, H, nodes = [], activeColor = PRI;

    function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
    window.addEventListener('resize', resize);
    resize();

    class Node {
        constructor() { this.reset(); }
        reset() {
            this.x = Math.random()*W; this.y = Math.random()*H;
            this.vx = (Math.random()-0.5)*0.35; this.vy = (Math.random()-0.5)*0.35;
            this.r = Math.random()*1.5+1.2; this.alpha = Math.random()*0.5+0.2;
            this.pulseAmt = 0; this.color = PRI;
        }
        update() {
            this.x += this.vx; this.y += this.vy;
            if (this.x<0||this.x>W) this.vx*=-1;
            if (this.y<0||this.y>H) this.vy*=-1;
            if (this.pulseAmt>0) this.pulseAmt = Math.max(0, this.pulseAmt-0.015);
        }
        draw() {
            const r = this.r + this.pulseAmt*4;
            const a = Math.min(1, this.alpha + this.pulseAmt*0.8);
            const c = this.pulseAmt>0.05 ? this.color : PRI;
            if (this.pulseAmt>0.05) {
                const g = ctx.createRadialGradient(this.x,this.y,0,this.x,this.y,r*6);
                g.addColorStop(0,`rgba(${c},${a*0.6})`); g.addColorStop(1,`rgba(${c},0)`);
                ctx.beginPath(); ctx.arc(this.x,this.y,r*6,0,Math.PI*2);
                ctx.fillStyle=g; ctx.fill();
            }
            ctx.beginPath(); ctx.arc(this.x,this.y,r,0,Math.PI*2);
            ctx.fillStyle=`rgba(${c},${a})`; ctx.fill();
        }
    }

    nodes = Array.from({length:70}, ()=>new Node());

    function render() {
        ctx.fillStyle='rgba(2,9,23,0.18)'; ctx.fillRect(0,0,W,H);
        for (let i=0;i<nodes.length;i++) {
            for (let j=i+1;j<nodes.length;j++) {
                const a=nodes[i],b=nodes[j];
                const dx=a.x-b.x, dy=a.y-b.y, dist=Math.sqrt(dx*dx+dy*dy);
                if (dist<180) {
                    const t=1-dist/180, pulse=Math.max(a.pulseAmt,b.pulseAmt);
                    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y);
                    ctx.strokeStyle=`rgba(${pulse>0.05?activeColor:PRI},${t*0.18+pulse*0.5})`;
                    ctx.lineWidth=t*1.2+pulse*1.5; ctx.stroke();
                }
            }
        }
        nodes.forEach(n=>{n.update();n.draw();});
        requestAnimationFrame(render);
    }
    render();

    window.neuralPulse = (color) => {
        activeColor = color||PRI;
        for (let i=0;i<6;i++) {
            const n=nodes[Math.floor(Math.random()*nodes.length)];
            n.pulseAmt=1; n.color=activeColor;
        }
    };
    window.neuralRipple = () => {
        const cx=W/2, cy=H/2;
        nodes.forEach(n=>{
            const d=Math.sqrt((n.x-cx)**2+(n.y-cy)**2);
            setTimeout(()=>{n.pulseAmt=Math.max(0.3,1-d/800);n.color=SUC;}, d*0.8);
        });
    };
})();

/* =============================================
   STATE
   ============================================= */
const state = {
    streaming    : false,
    sending      : false,      // prevents overlapping requests
    loopId       : null,
    totalFrames  : 0,
    successFrames: 0,
    confSum      : 0,
    sessionStart : null,
    sessionTimer : null,
    currentPred  : null,
    sentence     : [],
    fps          : { frames: 0, last: Date.now() },
    bufferPct    : 0,          // ✅ how full is the 30-frame buffer
};

// ✅ Reusable offscreen canvas — don't create a new one every frame
const offscreen    = document.createElement('canvas');
offscreen.width    = 320;
offscreen.height   = 240;
const offCtx       = offscreen.getContext('2d');

/* =============================================
   DOM
   ============================================= */
const $ = id => document.getElementById(id);
const dom = {
    video        : $('video'),
    videoWrapper : $('videoWrapper'),
    videoIdle    : $('videoIdle'),
    overlayPred  : $('overlayPred'),
    predOverlay  : $('predOverlay'),
    confBarFill  : $('confBarFill'),
    predText     : $('predictionText'),
    confFill     : $('confFill'),
    confGlow     : $('confGlow'),
    confPct      : $('confPct'),
    badge        : $('resultBadge'),
    sentDisplay  : $('sentenceDisplay'),
    wordCount    : $('wordCount'),
    logBody      : $('logBody'),
    logPulse     : $('logPulse'),
    statusPill   : $('statusPill'),
    statusText   : $('statusText'),
    fpsDisplay   : $('fpsDisplay'),
    frameCount   : $('frameCount'),
    totalFrames  : $('totalFrames'),
    successRate  : $('successRate'),
    avgConf      : $('avgConf'),
    sessionTime  : $('sessionTime'),
    btnStart     : $('btnStart'),
    btnStop      : $('btnStop'),
    toast        : $('toast'),
};

/* =============================================
   INIT
   ============================================= */
log('info', 'System ready. Press START to begin.');
log('info', 'SPACE=start/stop  ENTER=add word  BACKSPACE=undo');
setInterval(()=>{ if (!state.streaming) window.neuralPulse(); }, 2000);

/* =============================================
   CAMERA
   ============================================= */
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        dom.video.srcObject = stream;
        state.streaming = true;

        setStatus('online', 'LIVE');
        dom.videoWrapper.classList.add('active');
        dom.videoIdle.classList.add('hidden');
        dom.predOverlay.classList.add('visible');
        dom.btnStart.disabled = true;
        dom.btnStop.disabled  = false;
        dom.logPulse.classList.add('active');

        state.sessionStart = Date.now();
        state.sessionTimer = setInterval(updateTime, 1000);
        state.fps = { frames: 0, last: Date.now() };

        window.neuralPulse('56, 189, 248');
        log('success', 'Camera opened. Building frame buffer...');
        log('info', 'First prediction after 30 frames (~3 sec)');
        showToast('Camera started ▶');

        sendLoop();

    } catch (err) {
        log('error', 'Camera denied: ' + err.message);
        showToast('Camera access denied!');
    }
}

function stopCamera() {
    if (dom.video.srcObject) dom.video.srcObject.getTracks().forEach(t=>t.stop());
    state.streaming = false;
    state.sending   = false;
    clearTimeout(state.loopId);
    clearInterval(state.sessionTimer);

    // ✅ Reset buffer on server too
    fetch('/reset', { method: 'POST' }).catch(()=>{});

    setStatus('offline', 'OFFLINE');
    dom.videoWrapper.classList.remove('active');
    dom.videoIdle.classList.remove('hidden');
    dom.predOverlay.classList.remove('visible');
    dom.btnStart.disabled = false;
    dom.btnStop.disabled  = true;
    dom.logPulse.classList.remove('active');
    dom.fpsDisplay.textContent = '--';
    state.bufferPct = 0;

    updatePred(null, 0);
    log('warn', 'Stream stopped.');
    showToast('Camera stopped ■');
}

/* =============================================
   SEND LOOP
   ✅ Key fixes:
   - No frame skipping (was skipping 50% of frames)
   - Sends every 120ms instead of 300ms
   - Uses setTimeout not setInterval — waits for
     response before sending next frame
   - No overlapping requests
   ============================================= */
async function sendLoop() {
    if (!state.streaming) return;

    // Only send if previous request is done
    if (!state.sending && dom.video.videoWidth) {
        state.sending = true;

        try {
            // Draw to reusable canvas
            offCtx.drawImage(dom.video, 0, 0, 320, 240);

            const blob = await new Promise(res =>
                offscreen.toBlob(res, 'image/jpeg', 0.75)
            );

            const fd = new FormData();
            fd.append('frame', blob);

            const res  = await fetch('/predict', { method: 'POST', body: fd });
            const data = await res.json();

            // FPS counter
            state.fps.frames++;
            const now = Date.now();
            if (now - state.fps.last >= 1000) {
                dom.fpsDisplay.textContent = state.fps.frames;
                state.fps = { frames: 0, last: now };
            }

            // Frame counter
            state.totalFrames++;
            dom.frameCount.textContent  = state.totalFrames;
            dom.totalFrames.textContent = state.totalFrames;

            // ✅ Update buffer fill progress
            if (data.buffer_pct !== undefined) {
                state.bufferPct = data.buffer_pct;
                updateBufferUI(data.buffer_pct);
            }

            const { prediction, confidence } = data;

            if (prediction) {
                state.successFrames++;
                state.confSum += parseFloat(confidence);
                dom.avgConf.textContent = (state.confSum / state.totalFrames).toFixed(1);
                window.neuralPulse('34, 197, 94');
            }

            dom.successRate.textContent =
                Math.round((state.successFrames / state.totalFrames) * 100) + '%';

            updatePred(prediction, confidence);

        } catch (err) {
            console.error(err);
            log('error', 'Request failed: ' + err.message);
        }

        state.sending = false;
    }

    // ✅ 120ms interval — sends ~8 frames/sec to backend
    // 30 frames needed ÷ 8fps = ~3.5 seconds to first prediction
    // Much better than old 300ms skip = ~18 seconds
    state.loopId = setTimeout(sendLoop, 120);
}

/* =============================================
   BUFFER PROGRESS UI
   Shows user how full the 30-frame buffer is
   before first prediction appears
   ============================================= */
function updateBufferUI(pct) {
    if (pct >= 100) {
        dom.badge.className = dom.badge.className; // keep current
        return;
    }
    // While filling buffer, show progress in badge
    dom.badge.textContent = `BUFFERING ${pct}%`;
    dom.badge.className   = 'result-badge detecting';
    dom.overlayPred.textContent = `Loading... ${pct}%`;
}

/* =============================================
   PREDICTION UI
   ============================================= */
function updatePred(pred, conf) {
    if (!pred) {
        if (state.bufferPct < 100) return; // don't reset while buffering
        dom.predText.textContent    = '---';
        dom.overlayPred.textContent = 'Analyzing...';
        dom.confFill.style.width    = '0%';
        dom.confGlow.style.width    = '0%';
        dom.confBarFill.style.width = '0%';
        dom.confPct.textContent     = '0.0%';
        dom.badge.textContent       = 'DETECTING';
        dom.badge.className         = 'result-badge detecting';
        dom.predText.classList.remove('flash');
        state.currentPred = null;
        return;
    }

    if (pred !== state.currentPred) {
        dom.predText.classList.remove('flash');
        void dom.predText.offsetWidth;
        dom.predText.classList.add('flash');
        setTimeout(()=>dom.predText.classList.remove('flash'), 700);
        window.neuralRipple();
        state.currentPred = pred;
        log('success', `Detected: ${pred} (${parseFloat(conf).toFixed(1)}%)`);
    }

    const c = parseFloat(conf);
    dom.predText.textContent    = pred;
    dom.overlayPred.textContent = pred;
    dom.confFill.style.width    = c + '%';
    dom.confGlow.style.width    = c + '%';
    dom.confBarFill.style.width = c + '%';
    dom.confPct.textContent     = c.toFixed(1) + '%';

    if      (c >= 80) { dom.badge.textContent='HIGH CONF'; dom.badge.className='result-badge high'; }
    else if (c >= 50) { dom.badge.textContent='DETECTING'; dom.badge.className='result-badge detecting'; }
    else              { dom.badge.textContent='LOW CONF';  dom.badge.className='result-badge'; }
}

/* =============================================
   SENTENCE BUILDER
   ============================================= */
function addWordToSentence() {
    if (!state.currentPred) { showToast('No gesture detected!'); return; }
    state.sentence.push(state.currentPred);
    renderSentence();
    log('success', `Added: "${state.currentPred}"`);
    showToast(`"${state.currentPred}" added`);
}
function clearSentence() {
    state.sentence = [];
    renderSentence();
    log('info', 'Sentence cleared.');
    showToast('Cleared');
}
function renderSentence() {
    dom.wordCount.textContent = state.sentence.length + ' word' + (state.sentence.length!==1?'s':'');
    dom.sentDisplay.innerHTML = state.sentence.length
        ? ''
        : '<span class="sentence-placeholder">Your sentence will appear here...</span>';
    if (state.sentence.length) dom.sentDisplay.textContent = state.sentence.join(' ');
}
function copyToClipboard() {
    if (!state.sentence.length) { showToast('Nothing to copy!'); return; }
    navigator.clipboard.writeText(state.sentence.join(' '))
        .then(()=>showToast('Copied ⎘'))
        .catch(()=>showToast('Copy failed'));
}

/* =============================================
   HELPERS
   ============================================= */
function updateTime() {
    if (!state.sessionStart) return;
    const s = Math.floor((Date.now()-state.sessionStart)/1000);
    dom.sessionTime.textContent =
        String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
}
function setStatus(mode, text) {
    dom.statusPill.className   = 'status-pill'+(mode==='online'?' online':'');
    dom.statusText.textContent = text;
}
function log(type, msg) {
    const t = new Date();
    const time = String(t.getMinutes()).padStart(2,'0')+':'+String(t.getSeconds()).padStart(2,'0');
    const el = document.createElement('div');
    el.className = 'log-entry log-'+type;
    el.innerHTML = `<span class="log-time">${time}</span><span class="log-msg">${msg}</span>`;
    dom.logBody.appendChild(el);
    dom.logBody.scrollTop = dom.logBody.scrollHeight;
    while (dom.logBody.children.length>40) dom.logBody.removeChild(dom.logBody.firstChild);
}
let toastTimer;
function showToast(msg) {
    dom.toast.textContent = msg;
    dom.toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(()=>dom.toast.classList.remove('show'), 2500);
}

/* =============================================
   KEYBOARD
   ============================================= */
document.addEventListener('keydown', e => {
    if (e.target.matches('input, textarea')) return;
    if (e.code==='Space')     { e.preventDefault(); state.streaming ? stopCamera() : startCamera(); }
    if (e.code==='Enter')     { e.preventDefault(); addWordToSentence(); }
    if (e.code==='Backspace') {
        e.preventDefault();
        if (state.sentence.length) {
            const r = state.sentence.pop();
            renderSentence();
            showToast(`Removed: "${r}"`);
        }
    }
});
