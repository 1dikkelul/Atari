// public/js/game.js

const OPPONENTS = [
    {
        id: 'sb3_pong_v1',
        label: 'SB3 Pong v1',
        modelPath: './models/sb3_pong_actor.onnx',
    },
];

const LEADERBOARD_KEY = 'atari_pong_casual_leaderboard_v1';

class PongArena {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        this.canvas.width = 800;
        this.canvas.height = 500;

        this.ball = { x: 400, y: 250, vx: 5, vy: 3, radius: 8 };
        this.paddleWidth = 12;
        this.paddleHeight = 72;

        this.leftPaddle = { x: 20, y: 210, score: 0, speed: 6 };
        this.rightPaddle = { x: 768, y: 210, score: 0, speed: 6 };

        this.keys = {};
        this.setupKeyboardListeners();

        this.aiCanvas = document.createElement('canvas');
        this.aiCanvas.width = 84;
        this.aiCanvas.height = 84;
        this.aiCtx = this.aiCanvas.getContext('2d', { willReadFrequently: true });

        this.frameStack = [];
        this.maxFrames = 4;

        this.aiSession = null;
        this.loadedModelPath = null;
        this.isProcessingAI = false;

        this.frameCounter = 0;
        this.frameSkip = 4;
        this.currentAIAction = 0;
        this.maxBallSpeed = 14;

        this.targetScore = 5;
        this.matchActive = false;
        this.playerName = 'Player';
        this.selectedOpponent = OPPONENTS[0];
        this.currentMatchStartMs = null;
    }

    captureAIVision() {
        // 1. Fill with Atari Pong background gray (87/255 normalised internally by the ONNX model)
        this.aiCtx.fillStyle = 'rgb(87,87,87)';
        this.aiCtx.fillRect(0, 0, 84, 84);

        // 2. Map game coordinates → 84x84 AI canvas
        const scaleX = 84 / this.canvas.width;
        const scaleY = 84 / this.canvas.height;

        // 3. Draw paddles and ball in Atari white (236) so values match what the SB3 model was trained on
        this.aiCtx.fillStyle = 'rgb(236,236,236)';

        // Left Paddle
        const leftX = Math.round(this.leftPaddle.x * scaleX);
        const leftY = Math.round(this.leftPaddle.y * scaleY);
        const padW = Math.max(4, Math.round(this.paddleWidth * scaleX * 2.5));
        const padH = Math.max(6, Math.round(this.paddleHeight * scaleY));
        this.aiCtx.fillRect(leftX, leftY, padW, padH);

        // Right Paddle — pinned to x≈72 to match the actual Atari Pong coordinate the model trained on
        const rightY = Math.round(this.rightPaddle.y * scaleY);
        this.aiCtx.fillRect(72, rightY, padW, padH);

        // Ball
        const ballX = Math.round(this.ball.x * scaleX);
        const ballY = Math.round(this.ball.y * scaleY);
        const ballSize = Math.max(3, Math.round(this.ball.radius * scaleY * 2));
        this.aiCtx.fillRect(ballX - 1, ballY - 1, ballSize, ballSize);

        // 4. Extract pixel matrix — pass raw 0-255 float values; /255 is baked into the ONNX model
        const imgData = this.aiCtx.getImageData(0, 0, 84, 84);
        const pixels = imgData.data;
        const currentFrame = new Float32Array(84 * 84);

        let pixelIdx = 0;
        for (let i = 0; i < pixels.length; i += 4) {
            currentFrame[pixelIdx] = pixels[i];  // red channel = grayscale value (0-255)
            pixelIdx++;
        }

        // 6. Manage temporal frame stack history
        if (this.frameStack.length === 0) {
            for (let i = 0; i < this.maxFrames; i++) {
                this.frameStack.push(currentFrame);
            }
            if (window.uiLog) window.uiLog("Vision stack warmed up with thick visibility matrices.");
        } else {
            this.frameStack.shift();
            this.frameStack.push(currentFrame);
        }

        // 7. Mirror the 84x84 AI canvas to the debug view (scaled up 2x)
        const debugCanvas = document.getElementById('aiVisionDebug');
        if (debugCanvas) {
            const dCtx = debugCanvas.getContext('2d');
            dCtx.imageSmoothingEnabled = false;
            dCtx.drawImage(this.aiCanvas, 0, 0, 168, 168);
        }
    }

    setupKeyboardListeners() {
        window.addEventListener('keydown', (e) => this.keys[e.key] = true);
        window.addEventListener('keyup', (e) => this.keys[e.key] = false);
    }

    update() {
        if (!this.matchActive) {
            return;
        }

        // Human player now controls the LEFT paddle
        if (this.keys['ArrowUp'] || this.keys['w'] || this.keys['W']) {
            this.leftPaddle.y = Math.max(0, this.leftPaddle.y - this.leftPaddle.speed);
        }
        if (this.keys['ArrowDown'] || this.keys['s'] || this.keys['S']) {
            this.leftPaddle.y = Math.min(this.canvas.height - this.paddleHeight, this.leftPaddle.y + this.leftPaddle.speed);
        }

        // Atari-style semantics: repeat the last selected AI action every frame.
        this.applyRightPaddleAction(this.currentAIAction);

        // Ball movement physics
        this.ball.x += this.ball.vx;
        this.ball.y += this.ball.vy;

        if (this.ball.y - this.ball.radius <= 0 || this.ball.y + this.ball.radius >= this.canvas.height) {
            this.ball.vy *= -1;
        }

        if (this.ball.vx < 0 &&
            this.ball.x - this.ball.radius <= this.leftPaddle.x + this.paddleWidth &&
            this.ball.y >= this.leftPaddle.y &&
            this.ball.y <= this.leftPaddle.y + this.paddleHeight) {
            const relative = (this.ball.y - (this.leftPaddle.y + this.paddleHeight / 2)) / (this.paddleHeight / 2);
            this.ball.vx = Math.abs(this.ball.vx) * 1.03;
            this.ball.vy += relative * 1.8;
            this.ball.vx = Math.min(this.ball.vx, this.maxBallSpeed);
            this.ball.x = this.leftPaddle.x + this.paddleWidth + this.ball.radius;
        }

        if (this.ball.vx > 0 &&
            this.ball.x + this.ball.radius >= this.rightPaddle.x &&
            this.ball.y >= this.rightPaddle.y &&
            this.ball.y <= this.rightPaddle.y + this.paddleHeight) {
            const relative = (this.ball.y - (this.rightPaddle.y + this.paddleHeight / 2)) / (this.paddleHeight / 2);
            this.ball.vx = -Math.abs(this.ball.vx) * 1.03;
            this.ball.vy += relative * 1.8;
            this.ball.vx = Math.max(this.ball.vx, -this.maxBallSpeed);
            this.ball.x = this.rightPaddle.x - this.ball.radius;
        }

        if (this.ball.x < 0) {
            this.rightPaddle.score++;
            if (this.rightPaddle.score >= this.targetScore) {
                this.finishMatch(false);
            } else {
                this.resetBall();
            }
        } else if (this.ball.x > this.canvas.width) {
            this.leftPaddle.score++;
            if (this.leftPaddle.score >= this.targetScore) {
                this.finishMatch(true);
            } else {
                this.resetBall();
            }
        }
    }

    resetBall() {
        this.ball.x = this.canvas.width / 2;
        this.ball.y = this.canvas.height / 2;
        this.ball.vx = (Math.random() > 0.5 ? 4 : -4);
        this.ball.vy = (Math.random() * 6) - 3;
    }

    applyRightPaddleAction(action) {
        // Pong action mapping used in this project: [2,4] up, [3,5] down.
        if (action === 2 || action === 4) {
            this.rightPaddle.y = Math.max(0, this.rightPaddle.y - this.rightPaddle.speed);
        } else if (action === 3 || action === 5) {
            this.rightPaddle.y = Math.min(this.canvas.height - this.paddleHeight, this.rightPaddle.y + this.rightPaddle.speed);
        }
    }

    render() {
        this.ctx.fillStyle = '#0a0a12';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.strokeStyle = 'rgba(0, 255, 204, 0.15)';
        this.ctx.lineWidth = 4;
        this.ctx.setLineDash([10, 10]);
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width / 2, 0);
        this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        this.ctx.fillStyle = '#ff007f';
        this.ctx.fillRect(this.leftPaddle.x, this.leftPaddle.y, this.paddleWidth, this.paddleHeight);

        this.ctx.fillStyle = '#00ffcc';
        this.ctx.fillRect(this.rightPaddle.x, this.rightPaddle.y, this.paddleWidth, this.paddleHeight);

        this.ctx.fillStyle = '#39ff14';
        this.ctx.beginPath();
        this.ctx.arc(this.ball.x, this.ball.y, this.ball.radius, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.font = 'bold 32px monospace';
        this.ctx.fillStyle = 'rgba(255, 0, 127, 0.5)';
        this.ctx.fillText(this.leftPaddle.score, this.canvas.width / 4, 50);
        this.ctx.fillStyle = 'rgba(0, 255, 204, 0.5)';
        this.ctx.fillText(this.rightPaddle.score, (3 * this.canvas.width) / 4, 50);

        if (!this.matchActive) {
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
            this.ctx.font = 'bold 20px monospace';
            this.ctx.fillText('Press "Start Match" to play', 225, 250);
        }
    }

    async loadModel(modelPath) {
        const statusElement = document.getElementById('engineStatus');
        try {
            if (this.loadedModelPath === modelPath && this.aiSession) {
                return;
            }

            if (window.uiLog) window.uiLog(`HTTP request: loading model ${modelPath}`);
            const responseModel = await fetch(modelPath);
            if (!responseModel.ok) throw new Error(`HTTP ${responseModel.status} fetching blueprint`);
            const bufferModel = await responseModel.arrayBuffer();

            if (window.uiLog) window.uiLog(`Parsing neural layers... (${Math.round(bufferModel.byteLength / 1024)} KB payload)`);
            this.aiSession = await ort.InferenceSession.create(bufferModel);
            this.loadedModelPath = modelPath;

            if (window.uiLog) window.uiLog("✅ Core structural network mapping verified!");
            if (statusElement) {
                statusElement.innerText = `AI Engine: ACTIVE (${this.selectedOpponent.label})`;
                statusElement.style.border = "1px solid #00ff66";
                statusElement.style.color = "#00ff66";
                statusElement.style.boxShadow = "0 0 10px rgba(0, 255, 70, 0.2)";
            }
        } catch (err) {
            console.error("Failed to load ONNX model:", err);
            if (window.uiLog) window.uiLog(`❌ ENGINE SETUP FAILURE: ${err.message}`, true);
            if (statusElement) statusElement.innerText = "AI Engine: CRASHED";
        }
    }

    setPlayerName(name) {
        const trimmed = (name || '').trim();
        this.playerName = trimmed || 'Player';
    }

    async setOpponent(opponentId) {
        const next = OPPONENTS.find((o) => o.id === opponentId) || OPPONENTS[0];
        this.selectedOpponent = next;
        await this.loadModel(next.modelPath);
    }

    async startMatch() {
        await this.setOpponent(this.selectedOpponent.id);
        this.leftPaddle.score = 0;
        this.rightPaddle.score = 0;
        this.leftPaddle.y = (this.canvas.height - this.paddleHeight) / 2;
        this.rightPaddle.y = (this.canvas.height - this.paddleHeight) / 2;
        this.currentAIAction = 0;
        this.frameStack = [];
        this.matchActive = true;
        this.currentMatchStartMs = Date.now();
        this.resetBall();

        if (window.uiLog) {
            window.uiLog(`Match started: ${this.playerName} vs ${this.selectedOpponent.label}`);
        }
    }

    finishMatch(playerWon) {
        this.matchActive = false;
        const durationSec = Math.round((Date.now() - this.currentMatchStartMs) / 1000);
        const resultText = playerWon ? 'WIN' : 'LOSS';

        const record = {
            player: this.playerName,
            opponentId: this.selectedOpponent.id,
            opponentLabel: this.selectedOpponent.label,
            won: playerWon,
            playerScore: this.leftPaddle.score,
            opponentScore: this.rightPaddle.score,
            durationSec,
            timestamp: new Date().toISOString(),
        };
        this.saveLeaderboardRecord(record);

        const statusElement = document.getElementById('engineStatus');
        if (statusElement) {
            statusElement.innerText = `Match Complete: ${resultText}`;
        }

        if (window.uiLog) {
            window.uiLog(
                `${resultText}: ${this.playerName} ${this.leftPaddle.score}-${this.rightPaddle.score} ${this.selectedOpponent.label} (${durationSec}s)`
            );
        }
    }

    getLeaderboardData() {
        try {
            const raw = localStorage.getItem(LEADERBOARD_KEY);
            if (!raw) return [];
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
        } catch (err) {
            return [];
        }
    }

    saveLeaderboardData(data) {
        localStorage.setItem(LEADERBOARD_KEY, JSON.stringify(data));
    }

    saveLeaderboardRecord(record) {
        const existing = this.getLeaderboardData();
        existing.push(record);
        this.saveLeaderboardData(existing);
        this.renderLeaderboard();
    }

    resetLeaderboard() {
        this.saveLeaderboardData([]);
        this.renderLeaderboard();
    }

    renderLeaderboard() {
        const tbody = document.getElementById('leaderboardBody');
        if (!tbody) return;

        const records = this.getLeaderboardData();
        if (records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; opacity:0.7;">No matches recorded yet.</td></tr>';
            return;
        }

        const byKey = {};
        for (const r of records) {
            const key = `${r.player}__${r.opponentId}`;
            if (!byKey[key]) {
                byKey[key] = {
                    player: r.player,
                    opponentLabel: r.opponentLabel,
                    wins: 0,
                    losses: 0,
                    games: 0,
                    lastResult: '',
                };
            }

            byKey[key].games += 1;
            if (r.won) byKey[key].wins += 1;
            else byKey[key].losses += 1;
            byKey[key].lastResult = `${r.playerScore}-${r.opponentScore}`;
        }

        const rows = Object.values(byKey).sort((a, b) => {
            if (b.wins !== a.wins) return b.wins - a.wins;
            if (a.losses !== b.losses) return a.losses - b.losses;
            return b.games - a.games;
        });

        tbody.innerHTML = rows
            .map((row) => {
                const winRate = row.games > 0 ? Math.round((row.wins / row.games) * 100) : 0;
                return `
                    <tr>
                        <td>${row.player}</td>
                        <td>${row.opponentLabel}</td>
                        <td>${row.wins}</td>
                        <td>${row.losses}</td>
                        <td>${winRate}%</td>
                        <td>${row.lastResult}</td>
                    </tr>
                `;
            })
            .join('');
    }

    getOpponents() {
        return OPPONENTS.slice();
    }

    async processAI() {
        if (!this.aiSession || this.isProcessingAI) return;
        this.isProcessingAI = true;

        try {
            this.captureAIVision();
            // --- CHANNELS-FIRST (NCHW) TENSOR PIPELINE ---
            // Model was exported with shape [1, 4, 84, 84]: batch, frames, height, width
            const tensorSize = 1 * 4 * 84 * 84;
            const inputBuffer = new Float32Array(tensorSize);

            let offset = 0;
            // Channels-first: all pixels for frame 0, then frame 1, etc.
            for (let f = 0; f < this.maxFrames; f++) {
                for (let i = 0; i < 84 * 84; i++) {
                    inputBuffer[offset] = this.frameStack[f][i];
                    offset++;
                }
            }

            const inputTensor = new ort.Tensor('float32', inputBuffer, [1, 4, 84, 84]);
            // Input name 'input' matches the ONNX export input_names=['input']
            const feeds = { 'input': inputTensor };
            const results = await this.aiSession.run(feeds);
            const logits = results['output'].data;

            const currentFrameData = this.frameStack[this.frameStack.length - 1];
            let sum = 0;
            for (let i = 0; i < currentFrameData.length; i++) sum += currentFrameData[i];
            const mean = sum / currentFrameData.length;
            let varianceSum = 0;
            for (let i = 0; i < currentFrameData.length; i++) varianceSum += Math.pow(currentFrameData[i] - mean, 2);
            const frameVariance = varianceSum / currentFrameData.length;

            let action = 0;
            for (let i = 1; i < logits.length; i++) {
                if (logits[i] > logits[action]) action = i;
            }

            const metricsPanel = document.getElementById('metricsContent');
            if (metricsPanel && this.frameCounter % 4 === 0) {
                const nonZeroPixels = this.frameStack[this.frameStack.length - 1].filter(v => v > 0).length;
                metricsPanel.innerHTML = `
                    <p style="color:#66ff66; margin:0 0 5px 0;"><strong>DIAGNOSTIC TELEMETRY:</strong></p>
                    • Inference Frame:  <span style="color:#fff; font-weight:bold;">${this.frameCounter}</span><br>
                    • Vision Variance:  <span style="color:#fff; font-weight:bold;">${frameVariance.toFixed(6)}</span><br>
                    • Non-zero Pixels:  <span style="color:#fff;">${nonZeroPixels} / ${84*84}</span><br>
                    • Stack History:    <span style="color:#fff;">${this.frameStack.length} / ${this.maxFrames} frames</span><br>
                    • Argmax Choice:    <span style="color:#ff007f; font-weight:bold;">Action [${action}]</span><br>
                    
                    <p style="color:#ff007f; margin:10px 0 5px 0;"><strong>ACTION LOGITS:</strong></p>
                    • NO-OP:  <span style="color:#fff;">${logits[0].toFixed(3)}</span><br>
                    • FIRE:   <span style="color:#fff;">${logits[1].toFixed(3)}</span><br>
                    • UP:     <span style="color:#fff;">${logits[2].toFixed(3)}</span><br>
                    • DOWN:   <span style="color:#fff;">${logits[3].toFixed(3)}</span><br>
                    • RIGHT:  <span style="color:#fff;">${logits[4].toFixed(3)}</span><br>
                    • LEFT:   <span style="color:#fff;">${logits[5].toFixed(3)}</span>
                `;
            }

            // Keep the selected action active between inference ticks.
            this.currentAIAction = action;

        } catch (err) {
            console.error("AI Inference Error:", err);
            if (this.frameCounter % 60 === 0 && window.uiLog) {
                window.uiLog(`❌ Runtime Evaluation Crash: ${err.message}`, true);
            }
        } finally {
            this.isProcessingAI = false;
        }
    }

    loop() {
        this.frameCounter++;
        this.update();

        if (this.frameCounter % this.frameSkip === 0) {
            this.processAI();
        }

        this.render();
        requestAnimationFrame(() => this.loop());
    }
}