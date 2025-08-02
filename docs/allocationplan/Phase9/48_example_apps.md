# Micro-Phase 9.48: Example Apps

## Objective
Create example applications demonstrating key features and use cases of CortexKG, including pattern recognition, semantic search, real-time visualization, and cortical memory applications.

## Prerequisites
- Completed micro-phase 9.47 (Integration Guide)
- API documentation and integration guides available
- WASM bindings and JavaScript API fully functional

## Task Description
Develop comprehensive example applications showcasing different aspects of CortexKG functionality. Create applications ranging from simple demonstrations to complex real-world use cases, with complete source code, documentation, and deployment instructions.

## Specific Actions

1. **Create pattern recognition demo application**
```html
<!-- examples/pattern-recognition/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CortexKG Pattern Recognition Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .input-canvas {
            border: 2px solid #4CAF50;
            border-radius: 8px;
            background: white;
            cursor: crosshair;
            display: block;
            margin: 0 auto;
        }
        
        .cortical-canvas {
            border: 2px solid #2196F3;
            border-radius: 8px;
            background: #000;
            display: block;
            margin: 0 auto;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        
        button {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .pattern-list {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .pattern-item {
            background: rgba(255, 255, 255, 0.1);
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .pattern-item:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        
        .stat {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 18px;
        }
        
        .error {
            background: rgba(244, 67, 54, 0.3);
            color: #ffcdd2;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel">
            <h2>🎨 Pattern Input</h2>
            <p>Draw patterns on the canvas below. CortexKG will learn and recognize similar patterns.</p>
            
            <canvas id="inputCanvas" class="input-canvas" width="280" height="280"></canvas>
            
            <div class="controls">
                <button id="trainButton" disabled>Train Pattern</button>
                <button id="recognizeButton" disabled>Recognize</button>
                <button id="clearButton">Clear Canvas</button>
                <button id="generateButton" disabled>Generate Random</button>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="patternCount">0</div>
                    <div>Patterns Learned</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="recognitionAccuracy">0%</div>
                    <div>Accuracy</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="activeColumns">0</div>
                    <div>Active Columns</div>
                </div>
            </div>
            
            <div id="recognitionResult" class="pattern-list">
                <h4>Recognition Results</h4>
                <div id="resultsList"></div>
            </div>
        </div>
        
        <div class="panel">
            <h2>🧠 Cortical Visualization</h2>
            <p>Real-time visualization of cortical column activity during pattern processing.</p>
            
            <canvas id="corticalCanvas" class="cortical-canvas" width="400" height="400"></canvas>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="memoryUsage">0MB</div>
                    <div>Memory Usage</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="processTime">0ms</div>
                    <div>Process Time</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="sparsity">0%</div>
                    <div>Sparsity</div>
                </div>
            </div>
            
            <div class="pattern-list">
                <h4>Learned Patterns</h4>
                <div id="patternsList"></div>
            </div>
            
            <div id="status" class="loading">
                Initializing CortexKG...
            </div>
        </div>
    </div>

    <script src="../../dist/cortex-kg.js"></script>
    <script>
        class PatternRecognitionDemo {
            constructor() {
                this.isInitialized = false;
                this.patterns = [];
                this.currentPattern = null;
                this.isDrawing = false;
                this.recognitionHistory = [];
                
                this.inputCanvas = document.getElementById('inputCanvas');
                this.inputCtx = this.inputCanvas.getContext('2d');
                this.corticalCanvas = document.getElementById('corticalCanvas');
                
                this.setupCanvas();
                this.setupEventListeners();
                this.initialize();
            }
            
            async initialize() {
                try {
                    document.getElementById('status').textContent = 'Loading WASM module...';
                    
                    await wasmLoader.init({
                        memorySize: 32 * 1024 * 1024,
                        enableSIMD: true,
                        debug: false
                    });
                    
                    document.getElementById('status').textContent = 'Initializing cortical visualization...';
                    
                    cortexVisualizer.init(this.corticalCanvas, {
                        columnSize: 3,
                        showActivations: true,
                        animationSpeed: 1.5
                    });
                    
                    document.getElementById('status').textContent = 'Ready! Draw patterns to begin.';
                    
                    this.enableButtons();
                    this.isInitialized = true;
                    
                    // Start continuous visualization updates
                    this.startVisualizationLoop();
                    
                } catch (error) {
                    document.getElementById('status').innerHTML = 
                        `<div class="error">Failed to initialize: ${error.message}</div>`;
                }
            }
            
            setupCanvas() {
                this.inputCtx.fillStyle = 'white';
                this.inputCtx.fillRect(0, 0, this.inputCanvas.width, this.inputCanvas.height);
                this.inputCtx.strokeStyle = 'black';
                this.inputCtx.lineWidth = 8;
                this.inputCtx.lineCap = 'round';
                this.inputCtx.lineJoin = 'round';
            }
            
            setupEventListeners() {
                // Drawing on input canvas
                this.inputCanvas.addEventListener('mousedown', this.startDrawing.bind(this));
                this.inputCanvas.addEventListener('mousemove', this.draw.bind(this));
                this.inputCanvas.addEventListener('mouseup', this.stopDrawing.bind(this));
                
                // Touch events for mobile
                this.inputCanvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = this.inputCanvas.getBoundingClientRect();
                    const x = touch.clientX - rect.left;
                    const y = touch.clientY - rect.top;
                    this.startDrawing({ offsetX: x, offsetY: y });
                });
                
                this.inputCanvas.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = this.inputCanvas.getBoundingClientRect();
                    const x = touch.clientX - rect.left;
                    const y = touch.clientY - rect.top;
                    this.draw({ offsetX: x, offsetY: y });
                });
                
                this.inputCanvas.addEventListener('touchend', (e) => {
                    e.preventDefault();
                    this.stopDrawing();
                });
                
                // Button event listeners
                document.getElementById('trainButton').addEventListener('click', this.trainPattern.bind(this));
                document.getElementById('recognizeButton').addEventListener('click', this.recognizePattern.bind(this));
                document.getElementById('clearButton').addEventListener('click', this.clearCanvas.bind(this));
                document.getElementById('generateButton').addEventListener('click', this.generateRandomPattern.bind(this));
            }
            
            startDrawing(e) {
                this.isDrawing = true;
                this.inputCtx.beginPath();
                this.inputCtx.moveTo(e.offsetX, e.offsetY);
            }
            
            draw(e) {
                if (!this.isDrawing) return;
                
                this.inputCtx.lineTo(e.offsetX, e.offsetY);
                this.inputCtx.stroke();
            }
            
            stopDrawing() {
                this.isDrawing = false;
            }
            
            enableButtons() {
                document.getElementById('trainButton').disabled = false;
                document.getElementById('recognizeButton').disabled = false;
                document.getElementById('generateButton').disabled = false;
            }
            
            clearCanvas() {
                this.inputCtx.fillStyle = 'white';
                this.inputCtx.fillRect(0, 0, this.inputCanvas.width, this.inputCanvas.height);
            }
            
            async trainPattern() {
                if (!this.isInitialized) return;
                
                try {
                    const patternData = this.extractPatternData();
                    if (patternData.every(x => x === 0)) {
                        alert('Please draw a pattern first!');
                        return;
                    }
                    
                    const startTime = performance.now();
                    
                    // Allocate concept for this pattern
                    const patternId = `pattern_${Date.now()}`;
                    const concept = await cortexWrapper.allocateConcept(patternId, 2048, {
                        description: `Hand-drawn pattern ${this.patterns.length + 1}`,
                        type: 'visual_pattern',
                        timestamp: Date.now()
                    });
                    
                    // Perform spatial pooling
                    const spatialResult = await cortexWrapper.spatialPooling(patternData, {
                        sparsity: 0.02,
                        boostStrength: 2.0
                    });
                    
                    const processTime = performance.now() - startTime;
                    
                    // Store pattern
                    this.patterns.push({
                        id: concept.id,
                        name: `Pattern ${this.patterns.length + 1}`,
                        data: patternData,
                        spatialResult: spatialResult,
                        timestamp: Date.now()
                    });
                    
                    // Update visualization
                    this.updateCorticalVisualization(spatialResult);
                    this.updateStats(processTime, spatialResult);
                    this.updatePatternsList();
                    
                    document.getElementById('status').textContent = 
                        `Pattern trained successfully! ${spatialResult.activeColumns.length} columns activated.`;
                    
                } catch (error) {
                    document.getElementById('status').innerHTML = 
                        `<div class="error">Training failed: ${error.message}</div>`;
                }
            }
            
            async recognizePattern() {
                if (!this.isInitialized || this.patterns.length === 0) {
                    alert('Please train some patterns first!');
                    return;
                }
                
                try {
                    const patternData = this.extractPatternData();
                    if (patternData.every(x => x === 0)) {
                        alert('Please draw a pattern first!');
                        return;
                    }
                    
                    const startTime = performance.now();
                    
                    // Perform spatial pooling on input
                    const inputResult = await cortexWrapper.spatialPooling(patternData, {
                        sparsity: 0.02,
                        boostStrength: 2.0
                    });
                    
                    // Calculate similarity with all trained patterns
                    const similarities = this.patterns.map(pattern => {
                        const similarity = this.calculateSimilarity(
                            inputResult.activeColumns,
                            pattern.spatialResult.activeColumns
                        );
                        return {
                            pattern: pattern,
                            similarity: similarity
                        };
                    });
                    
                    // Sort by similarity
                    similarities.sort((a, b) => b.similarity - a.similarity);
                    
                    const processTime = performance.now() - startTime;
                    
                    // Update visualization and results
                    this.updateCorticalVisualization(inputResult);
                    this.updateStats(processTime, inputResult);
                    this.displayRecognitionResults(similarities);
                    
                    // Update recognition accuracy
                    this.recognitionHistory.push({
                        timestamp: Date.now(),
                        bestMatch: similarities[0],
                        confidence: similarities[0].similarity
                    });
                    
                    this.updateRecognitionAccuracy();
                    
                } catch (error) {
                    document.getElementById('status').innerHTML = 
                        `<div class="error">Recognition failed: ${error.message}</div>`;
                }
            }
            
            generateRandomPattern() {
                this.clearCanvas();
                
                // Generate random pattern
                const centerX = this.inputCanvas.width / 2;
                const centerY = this.inputCanvas.height / 2;
                const numStrokes = 3 + Math.floor(Math.random() * 5);
                
                this.inputCtx.strokeStyle = 'black';
                this.inputCtx.lineWidth = 8;
                
                for (let i = 0; i < numStrokes; i++) {
                    this.inputCtx.beginPath();
                    
                    const startX = centerX + (Math.random() - 0.5) * 200;
                    const startY = centerY + (Math.random() - 0.5) * 200;
                    const endX = startX + (Math.random() - 0.5) * 100;
                    const endY = startY + (Math.random() - 0.5) * 100;
                    
                    this.inputCtx.moveTo(startX, startY);
                    
                    // Create curved line
                    const cpX = (startX + endX) / 2 + (Math.random() - 0.5) * 50;
                    const cpY = (startY + endY) / 2 + (Math.random() - 0.5) * 50;
                    
                    this.inputCtx.quadraticCurveTo(cpX, cpY, endX, endY);
                    this.inputCtx.stroke();
                }
            }
            
            extractPatternData() {
                // Convert canvas to binary pattern
                const imageData = this.inputCtx.getImageData(0, 0, this.inputCanvas.width, this.inputCanvas.height);
                const data = imageData.data;
                const pattern = [];
                
                // Downsample to 28x28 grid for processing
                const gridSize = 28;
                const cellWidth = this.inputCanvas.width / gridSize;
                const cellHeight = this.inputCanvas.height / gridSize;
                
                for (let y = 0; y < gridSize; y++) {
                    for (let x = 0; x < gridSize; x++) {
                        let sum = 0;
                        let count = 0;
                        
                        // Sample pixels in this cell
                        for (let sy = y * cellHeight; sy < (y + 1) * cellHeight; sy += 2) {
                            for (let sx = x * cellWidth; sx < (x + 1) * cellWidth; sx += 2) {
                                const idx = (Math.floor(sy) * this.inputCanvas.width + Math.floor(sx)) * 4;
                                const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                                sum += brightness;
                                count++;
                            }
                        }
                        
                        const avgBrightness = sum / count;
                        pattern.push(avgBrightness < 128 ? 1 : 0); // Threshold to binary
                    }
                }
                
                return pattern;
            }
            
            calculateSimilarity(columns1, columns2) {
                const set1 = new Set(columns1);
                const set2 = new Set(columns2);
                
                const intersection = new Set([...set1].filter(x => set2.has(x)));
                const union = new Set([...set1, ...set2]);
                
                return union.size === 0 ? 0 : intersection.size / union.size;
            }
            
            updateCorticalVisualization(spatialResult) {
                // Generate cortical data for visualization
                const columns = [];
                const numColumns = 64 * 64; // 64x64 grid
                
                for (let i = 0; i < numColumns; i++) {
                    columns.push({
                        id: i,
                        x: (i % 64) * 6,
                        y: Math.floor(i / 64) * 6,
                        isActive: spatialResult.activeColumns.includes(i),
                        isPredictive: Math.random() > 0.98,
                        boost: 1.0 + (Math.random() * 0.3)
                    });
                }
                
                const corticalData = {
                    columns: columns,
                    activations: columns.map(c => c.isActive ? 1.0 : 0.0),
                    predictions: columns.map(c => c.isPredictive ? 1.0 : 0.0),
                    timestamp: Date.now()
                };
                
                cortexVisualizer.updateCorticalData(corticalData);
            }
            
            updateStats(processTime, spatialResult) {
                document.getElementById('patternCount').textContent = this.patterns.length;
                document.getElementById('activeColumns').textContent = spatialResult.activeColumns.length;
                document.getElementById('processTime').textContent = `${processTime.toFixed(1)}ms`;
                
                const memoryStats = wasmLoader.getMemoryUsage();
                document.getElementById('memoryUsage').textContent = 
                    `${(memoryStats.usedSize / (1024 * 1024)).toFixed(1)}MB`;
                
                const sparsity = (spatialResult.activeColumns.length / 4096) * 100;
                document.getElementById('sparsity').textContent = `${sparsity.toFixed(1)}%`;
            }
            
            updatePatternsList() {
                const patternsList = document.getElementById('patternsList');
                patternsList.innerHTML = '';
                
                this.patterns.forEach((pattern, index) => {
                    const item = document.createElement('div');
                    item.className = 'pattern-item';
                    item.innerHTML = `
                        <strong>${pattern.name}</strong><br>
                        <small>Columns: ${pattern.spatialResult.activeColumns.length}</small><br>
                        <small>${new Date(pattern.timestamp).toLocaleTimeString()}</small>
                    `;
                    
                    item.addEventListener('click', () => {
                        this.updateCorticalVisualization(pattern.spatialResult);
                    });
                    
                    patternsList.appendChild(item);
                });
            }
            
            displayRecognitionResults(similarities) {
                const resultsList = document.getElementById('resultsList');
                resultsList.innerHTML = '';
                
                similarities.slice(0, 5).forEach((result, index) => {
                    const item = document.createElement('div');
                    item.className = 'pattern-item';
                    item.style.background = index === 0 ? 'rgba(76, 175, 80, 0.3)' : 'rgba(255, 255, 255, 0.1)';
                    
                    item.innerHTML = `
                        <strong>${result.pattern.name}</strong><br>
                        <small>Similarity: ${(result.similarity * 100).toFixed(1)}%</small>
                    `;
                    
                    resultsList.appendChild(item);
                });
            }
            
            updateRecognitionAccuracy() {
                if (this.recognitionHistory.length === 0) return;
                
                const recentResults = this.recognitionHistory.slice(-10);
                const avgConfidence = recentResults.reduce((sum, r) => sum + r.confidence, 0) / recentResults.length;
                
                document.getElementById('recognitionAccuracy').textContent = 
                    `${(avgConfidence * 100).toFixed(0)}%`;
            }
            
            startVisualizationLoop() {
                const animate = () => {
                    if (this.isInitialized) {
                        cortexVisualizer.render();
                    }
                    requestAnimationFrame(animate);
                };
                animate();
            }
        }
        
        // Initialize the demo when page loads
        window.addEventListener('load', () => {
            new PatternRecognitionDemo();
        });
    </script>
</body>
</html>
```

2. **Create semantic search application**
```html
<!-- examples/semantic-search/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CortexKG Semantic Search Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .search-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .search-container {
            position: relative;
            margin-bottom: 20px;
        }
        
        .search-input {
            width: 100%;
            padding: 15px 50px 15px 20px;
            font-size: 18px;
            border: none;
            border-radius: 50px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            outline: none;
            transition: all 0.3s ease;
        }
        
        .search-input:focus {
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        
        .search-button {
            position: absolute;
            right: 5px;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .search-button:hover {
            transform: translateY(-50%) scale(1.1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
        }
        
        .knowledge-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .results-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .panel-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .knowledge-item {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
        }
        
        .knowledge-item:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
            border-left-color: #4ecdc4;
        }
        
        .knowledge-item.selected {
            background: rgba(76, 175, 80, 0.3);
            border-left-color: #4CAF50;
        }
        
        .knowledge-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .knowledge-preview {
            font-size: 0.9em;
            opacity: 0.8;
            line-height: 1.4;
        }
        
        .result-item {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
        }
        
        .result-item:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        
        .similarity-high {
            border-left-color: #4CAF50;
        }
        
        .similarity-medium {
            border-left-color: #FF9800;
        }
        
        .similarity-low {
            border-left-color: #f44336;
        }
        
        .result-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .result-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .similarity-score {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .result-content {
            line-height: 1.6;
            margin-bottom: 10px;
        }
        
        .result-metadata {
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .add-content-form {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .form-input {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-size: 14px;
        }
        
        .form-textarea {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
        }
        
        .form-button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .form-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .status {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            text-align: center;
        }
        
        .status.loading {
            background: rgba(33, 150, 243, 0.3);
        }
        
        .status.error {
            background: rgba(244, 67, 54, 0.3);
        }
        
        .status.success {
            background: rgba(76, 175, 80, 0.3);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #4ecdc4;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 CortexKG Semantic Search</h1>
            <p>Intelligent knowledge discovery powered by cortical computing</p>
        </div>
        
        <div class="search-section">
            <div class="search-container">
                <input 
                    type="text" 
                    id="searchInput" 
                    class="search-input" 
                    placeholder="Search for concepts, ideas, or ask questions..."
                    disabled
                >
                <button id="searchButton" class="search-button" disabled>🔍</button>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="conceptCount">0</div>
                    <div class="stat-label">Knowledge Items</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="searchCount">0</div>
                    <div class="stat-label">Searches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avgAccuracy">0%</div>
                    <div class="stat-label">Avg Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="memoryUsage">0MB</div>
                    <div class="stat-label">Memory</div>
                </div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="knowledge-panel">
                <h2 class="panel-title">📚 Knowledge Base</h2>
                
                <div class="add-content-form">
                    <div class="form-group">
                        <label class="form-label">Title</label>
                        <input type="text" id="contentTitle" class="form-input" placeholder="Knowledge item title">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Content</label>
                        <textarea id="contentText" class="form-textarea" placeholder="Detailed content or description"></textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Category</label>
                        <input type="text" id="contentCategory" class="form-input" placeholder="e.g., Technology, Science, Business">
                    </div>
                    <button id="addContentButton" class="form-button" disabled>Add Knowledge</button>
                </div>
                
                <div id="knowledgeList">
                    <!-- Knowledge items will be populated here -->
                </div>
            </div>
            
            <div class="results-panel">
                <h2 class="panel-title">🎯 Search Results</h2>
                
                <div id="searchResults">
                    <div class="status">
                        Enter a search query to find relevant knowledge items
                    </div>
                </div>
            </div>
        </div>
        
        <div id="systemStatus" class="status loading">
            Initializing CortexKG semantic search engine...
        </div>
    </div>

    <script src="../../dist/cortex-kg.js"></script>
    <script>
        class SemanticSearchDemo {
            constructor() {
                this.knowledgeBase = [];
                this.searchHistory = [];
                this.isInitialized = false;
                this.searchCount = 0;
                
                this.setupEventListeners();
                this.loadSampleData();
                this.initialize();
            }
            
            async initialize() {
                try {
                    this.updateStatus('Loading WASM module...', 'loading');
                    
                    await wasmLoader.init({
                        memorySize: 64 * 1024 * 1024, // 64MB for larger knowledge base
                        enableSIMD: true,
                        debug: false
                    });
                    
                    this.updateStatus('Initializing semantic processing...', 'loading');
                    
                    // Pre-process sample data
                    await this.processSampleData();
                    
                    this.updateStatus('Ready! Add knowledge or start searching.', 'success');
                    this.enableControls();
                    this.isInitialized = true;
                    
                    this.updateStats();
                    
                } catch (error) {
                    this.updateStatus(`Failed to initialize: ${error.message}`, 'error');
                }
            }
            
            setupEventListeners() {
                document.getElementById('searchInput').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.performSearch();
                    }
                });
                
                document.getElementById('searchButton').addEventListener('click', this.performSearch.bind(this));
                document.getElementById('addContentButton').addEventListener('click', this.addKnowledge.bind(this));
                
                // Auto-resize textarea
                document.getElementById('contentText').addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = this.scrollHeight + 'px';
                });
            }
            
            enableControls() {
                document.getElementById('searchInput').disabled = false;
                document.getElementById('searchButton').disabled = false;
                document.getElementById('addContentButton').disabled = false;
            }
            
            loadSampleData() {
                // Sample knowledge base for demonstration
                this.sampleData = [
                    {
                        title: "Machine Learning Fundamentals",
                        content: "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way humans learn, gradually improving accuracy. It involves training algorithms on datasets to make predictions or decisions without being explicitly programmed for every scenario.",
                        category: "Technology"
                    },
                    {
                        title: "Neural Networks and Deep Learning",
                        content: "Neural networks are computing systems inspired by biological neural networks. Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks can learn hierarchical representations of data.",
                        category: "Technology"
                    },
                    {
                        title: "Climate Change and Global Warming",
                        content: "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, scientific evidence shows that human activities, particularly greenhouse gas emissions, are the primary driver of climate change since the 1800s.",
                        category: "Science"
                    },
                    {
                        title: "Quantum Computing Principles",
                        content: "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously, enabling exponentially faster computations for certain problems.",
                        category: "Technology"
                    },
                    {
                        title: "Sustainable Energy Solutions",
                        content: "Sustainable energy includes renewable sources like solar, wind, hydroelectric, and geothermal power. These energy sources are naturally replenished and have minimal environmental impact compared to fossil fuels, playing a crucial role in combating climate change.",
                        category: "Science"
                    },
                    {
                        title: "Digital Marketing Strategies",
                        content: "Digital marketing encompasses all marketing efforts using electronic devices and the internet. It includes search engine optimization, social media marketing, content marketing, email campaigns, and pay-per-click advertising to reach and engage target audiences online.",
                        category: "Business"
                    },
                    {
                        title: "Blockchain Technology",
                        content: "Blockchain is a distributed ledger technology that maintains a growing list of records, called blocks, linked using cryptography. Each block contains a cryptographic hash of the previous block, timestamp, and transaction data, making it resistant to modification.",
                        category: "Technology"
                    },
                    {
                        title: "Cognitive Psychology",
                        content: "Cognitive psychology studies mental processes including attention, memory, perception, language use, problem solving, creativity, and thinking. It examines how people acquire, process, and store information, bridging neuroscience and artificial intelligence research.",
                        category: "Science"
                    }
                ];
            }
            
            async processSampleData() {
                for (const item of this.sampleData) {
                    await this.addKnowledgeItem(item.title, item.content, item.category, false);
                }
                this.updateKnowledgeList();
            }
            
            async addKnowledge() {
                const title = document.getElementById('contentTitle').value.trim();
                const content = document.getElementById('contentText').value.trim();
                const category = document.getElementById('contentCategory').value.trim() || 'General';
                
                if (!title || !content) {
                    alert('Please provide both title and content');
                    return;
                }
                
                try {
                    await this.addKnowledgeItem(title, content, category, true);
                    
                    // Clear form
                    document.getElementById('contentTitle').value = '';
                    document.getElementById('contentText').value = '';
                    document.getElementById('contentCategory').value = '';
                    
                    this.updateKnowledgeList();
                    this.updateStats();
                    
                } catch (error) {
                    this.updateStatus(`Failed to add knowledge: ${error.message}`, 'error');
                }
            }
            
            async addKnowledgeItem(title, content, category, showStatus) {
                if (showStatus) {
                    this.updateStatus('Processing knowledge item...', 'loading');
                }
                
                // Create semantic representation
                const semanticPattern = this.createSemanticPattern(content);
                
                // Allocate concept in CortexKG
                const concept = await cortexWrapper.allocateConcept(
                    `knowledge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    2048 + content.length,
                    {
                        title: title,
                        content: content,
                        category: category,
                        timestamp: Date.now(),
                        type: 'knowledge_item'
                    }
                );
                
                // Perform spatial pooling on semantic pattern
                const spatialResult = await cortexWrapper.spatialPooling(semanticPattern, {
                    sparsity: 0.02,
                    boostStrength: 2.0
                });
                
                // Store in knowledge base
                const knowledgeItem = {
                    id: concept.id,
                    title: title,
                    content: content,
                    category: category,
                    semanticPattern: semanticPattern,
                    spatialRepresentation: spatialResult,
                    timestamp: Date.now(),
                    accessCount: 0
                };
                
                this.knowledgeBase.push(knowledgeItem);
                
                if (showStatus) {
                    this.updateStatus('Knowledge item added successfully!', 'success');
                }
                
                return knowledgeItem;
            }
            
            createSemanticPattern(text) {
                // Simple semantic pattern generation based on text analysis
                const words = text.toLowerCase().match(/\b\w+\b/g) || [];
                const pattern = new Array(2048).fill(0);
                
                // Hash words to bit positions
                words.forEach(word => {
                    const hash = this.simpleHash(word);
                    for (let i = 0; i < 3; i++) { // Each word activates 3 bits
                        const pos = (hash + i * 17) % pattern.length;
                        pattern[pos] = 1;
                    }
                });
                
                // Add category-specific patterns
                const categoryHash = this.simpleHash(text.substring(0, Math.min(text.length, 100)));
                for (let i = 0; i < 10; i++) {
                    const pos = (categoryHash + i * 23) % pattern.length;
                    pattern[pos] = 1;
                }
                
                return pattern;
            }
            
            simpleHash(str) {
                let hash = 0;
                for (let i = 0; i < str.length; i++) {
                    const char = str.charCodeAt(i);
                    hash = ((hash << 5) - hash) + char;
                    hash = hash & hash; // Convert to 32-bit integer
                }
                return Math.abs(hash);
            }
            
            async performSearch() {
                if (!this.isInitialized || this.knowledgeBase.length === 0) {
                    this.updateStatus('Please wait for initialization or add some knowledge first', 'error');
                    return;
                }
                
                const query = document.getElementById('searchInput').value.trim();
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                try {
                    this.updateStatus('Searching knowledge base...', 'loading');
                    
                    const startTime = performance.now();
                    
                    // Create semantic pattern for query
                    const queryPattern = this.createSemanticPattern(query);
                    
                    // Perform spatial pooling on query
                    const queryResult = await cortexWrapper.spatialPooling(queryPattern, {
                        sparsity: 0.02,
                        boostStrength: 2.0
                    });
                    
                    // Calculate similarity with all knowledge items
                    const results = [];
                    
                    for (const item of this.knowledgeBase) {
                        const similarity = this.calculateSimilarity(
                            queryResult.activeColumns,
                            item.spatialRepresentation.activeColumns
                        );
                        
                        // Text-based similarity for comparison
                        const textSimilarity = this.calculateTextSimilarity(query, item.content);
                        
                        // Combined similarity score
                        const combinedSimilarity = (similarity * 0.7) + (textSimilarity * 0.3);
                        
                        if (combinedSimilarity > 0.1) { // Threshold for relevance
                            results.push({
                                item: item,
                                similarity: combinedSimilarity,
                                spatialSimilarity: similarity,
                                textSimilarity: textSimilarity
                            });
                        }
                    }
                    
                    // Sort by similarity
                    results.sort((a, b) => b.similarity - a.similarity);
                    
                    const searchTime = performance.now() - startTime;
                    
                    // Update search history
                    this.searchHistory.push({
                        query: query,
                        timestamp: Date.now(),
                        resultsCount: results.length,
                        searchTime: searchTime,
                        topSimilarity: results.length > 0 ? results[0].similarity : 0
                    });
                    
                    this.searchCount++;
                    
                    // Display results
                    this.displaySearchResults(results, query, searchTime);
                    this.updateStats();
                    
                    // Increment access count for returned results
                    results.forEach(result => {
                        result.item.accessCount++;
                    });
                    
                } catch (error) {
                    this.updateStatus(`Search failed: ${error.message}`, 'error');
                }
            }
            
            calculateSimilarity(columns1, columns2) {
                const set1 = new Set(columns1);
                const set2 = new Set(columns2);
                
                const intersection = new Set([...set1].filter(x => set2.has(x)));
                const union = new Set([...set1, ...set2]);
                
                return union.size === 0 ? 0 : intersection.size / union.size;
            }
            
            calculateTextSimilarity(query, text) {
                const queryWords = new Set(query.toLowerCase().match(/\b\w+\b/g) || []);
                const textWords = new Set(text.toLowerCase().match(/\b\w+\b/g) || []);
                
                const intersection = new Set([...queryWords].filter(x => textWords.has(x)));
                const union = new Set([...queryWords, ...textWords]);
                
                return union.size === 0 ? 0 : intersection.size / union.size;
            }
            
            displaySearchResults(results, query, searchTime) {
                const resultsContainer = document.getElementById('searchResults');
                
                if (results.length === 0) {
                    resultsContainer.innerHTML = `
                        <div class="status">
                            No relevant results found for "${query}". Try different keywords or add more knowledge items.
                        </div>
                    `;
                    return;
                }
                
                let html = `
                    <div class="status success">
                        Found ${results.length} relevant results for "${query}" (${searchTime.toFixed(1)}ms)
                    </div>
                `;
                
                results.forEach((result, index) => {
                    const similarity = result.similarity;
                    const similarityClass = similarity > 0.6 ? 'similarity-high' : 
                                          similarity > 0.3 ? 'similarity-medium' : 'similarity-low';
                    
                    const highlightedContent = this.highlightSearchTerms(result.item.content, query);
                    
                    html += `
                        <div class="result-item ${similarityClass}">
                            <div class="result-header">
                                <div class="result-title">${result.item.title}</div>
                                <div class="similarity-score">${(similarity * 100).toFixed(1)}%</div>
                            </div>
                            <div class="result-content">${highlightedContent}</div>
                            <div class="result-metadata">
                                <span>📁 ${result.item.category}</span>
                                <span>🕒 ${new Date(result.item.timestamp).toLocaleDateString()}</span>
                                <span>👁️ ${result.item.accessCount} views</span>
                                <span>🧠 Spatial: ${(result.spatialSimilarity * 100).toFixed(1)}%</span>
                                <span>📝 Text: ${(result.textSimilarity * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `;
                });
                
                resultsContainer.innerHTML = html;
            }
            
            highlightSearchTerms(text, query) {
                const words = query.toLowerCase().match(/\b\w+\b/g) || [];
                let highlightedText = text;
                
                words.forEach(word => {
                    const regex = new RegExp(`\\b${word}\\b`, 'gi');
                    highlightedText = highlightedText.replace(regex, `<mark style="background: rgba(255, 235, 59, 0.6); padding: 2px 4px; border-radius: 3px;">$&</mark>`);
                });
                
                return highlightedText;
            }
            
            updateKnowledgeList() {
                const knowledgeList = document.getElementById('knowledgeList');
                
                if (this.knowledgeBase.length === 0) {
                    knowledgeList.innerHTML = '<div class="status">No knowledge items yet. Add some content above!</div>';
                    return;
                }
                
                // Sort by access count and recency
                const sortedItems = [...this.knowledgeBase].sort((a, b) => {
                    return (b.accessCount * 1000 + b.timestamp) - (a.accessCount * 1000 + a.timestamp);
                });
                
                let html = '';
                sortedItems.forEach(item => {
                    const preview = item.content.length > 100 ? 
                        item.content.substring(0, 100) + '...' : item.content;
                    
                    html += `
                        <div class="knowledge-item" data-id="${item.id}">
                            <div class="knowledge-title">${item.title}</div>
                            <div class="knowledge-preview">${preview}</div>
                            <div style="font-size: 0.8em; margin-top: 8px; opacity: 0.7;">
                                📁 ${item.category} | 👁️ ${item.accessCount} views | 
                                ${new Date(item.timestamp).toLocaleDateString()}
                            </div>
                        </div>
                    `;
                });
                
                knowledgeList.innerHTML = html;
                
                // Add click handlers
                knowledgeList.querySelectorAll('.knowledge-item').forEach(item => {
                    item.addEventListener('click', () => {
                        // Highlight selected item
                        knowledgeList.querySelectorAll('.knowledge-item').forEach(i => 
                            i.classList.remove('selected'));
                        item.classList.add('selected');
                        
                        // Find and highlight this item in search results if visible
                        const itemId = item.dataset.id;
                        const knowledgeItem = this.knowledgeBase.find(k => k.id === itemId);
                        if (knowledgeItem) {
                            // Auto-search for this item's title
                            document.getElementById('searchInput').value = knowledgeItem.title;
                        }
                    });
                });
            }
            
            updateStats() {
                document.getElementById('conceptCount').textContent = this.knowledgeBase.length;
                document.getElementById('searchCount').textContent = this.searchCount;
                
                // Calculate average accuracy from search history
                if (this.searchHistory.length > 0) {
                    const avgAccuracy = this.searchHistory.reduce((sum, search) => 
                        sum + search.topSimilarity, 0) / this.searchHistory.length;
                    document.getElementById('avgAccuracy').textContent = `${(avgAccuracy * 100).toFixed(0)}%`;
                } else {
                    document.getElementById('avgAccuracy').textContent = '0%';
                }
                
                // Update memory usage
                if (this.isInitialized) {
                    const memoryStats = wasmLoader.getMemoryUsage();
                    document.getElementById('memoryUsage').textContent = 
                        `${(memoryStats.usedSize / (1024 * 1024)).toFixed(1)}MB`;
                }
            }
            
            updateStatus(message, type = '') {
                const statusElement = document.getElementById('systemStatus');
                statusElement.textContent = message;
                statusElement.className = `status ${type}`;
                
                // Auto-hide success messages
                if (type === 'success') {
                    setTimeout(() => {
                        statusElement.style.display = 'none';
                    }, 3000);
                } else {
                    statusElement.style.display = 'block';
                }
            }
        }
        
        // Initialize the demo when page loads
        window.addEventListener('load', () => {
            new SemanticSearchDemo();
        });
    </script>
</body>
</html>
```

3. **Create real-time memory visualization app**
```javascript
// examples/memory-visualization/app.js
class MemoryVisualizationApp {
    constructor() {
        this.memoryStates = [];
        this.animationId = null;
        this.isRecording = false;
        this.temporalMemory = null;
        this.currentStep = 0;
        this.playbackSpeed = 1.0;
        
        this.setupUI();
        this.initialize();
    }
    
    async initialize() {
        try {
            this.updateStatus('Initializing CortexKG temporal memory system...');
            
            await wasmLoader.init({
                memorySize: 64 * 1024 * 1024,
                enableSIMD: true,
                debug: false
            });
            
            // Initialize visualization canvases
            this.initializeVisualization();
            
            // Create sample temporal sequences
            await this.createSampleSequences();
            
            this.updateStatus('Ready! Select a sequence to begin visualization.');
            this.enableControls();
            
        } catch (error) {
            this.updateStatus(`Initialization failed: ${error.message}`, 'error');
        }
    }
    
    setupUI() {
        const container = document.getElementById('app-container');
        container.innerHTML = `
            <div class="header">
                <h1>🧠 CortexKG Temporal Memory Visualization</h1>
                <p>Real-time visualization of cortical temporal memory processing</p>
            </div>
            
            <div class="main-grid">
                <div class="control-panel">
                    <h3>🎮 Control Panel</h3>
                    
                    <div class="sequence-selector">
                        <label>Memory Sequence:</label>
                        <select id="sequenceSelect" disabled>
                            <option value="">Select a sequence...</option>
                        </select>
                    </div>
                    
                    <div class="playback-controls">
                        <button id="playButton" disabled>▶️ Play</button>
                        <button id="pauseButton" disabled>⏸️ Pause</button>
                        <button id="resetButton" disabled>⏹️ Reset</button>
                        <button id="stepButton" disabled>⏭️ Step</button>
                    </div>
                    
                    <div class="speed-control">
                        <label>Playback Speed:</label>
                        <input type="range" id="speedSlider" min="0.1" max="3.0" step="0.1" value="1.0" disabled>
                        <span id="speedDisplay">1.0x</span>
                    </div>
                    
                    <div class="recording-controls">
                        <button id="recordButton" disabled>🔴 Record New</button>
                        <button id="stopRecordButton" disabled style="display: none;">⏹️ Stop Recording</button>
                        <input type="text" id="recordName" placeholder="Sequence name..." disabled>
                    </div>
                    
                    <div class="stats-panel">
                        <h4>📊 Memory Statistics</h4>
                        <div class="stat-grid">
                            <div class="stat-item">
                                <span class="stat-value" id="activeCells">0</span>
                                <span class="stat-label">Active Cells</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="predictiveCells">0</span>
                                <span class="stat-label">Predictive Cells</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="sequenceLength">0</span>
                                <span class="stat-label">Sequence Length</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="predictionAccuracy">0%</span>
                                <span class="stat-label">Prediction Accuracy</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="visualization-area">
                    <div class="canvas-container">
                        <h3>🎯 Current State</h3>
                        <canvas id="currentStateCanvas" width="400" height="400"></canvas>
                        <div class="canvas-legend">
                            <span class="legend-item"><span class="legend-color active"></span> Active Cells</span>
                            <span class="legend-item"><span class="legend-color predictive"></span> Predictive Cells</span>
                            <span class="legend-item"><span class="legend-color both"></span> Both</span>
                        </div>
                    </div>
                    
                    <div class="canvas-container">
                        <h3>📈 Temporal Timeline</h3>
                        <canvas id="timelineCanvas" width="400" height="200"></canvas>
                        <div class="timeline-controls">
                            <input type="range" id="timelineSlider" min="0" max="100" value="0" disabled>
                            <span id="timelinePosition">Step 0 / 0</span>
                        </div>
                    </div>
                </div>
                
                <div class="sequence-panel">
                    <h3>📝 Sequence Data</h3>
                    
                    <div class="input-pattern-display">
                        <h4>Current Input Pattern</h4>
                        <div id="inputPatternGrid"></div>
                    </div>
                    
                    <div class="prediction-display">
                        <h4>Predictions vs Reality</h4>
                        <div id="predictionGrid"></div>
                    </div>
                    
                    <div class="sequence-info">
                        <h4>Sequence Information</h4>
                        <div id="sequenceDetails"></div>
                    </div>
                </div>
            </div>
            
            <div id="status" class="status">Initializing...</div>
        `;
        
        this.addStyles();
        this.setupEventListeners();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .main-grid {
                display: grid;
                grid-template-columns: 300px 1fr 300px;
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .control-panel, .visualization-area, .sequence-panel {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(15px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .canvas-container {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                text-align: center;
            }
            
            canvas {
                background: #000;
                border-radius: 8px;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
            
            .canvas-legend {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-top: 10px;
                font-size: 0.9em;
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .legend-color {
                width: 12px;
                height: 12px;
                border-radius: 50%;
            }
            
            .legend-color.active { background: #4CAF50; }
            .legend-color.predictive { background: #FF9800; }
            .legend-color.both { background: #E91E63; }
            
            .sequence-selector, .playback-controls, .speed-control, .recording-controls {
                margin-bottom: 20px;
            }
            
            select, input, button {
                width: 100%;
                padding: 10px;
                margin: 5px 0;
                border: none;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.9);
                color: #333;
                font-size: 14px;
            }
            
            button {
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            
            button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }
            
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            
            .stat-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 10px;
            }
            
            .stat-item {
                background: rgba(0, 0, 0, 0.3);
                padding: 10px;
                border-radius: 8px;
                text-align: center;
            }
            
            .stat-value {
                display: block;
                font-size: 1.5em;
                font-weight: bold;
                color: #4ecdc4;
            }
            
            .stat-label {
                font-size: 0.8em;
                opacity: 0.8;
            }
            
            .timeline-controls {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: 10px;
            }
            
            .timeline-controls input {
                flex: 1;
                margin: 0;
            }
            
            .input-pattern-display, .prediction-display, .sequence-info {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
            }
            
            #inputPatternGrid, #predictionGrid {
                display: grid;
                grid-template-columns: repeat(8, 1fr);
                gap: 2px;
                margin-top: 10px;
            }
            
            .pattern-cell {
                width: 20px;
                height: 20px;
                border-radius: 3px;
                background: rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
            }
            
            .pattern-cell.active {
                background: #4CAF50;
                box-shadow: 0 0 10px rgba(76, 175, 80, 0.8);
            }
            
            .pattern-cell.predicted {
                background: #FF9800;
                box-shadow: 0 0 10px rgba(255, 152, 0, 0.8);
            }
            
            .pattern-cell.correct {
                background: #4CAF50;
                border: 2px solid #FFF;
            }
            
            .pattern-cell.incorrect {
                background: #f44336;
                border: 2px solid #FFF;
            }
            
            .status {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(15px);
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .status.error {
                background: rgba(244, 67, 54, 0.3);
                border-color: rgba(244, 67, 54, 0.5);
            }
            
            @media (max-width: 1200px) {
                .main-grid {
                    grid-template-columns: 1fr;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    setupEventListeners() {
        document.getElementById('sequenceSelect').addEventListener('change', this.loadSequence.bind(this));
        document.getElementById('playButton').addEventListener('click', this.play.bind(this));
        document.getElementById('pauseButton').addEventListener('click', this.pause.bind(this));
        document.getElementById('resetButton').addEventListener('click', this.reset.bind(this));
        document.getElementById('stepButton').addEventListener('click', this.step.bind(this));
        document.getElementById('recordButton').addEventListener('click', this.startRecording.bind(this));
        document.getElementById('stopRecordButton').addEventListener('click', this.stopRecording.bind(this));
        
        document.getElementById('speedSlider').addEventListener('input', (e) => {
            this.playbackSpeed = parseFloat(e.target.value);
            document.getElementById('speedDisplay').textContent = `${this.playbackSpeed}x`;
        });
        
        document.getElementById('timelineSlider').addEventListener('input', (e) => {
            if (this.currentSequence) {
                this.currentStep = parseInt(e.target.value);
                this.updateVisualization();
            }
        });
    }
    
    async createSampleSequences() {
        this.sequences = [
            {
                name: "ABC Pattern",
                description: "Simple repeating sequence: A-B-C-A-B-C",
                patterns: [
                    [1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]
                ]
            },
            {
                name: "Fibonacci Binary",
                description: "Fibonacci sequence in binary representation",
                patterns: [
                    [1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0], // 1
                    [1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0], // 1
                    [0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0], // 2
                    [1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0], // 3
                    [0,0,1,0,1,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0], // 5
                    [0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0], // 8
                ]
            },
            {
                name: "Spatial Patterns",
                description: "Evolving spatial patterns",
                patterns: [
                    [1,1,0,0,0,0,0,0, 1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,1,1,0,0,0,0,0, 0,1,1,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,0,1,1,0,0,0,0, 0,0,1,1,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,1,0,0, 0,0,0,0,1,1,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,1,0, 0,0,0,0,0,1,1,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]
                ]
            }
        ];
        
        // Populate sequence selector
        const select = document.getElementById('sequenceSelect');
        this.sequences.forEach((seq, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = seq.name;
            select.appendChild(option);
        });
    }
    
    async loadSequence() {
        const select = document.getElementById('sequenceSelect');
        const sequenceIndex = parseInt(select.value);
        
        if (isNaN(sequenceIndex)) return;
        
        this.currentSequence = this.sequences[sequenceIndex];
        this.currentStep = 0;
        
        try {
            this.updateStatus('Processing temporal sequence...');
            
            // Process sequence through temporal memory
            const result = await cortexWrapper.temporalMemory(
                this.currentSequence.patterns, 
                true // Enable predictions
            );
            
            this.temporalResult = result;
            
            // Setup timeline
            const slider = document.getElementById('timelineSlider');
            slider.max = this.currentSequence.patterns.length - 1;
            slider.value = 0;
            
            this.updateVisualization();
            this.updateSequenceInfo();
            this.updateStatus('Sequence loaded successfully!');
            
        } catch (error) {
            this.updateStatus(`Failed to load sequence: ${error.message}`, 'error');
        }
    }
    
    initializeVisualization() {
        this.currentStateCanvas = document.getElementById('currentStateCanvas');
        this.currentStateCtx = this.currentStateCanvas.getContext('2d');
        
        this.timelineCanvas = document.getElementById('timelineCanvas');
        this.timelineCtx = this.timelineCanvas.getContext('2d');
        
        // Initialize with empty state
        this.clearCanvas(this.currentStateCtx, this.currentStateCanvas);
        this.clearCanvas(this.timelineCtx, this.timelineCanvas);
    }
    
    clearCanvas(ctx, canvas) {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    updateVisualization() {
        if (!this.currentSequence || !this.temporalResult) return;
        
        this.renderCurrentState();
        this.renderTimeline();
        this.updateInputPattern();
        this.updatePredictionDisplay();
        this.updateStats();
        
        // Update timeline position
        document.getElementById('timelinePosition').textContent = 
            `Step ${this.currentStep + 1} / ${this.currentSequence.patterns.length}`;
    }
    
    renderCurrentState() {
        const canvas = this.currentStateCanvas;
        const ctx = this.currentStateCtx;
        
        this.clearCanvas(ctx, canvas);
        
        if (!this.temporalResult) return;
        
        const gridSize = 20; // 20x20 grid
        const cellSize = canvas.width / gridSize;
        
        // Get current state
        const activeCells = this.temporalResult.activeCells || [];
        const predictiveCells = this.temporalResult.predictiveCells || [];
        
        // Draw cells
        for (let i = 0; i < gridSize * gridSize; i++) {
            const x = (i % gridSize) * cellSize;
            const y = Math.floor(i / gridSize) * cellSize;
            
            const isActive = activeCells.includes(i);
            const isPredictive = predictiveCells.includes(i);
            
            if (isActive && isPredictive) {
                ctx.fillStyle = '#E91E63'; // Both
            } else if (isActive) {
                ctx.fillStyle = '#4CAF50'; // Active
            } else if (isPredictive) {
                ctx.fillStyle = '#FF9800'; // Predictive
            } else {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.1)'; // Inactive
            }
            
            ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            
            // Add subtle glow effect for active cells
            if (isActive || isPredictive) {
                ctx.shadowColor = ctx.fillStyle;
                ctx.shadowBlur = 5;
                ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                ctx.shadowBlur = 0;
            }
        }
    }
    
    renderTimeline() {
        const canvas = this.timelineCanvas;
        const ctx = this.timelineCtx;
        
        this.clearCanvas(ctx, canvas);
        
        if (!this.currentSequence) return;
        
        const patterns = this.currentSequence.patterns;
        const stepWidth = canvas.width / patterns.length;
        const maxHeight = canvas.height - 40;
        
        patterns.forEach((pattern, index) => {
            const x = index * stepWidth;
            const activity = pattern.reduce((sum, bit) => sum + bit, 0) / pattern.length;
            const height = activity * maxHeight;
            
            // Current step highlight
            if (index === this.currentStep) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.fillRect(x, 0, stepWidth, canvas.height);
            }
            
            // Activity bar
            ctx.fillStyle = index <= this.currentStep ? '#4CAF50' : 'rgba(255, 255, 255, 0.3)';
            ctx.fillRect(x + 2, canvas.height - height - 20, stepWidth - 4, height);
            
            // Step number
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText((index + 1).toString(), x + stepWidth/2, canvas.height - 5);
        });
        
        // Current position indicator
        const currentX = this.currentStep * stepWidth + stepWidth/2;
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(currentX, 0);
        ctx.lineTo(currentX, canvas.height - 25);
        ctx.stroke();
    }
    
    updateInputPattern() {
        const container = document.getElementById('inputPatternGrid');
        
        if (!this.currentSequence) {
            container.innerHTML = '';
            return;
        }
        
        const pattern = this.currentSequence.patterns[this.currentStep];
        container.innerHTML = '';
        
        pattern.slice(0, 64).forEach((bit, index) => { // Show first 64 bits in 8x8 grid
            const cell = document.createElement('div');
            cell.className = `pattern-cell ${bit ? 'active' : ''}`;
            container.appendChild(cell);
        });
    }
    
    updatePredictionDisplay() {
        const container = document.getElementById('predictionGrid');
        
        if (!this.temporalResult || !this.currentSequence) {
            container.innerHTML = '';
            return;
        }
        
        const predictions = this.temporalResult.prediction || [];
        const actualPattern = this.currentSequence.patterns[this.currentStep];
        
        container.innerHTML = '';
        
        predictions.slice(0, 64).forEach((pred, index) => {
            const cell = document.createElement('div');
            const predicted = pred > 0.5;
            const actual = actualPattern[index] === 1;
            
            let className = 'pattern-cell';
            if (predicted && actual) className += ' correct';
            else if (predicted && !actual) className += ' incorrect';
            else if (!predicted && actual) className += ' incorrect';
            
            if (predicted) className += ' predicted';
            
            cell.className = className;
            container.appendChild(cell);
        });
    }
    
    updateSequenceInfo() {
        const container = document.getElementById('sequenceDetails');
        
        if (!this.currentSequence) {
            container.innerHTML = '';
            return;
        }
        
        container.innerHTML = `
            <div><strong>Name:</strong> ${this.currentSequence.name}</div>
            <div><strong>Description:</strong> ${this.currentSequence.description}</div>
            <div><strong>Length:</strong> ${this.currentSequence.patterns.length} steps</div>
            <div><strong>Pattern Size:</strong> ${this.currentSequence.patterns[0].length} bits</div>
        `;
    }
    
    updateStats() {
        if (!this.temporalResult) return;
        
        document.getElementById('activeCells').textContent = 
            this.temporalResult.activeCells?.length || 0;
        document.getElementById('predictiveCells').textContent = 
            this.temporalResult.predictiveCells?.length || 0;
        document.getElementById('sequenceLength').textContent = 
            this.currentSequence?.patterns.length || 0;
        
        // Calculate prediction accuracy
        if (this.temporalResult.prediction && this.currentSequence) {
            const predictions = this.temporalResult.prediction;
            const actual = this.currentSequence.patterns[this.currentStep];
            let correct = 0;
            
            for (let i = 0; i < Math.min(predictions.length, actual.length); i++) {
                const predicted = predictions[i] > 0.5;
                const actualBit = actual[i] === 1;
                if (predicted === actualBit) correct++;
            }
            
            const accuracy = (correct / actual.length) * 100;
            document.getElementById('predictionAccuracy').textContent = `${accuracy.toFixed(1)}%`;
        }
    }
    
    enableControls() {
        const controls = [
            'sequenceSelect', 'playButton', 'pauseButton', 'resetButton', 
            'stepButton', 'speedSlider', 'timelineSlider', 'recordButton', 'recordName'
        ];
        
        controls.forEach(id => {
            document.getElementById(id).disabled = false;
        });
    }
    
    play() {
        if (!this.currentSequence) return;
        
        this.isPlaying = true;
        document.getElementById('playButton').style.display = 'none';
        document.getElementById('pauseButton').style.display = 'inline-block';
        
        this.playAnimation();
    }
    
    pause() {
        this.isPlaying = false;
        document.getElementById('playButton').style.display = 'inline-block';
        document.getElementById('pauseButton').style.display = 'none';
        
        if (this.animationId) {
            clearTimeout(this.animationId);
        }
    }
    
    reset() {
        this.pause();
        this.currentStep = 0;
        document.getElementById('timelineSlider').value = 0;
        this.updateVisualization();
    }
    
    step() {
        if (!this.currentSequence) return;
        
        this.currentStep = (this.currentStep + 1) % this.currentSequence.patterns.length;
        document.getElementById('timelineSlider').value = this.currentStep;
        this.updateVisualization();
    }
    
    playAnimation() {
        if (!this.isPlaying) return;
        
        this.step();
        
        const delay = 1000 / this.playbackSpeed; // Convert speed to delay
        this.animationId = setTimeout(() => {
            this.playAnimation();
        }, delay);
    }
    
    startRecording() {
        // Implementation for recording new sequences
        this.isRecording = true;
        document.getElementById('recordButton').style.display = 'none';
        document.getElementById('stopRecordButton').style.display = 'inline-block';
        
        this.recordedPatterns = [];
        this.updateStatus('Recording new sequence... Draw patterns or input data.');
    }
    
    stopRecording() {
        this.isRecording = false;
        document.getElementById('recordButton').style.display = 'inline-block';
        document.getElementById('stopRecordButton').style.display = 'none';
        
        const name = document.getElementById('recordName').value.trim() || 'Custom Sequence';
        
        if (this.recordedPatterns.length > 0) {
            const newSequence = {
                name: name,
                description: `User-recorded sequence with ${this.recordedPatterns.length} patterns`,
                patterns: this.recordedPatterns
            };
            
            this.sequences.push(newSequence);
            
            // Add to selector
            const select = document.getElementById('sequenceSelect');
            const option = document.createElement('option');
            option.value = this.sequences.length - 1;
            option.textContent = name;
            select.appendChild(option);
            
            this.updateStatus(`Sequence "${name}" recorded successfully!`);
        } else {
            this.updateStatus('No patterns recorded.', 'error');
        }
        
        document.getElementById('recordName').value = '';
    }
    
    updateStatus(message, type = '') {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MemoryVisualizationApp();
});
```

4. **Create complete example applications index**
```html
<!-- examples/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CortexKG Example Applications</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 50px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 15px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
            line-height: 1.6;
        }
        
        .examples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-bottom: 50px;
        }
        
        .example-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .example-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            background: rgba(255, 255, 255, 0.15);
        }
        
        .example-icon {
            font-size: 3em;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .example-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .example-description {
            line-height: 1.6;
            margin-bottom: 20px;
            opacity: 0.9;
        }
        
        .example-features {
            list-style: none;
            margin-bottom: 25px;
        }
        
        .example-features li {
            padding: 5px 0;
            opacity: 0.8;
        }
        
        .example-features li:before {
            content: "✓ ";
            color: #4CAF50;
            font-weight: bold;
        }
        
        .example-button {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1em;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .example-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }
        
        .tech-specs {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 30px;
        }
        
        .specs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .spec-item {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        .spec-value {
            font-size: 2em;
            font-weight: bold;
            color: #4ecdc4;
            margin-bottom: 10px;
        }
        
        .getting-started {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .code-snippet {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 20px;
            font-family: 'Monaco', monospace;
            font-size: 0.9em;
            margin: 15px 0;
            overflow-x: auto;
            border-left: 4px solid #4ecdc4;
        }
        
        .footer {
            text-align: center;
            margin-top: 50px;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2.5em;
            }
            
            .examples-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 CortexKG Examples</h1>
            <p>Explore the power of cortical computing through interactive demonstrations<br>
            showcasing pattern recognition, semantic search, and temporal memory processing.</p>
        </div>
        
        <div class="examples-grid">
            <div class="example-card" onclick="location.href='pattern-recognition/'">
                <div class="example-icon">🎨</div>
                <div class="example-title">Pattern Recognition</div>
                <div class="example-description">
                    Interactive canvas for drawing patterns that CortexKG learns and recognizes. 
                    Demonstrates spatial pooling and cortical column activation in real-time.
                </div>
                <ul class="example-features">
                    <li>Hand-drawn pattern training</li>
                    <li>Real-time cortical visualization</li>
                    <li>Similarity scoring</li>
                    <li>Mobile touch support</li>
                    <li>Performance metrics</li>
                </ul>
                <button class="example-button">Try Pattern Recognition</button>
            </div>
            
            <div class="example-card" onclick="location.href='semantic-search/'">
                <div class="example-icon">🔍</div>
                <div class="example-title">Semantic Search</div>
                <div class="example-description">
                    Intelligent knowledge base that understands context and meaning. 
                    Add documents and search using natural language queries.
                </div>
                <ul class="example-features">
                    <li>Natural language queries</li>
                    <li>Semantic similarity scoring</li>
                    <li>Dynamic knowledge base</li>
                    <li>Contextual understanding</li>
                    <li>Relevance ranking</li>
                </ul>
                <button class="example-button">Explore Semantic Search</button>
            </div>
            
            <div class="example-card" onclick="location.href='memory-visualization/'">
                <div class="example-icon">📈</div>
                <div class="example-title">Temporal Memory</div>
                <div class="example-description">
                    Visualize how cortical temporal memory processes sequences over time. 
                    Watch predictions form and evolve with each time step.
                </div>
                <ul class="example-features">
                    <li>Sequence visualization</li>
                    <li>Prediction tracking</li>
                    <li>Temporal patterns</li>
                    <li>Interactive timeline</li>
                    <li>Memory state analysis</li>
                </ul>
                <button class="example-button">View Memory Visualization</button>
            </div>
            
            <div class="example-card" onclick="location.href='cortical-simulator/'">
                <div class="example-icon">⚡</div>
                <div class="example-title">Cortical Simulator</div>
                <div class="example-description">
                    Advanced simulation of cortical column behavior with adjustable parameters. 
                    Experiment with different algorithms and observe emergent patterns.
                </div>
                <ul class="example-features">
                    <li>Parameter adjustment</li>
                    <li>Algorithm comparison</li>
                    <li>Performance profiling</li>
                    <li>Export capabilities</li>
                    <li>Advanced analytics</li>
                </ul>
                <button class="example-button">Launch Simulator</button>
            </div>
            
            <div class="example-card" onclick="location.href='realtime-analytics/'">
                <div class="example-icon">📊</div>
                <div class="example-title">Real-time Analytics</div>
                <div class="example-description">
                    Monitor CortexKG performance metrics in real-time. Track memory usage, 
                    processing speed, and system health across different workloads.
                </div>
                <ul class="example-features">
                    <li>Live performance metrics</li>
                    <li>Memory monitoring</li>
                    <li>Processing analytics</li>
                    <li>Alert system</li>
                    <li>Historical tracking</li>
                </ul>
                <button class="example-button">Monitor Analytics</button>
            </div>
            
            <div class="example-card" onclick="location.href='integration-playground/'">
                <div class="example-icon">🔧</div>
                <div class="example-title">Integration Playground</div>
                <div class="example-description">
                    Test different integration scenarios and configurations. 
                    Perfect for developers learning to implement CortexKG in their projects.
                </div>
                <ul class="example-features">
                    <li>Configuration testing</li>
                    <li>API exploration</li>
                    <li>Code generation</li>
                    <li>Error simulation</li>
                    <li>Best practices guide</li>
                </ul>
                <button class="example-button">Open Playground</button>
            </div>
        </div>
        
        <div class="tech-specs">
            <h2>🚀 Technical Specifications</h2>
            <div class="specs-grid">
                <div class="spec-item">
                    <div class="spec-value">WebAssembly</div>
                    <div>High-performance computing</div>
                </div>
                <div class="spec-item">
                    <div class="spec-value">64MB</div>
                    <div>Default memory allocation</div>
                </div>
                <div class="spec-item">
                    <div class="spec-value">SIMD</div>
                    <div>Vectorized operations</div>
                </div>
                <div class="spec-item">
                    <div class="spec-value">IndexedDB</div>
                    <div>Persistent storage</div>
                </div>
                <div class="spec-item">
                    <div class="spec-value">Canvas 2D</div>
                    <div>Real-time visualization</div>
                </div>
                <div class="spec-item">
                    <div class="spec-value">TypeScript</div>
                    <div>Full type definitions</div>
                </div>
            </div>
        </div>
        
        <div class="getting-started">
            <h2>🏃‍♂️ Quick Start</h2>
            <p>Get up and running with CortexKG in just a few lines of code:</p>
            
            <div class="code-snippet">
// Initialize CortexKG
await wasmLoader.init({
    memorySize: 32 * 1024 * 1024,
    enableSIMD: true
});

// Allocate a concept
const concept = await cortexWrapper.allocateConcept('my-concept', 1024);

// Perform spatial pooling
const pattern = new Array(2048).fill(0).map(() => Math.random() > 0.8 ? 1 : 0);
const result = await cortexWrapper.spatialPooling(pattern);

console.log('Active columns:', result.activeColumns.length);
            </div>
            
            <p>Each example includes complete source code, detailed documentation, and step-by-step instructions for integration into your own projects.</p>
        </div>
        
        <div class="footer">
            <p>🧠 Powered by CortexKG - Cortical Computing for the Web</p>
            <p>Explore the source code and documentation for complete implementation details</p>
        </div>
    </div>
</body>
</html>
```

5. **Create example applications README and documentation**
```markdown
# CortexKG Example Applications

This directory contains comprehensive example applications demonstrating the capabilities of CortexKG across different use cases and scenarios.

## 📁 Directory Structure

```
examples/
├── index.html                     # Main examples portal
├── pattern-recognition/           # Interactive pattern learning demo
│   ├── index.html
│   ├── app.js
│   └── README.md
├── semantic-search/              # Knowledge base search application
│   ├── index.html
│   ├── app.js
│   └── README.md
├── memory-visualization/         # Temporal memory visualization
│   ├── index.html
│   ├── app.js
│   └── README.md
├── cortical-simulator/          # Advanced cortical simulation
│   ├── index.html
│   ├── simulator.js
│   ├── controls.js
│   └── README.md
├── realtime-analytics/          # Performance monitoring dashboard
│   ├── index.html
│   ├── dashboard.js
│   └── README.md
├── integration-playground/      # Developer testing environment
│   ├── index.html
│   ├── playground.js
│   └── README.md
└── shared/                      # Shared utilities and styles
    ├── common.css
    ├── utils.js
    └── test-data.js
```

## 🎯 Example Applications

### 1. Pattern Recognition Demo
**Path:** `pattern-recognition/`

Interactive demonstration of CortexKG's spatial pooling capabilities. Users can draw patterns on a canvas, train the system to recognize them, and test recognition accuracy.

**Key Features:**
- Hand-drawn pattern input via canvas
- Real-time cortical column visualization
- Pattern similarity scoring
- Mobile touch support
- Performance metrics and statistics

**Technologies:**
- Canvas 2D API for drawing
- CortexKG spatial pooling
- Touch event handling
- Real-time visualization

### 2. Semantic Search Application
**Path:** `semantic-search/`

Intelligent knowledge base that demonstrates semantic understanding and contextual search capabilities.

**Key Features:**
- Natural language query processing
- Dynamic knowledge base management
- Semantic similarity scoring
- Contextual result ranking
- Multi-modal search (text and patterns)

**Technologies:**
- CortexKG semantic processing
- IndexedDB for persistence
- Text analysis and tokenization
- Similarity algorithms

### 3. Temporal Memory Visualization
**Path:** `memory-visualization/`

Advanced visualization of how cortical temporal memory processes sequences over time, showing prediction formation and accuracy.

**Key Features:**
- Sequence timeline visualization
- Prediction vs. reality comparison
- Interactive playback controls
- Memory state analysis
- Custom sequence recording

**Technologies:**
- CortexKG temporal memory
- Canvas animation
- Timeline controls
- State visualization

### 4. Cortical Simulator
**Path:** `cortical-simulator/`

Comprehensive simulation environment for experimenting with cortical algorithms and parameters.

**Key Features:**
- Parameter adjustment interface
- Algorithm comparison tools
- Performance profiling
- Data export capabilities
- Advanced analytics

**Technologies:**
- Advanced CortexKG configuration
- WebWorkers for background processing
- Chart.js for analytics
- File API for data export

### 5. Real-time Analytics Dashboard
**Path:** `realtime-analytics/`

Monitoring dashboard for tracking CortexKG performance metrics and system health in real-time.

**Key Features:**
- Live performance metrics
- Memory usage monitoring
- Processing speed analytics
- Alert system for anomalies
- Historical data tracking

**Technologies:**
- WebSockets for real-time updates
- Chart.js for data visualization
- Local storage for historical data
- Alert system implementation

### 6. Integration Playground
**Path:** `integration-playground/`

Developer-focused environment for testing integration scenarios and exploring API functionality.

**Key Features:**
- Interactive API explorer
- Configuration testing
- Code generation tools
- Error simulation
- Best practices guidance

**Technologies:**
- Dynamic code execution
- API introspection
- Code generation
- Error handling examples

## 🚀 Getting Started

### Prerequisites

1. **Web Server**: Examples require a local web server due to WASM loading restrictions
2. **Modern Browser**: Chrome 88+, Firefox 89+, Safari 14+, or Edge 88+
3. **CortexKG Distribution**: Ensure CortexKG WASM and JS files are available

### Quick Setup

1. **Clone or download** the examples directory
2. **Start a local web server**:
   ```bash
   # Using Python
   python -m http.server 8000
   
   # Using Node.js
   npx serve .
   
   # Using PHP
   php -S localhost:8000
   ```
3. **Navigate** to `http://localhost:8000` in your browser
4. **Click** on any example card to explore the demonstrations

### Individual Example Setup

Each example can be run independently:

```bash
# Navigate to specific example
cd pattern-recognition

# Start local server
python -m http.server 8001

# Open http://localhost:8001 in browser
```

## 🔧 Customization Guide

### Adding New Examples

1. **Create directory** in `examples/` folder
2. **Copy template** from `shared/template/`
3. **Implement functionality** using CortexKG APIs
4. **Add entry** to main `index.html`
5. **Document** in README.md

### Modifying Existing Examples

Examples are designed to be easily customizable:

```javascript
// Modify initialization parameters
await wasmLoader.init({
    memorySize: 64 * 1024 * 1024,  // Increase memory
    enableSIMD: true,
    enableThreads: false,          // Enable for better performance
    debug: true                    // Enable for development
});

// Adjust spatial pooling parameters
const result = await cortexWrapper.spatialPooling(pattern, {
    sparsity: 0.03,               // Increase sparsity
    boostStrength: 3.0,           // Stronger boosting
    dutyCyclePeriod: 2000         // Longer duty cycle
});
```

### Configuration Options

Each example supports environment-specific configuration:

```javascript
const config = {
    development: {
        memorySize: 16 * 1024 * 1024,
        debug: true,
        enableProfiling: true
    },
    production: {
        memorySize: 64 * 1024 * 1024,
        debug: false,
        enableProfiling: false
    }
};
```

## 📊 Performance Considerations

### Memory Usage
- **Pattern Recognition**: ~16MB typical, 32MB recommended
- **Semantic Search**: ~32MB typical, 64MB recommended  
- **Memory Visualization**: ~32MB typical, 64MB recommended
- **Cortical Simulator**: ~64MB typical, 128MB recommended

### Browser Compatibility
| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebAssembly | ✅ 57+ | ✅ 52+ | ✅ 11+ | ✅ 16+ |
| SIMD | ✅ 91+ | ✅ 89+ | ❌ | ✅ 91+ |
| SharedArrayBuffer | ✅ 68+ | ✅ 79+ | ❌ | ✅ 79+ |
| OffscreenCanvas | ✅ 69+ | ✅ 105+ | ❌ | ✅ 79+ |

### Mobile Considerations
- **Touch Events**: All examples support touch input
- **Memory Limits**: Reduced memory allocation on mobile
- **Performance**: Automatic quality adjustment based on device capabilities
- **Responsive Design**: Adaptive layouts for different screen sizes

## 🐛 Troubleshooting

### Common Issues

**WASM Loading Errors**
```javascript
// Solution: Check file paths and server configuration
const wasmPath = './dist/cortex-kg.wasm';
if (!await checkFileExists(wasmPath)) {
    console.error('WASM file not found at:', wasmPath);
}
```

**Memory Allocation Failures**
```javascript
// Solution: Reduce memory size or implement cleanup
try {
    await wasmLoader.init({ memorySize: 32 * 1024 * 1024 });
} catch (error) {
    // Fallback to smaller memory size
    await wasmLoader.init({ memorySize: 16 * 1024 * 1024 });
}
```

**Browser Compatibility Issues**
```javascript
// Solution: Feature detection and fallbacks
if (!window.WebAssembly) {
    showFallbackMessage('WebAssembly not supported');
    return;
}

if (!navigator.userAgent.includes('Chrome')) {
    // Disable SIMD on non-Chrome browsers
    config.enableSIMD = false;
}
```

### Debug Mode

Enable debug mode for detailed logging:

```javascript
await wasmLoader.init({
    debug: true,
    logLevel: 'debug'  // 'debug', 'info', 'warn', 'error'
});
```

## 📚 Learning Path

Recommended order for exploring examples:

1. **Pattern Recognition** - Learn basic spatial pooling concepts
2. **Semantic Search** - Understand concept allocation and querying  
3. **Memory Visualization** - Explore temporal memory processing
4. **Integration Playground** - Practice API usage patterns
5. **Real-time Analytics** - Monitor system performance
6. **Cortical Simulator** - Advanced experimentation

## 🔗 Additional Resources

- **API Documentation**: `../docs/api-reference.md`
- **Integration Guide**: `../docs/integration-guide.md`
- **Troubleshooting**: `../docs/troubleshooting.md`
- **GitHub Repository**: `https://github.com/your-org/cortex-kg`
- **Community Forum**: `https://community.cortex-kg.org`

## 📄 License

All examples are provided under the same license as CortexKG. See LICENSE file for details.
```

## Expected Outputs
- Complete collection of interactive example applications demonstrating all major CortexKG features
- Pattern recognition demo with drawing interface and real-time cortical visualization
- Semantic search application with natural language processing and knowledge management
- Temporal memory visualization with sequence playback and prediction analysis
- Comprehensive documentation and setup instructions for all examples
- Mobile-responsive designs with touch support and performance optimizations

## Validation
1. Verify all example applications load and function correctly across different browsers
2. Test mobile responsiveness and touch interactions on various devices  
3. Confirm educational value and clear demonstration of CortexKG capabilities
4. Validate code quality, documentation completeness, and ease of customization
5. Ensure examples serve as effective learning tools for developers at different skill levels

## Next Steps
- Proceed to micro-phase 9.49 (Troubleshooting Guide)
- Deploy examples to demonstration server
- Create video tutorials for complex examples