// Navigation
const navClinical = document.getElementById('nav-clinical');
const navEval = document.getElementById('nav-eval');

const viewClinical = document.getElementById('view-clinical');
const viewEval = document.getElementById('view-eval');

navClinical.addEventListener('click', () => {
    setActiveNav(navClinical);
    showView(viewClinical);
});

navEval.addEventListener('click', () => {
    setActiveNav(navEval);
    showView(viewEval);
});

function setActiveNav(activeBtn) {
    [navClinical, navEval].forEach(btn => btn.classList.remove('active'));
    activeBtn.classList.add('active');
}

function showView(activeView) {
    [viewClinical, viewEval].forEach(view => view.style.display = 'none');
    activeView.style.display = 'block';
}

function selectRole(role) {
    document.getElementById('role-selection-screen').style.display = 'none';
    document.getElementById('main-app').style.display = 'flex';

    if (role === 'clinical') {
        navClinical.style.display = 'block';
        navEval.style.display = 'none';
        navClinical.click();
    } else {
        navClinical.style.display = 'none';
        navEval.style.display = 'block';
        navEval.click();
    }
}

function logout() {
    document.getElementById('role-selection-screen').style.display = 'flex';
    document.getElementById('main-app').style.display = 'none';

    // reset UI state
    analyzeBtn.disabled = true;
    reportBox.className = 'report-box waiting';
    reportBox.innerHTML = 'Awaiting image upload for analysis...';
    metricsGrid.style.display = 'none';
    resultImage.style.display = 'none';
    heatmapImage.style.display = 'none';
    heatmapLegend.style.display = 'none';
    overlayMode.style.display = 'none';
    pdfBtn.style.display = 'none';
    resultPlaceholder.style.display = 'block';
    previewImage.style.display = 'none';
    uploadPlaceholder.style.display = 'block';
    selectedFile = null;
    fileInput.value = '';
    overlayHelpText.style.display = 'none';
}

// Clinical Upload logic
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const analyzeBtn = document.getElementById('analyze-btn');

const reportBox = document.getElementById('report-box');
const metricsGrid = document.getElementById('metrics-grid');
const resultImage = document.getElementById('result-image');
const heatmapImage = document.getElementById('heatmap-image');
const heatmapLegend = document.getElementById('heatmap-legend');
const resultPlaceholder = document.getElementById('result-placeholder');
const overlayMode = document.getElementById('overlay-mode');
const pdfBtn = document.getElementById('pdf-btn');
const overlayHelpText = document.getElementById('overlay-help-text');

let selectedFile = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#0f766e';
    uploadArea.style.background = '#f0fdf4';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#cbd5e1';
    uploadArea.style.background = '#f8fafc';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#cbd5e1';
    uploadArea.style.background = '#f8fafc';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadPlaceholder.style.display = 'none';
        analyzeBtn.disabled = false;
    }
    reader.readAsDataURL(selectedFile);
}

// Toggle overlays
overlayMode.addEventListener('change', (e) => {
    if (e.target.value === 'heatmap') {
        resultImage.style.display = 'none';
        heatmapImage.style.display = 'block';
        heatmapLegend.style.display = 'block';
        overlayHelpText.style.display = 'block';
        overlayHelpText.textContent = 'Showing color-coded certainty map. Red/Orange areas show where the AI is highly sure there is a spot, while Yellow/Blue areas show where it is less certain.';
    } else {
        resultImage.style.display = 'block';
        heatmapImage.style.display = 'none';
        heatmapLegend.style.display = 'none';
        overlayHelpText.style.display = 'block';
        overlayHelpText.textContent = 'Showing the solid outline of the detected skin spot boundary.';
    }
});

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const activeModel = document.getElementById('model-arch-select')?.value || 'unet';
    if (window.logToTerminal) {
        window.logToTerminal('API', `POST /api/predict?model=${activeModel.toUpperCase()} -> Spawned boundary search thread`);
    }

    analyzeBtn.textContent = 'Processing Image Data...';
    analyzeBtn.disabled = true;
    reportBox.className = 'report-box waiting';
    reportBox.innerHTML = 'AI is analyzing the structural geometry...';
    metricsGrid.style.display = 'none';
    resultImage.style.display = 'none';
    heatmapImage.style.display = 'none';
    heatmapLegend.style.display = 'none';
    overlayMode.style.display = 'none';
    overlayHelpText.style.display = 'none';
    pdfBtn.style.display = 'none';
    resultPlaceholder.style.display = 'block';

    const formData = new FormData();
    formData.append('file', selectedFile);
    
    const modelSelect = document.getElementById('model-arch-select');
    if (modelSelect) {
        formData.append('model', modelSelect.value);
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();

        // Update UI
        let statusHtml = '';
        let showOverlaySelector = true;
        if (data.status === 'unrelated') {
            reportBox.className = 'report-box unrelated';
            statusHtml = `<strong style="font-size:1.2em; display:block; margin-bottom:8px;">❌ Unrecognized Image</strong>${data.message}`;
            showOverlaySelector = false;
        } else if (data.status === 'warning') {
            reportBox.className = 'report-box warning';
            statusHtml = `<strong style="font-size:1.2em; display:block; margin-bottom:8px;">⚠️ Scan Quality Warning</strong>${data.message}`;
        } else if (data.status === 'anomaly') {
            reportBox.className = 'report-box danger';
            statusHtml = `<strong style="font-size:1.2em; display:block; margin-bottom:8px;">⚠️ Skin Spot Detected</strong>${data.message}`;
        } else {
            reportBox.className = 'report-box safe';
            statusHtml = `<strong style="font-size:1.2em; display:block; margin-bottom:8px;">✅ No Spots Detected</strong>${data.message}`;
        }
        reportBox.innerHTML = statusHtml;

        metricsGrid.style.display = 'grid';
        document.getElementById('area-val').textContent = data.area_percentage;
        document.getElementById('conf-val').textContent = data.confidence;
        document.getElementById('irregularity-val').textContent = data.irregularity || '1.00';
        document.getElementById('diameter-val').textContent = data.diameter || '0.0 mm';
        document.getElementById('asymmetry-val').textContent = data.asymmetry || '0.0%';

        resultImage.src = data.overlay_image;
        heatmapImage.src = data.heatmap_image;

        // Default view: boundary
        overlayMode.value = 'boundary';
        resultImage.style.display = 'block';
        heatmapImage.style.display = 'none';

        if (showOverlaySelector) {
            overlayMode.style.display = 'block';
            overlayHelpText.style.display = 'block';
            overlayHelpText.textContent = 'Showing the solid outline of the detected skin spot boundary.';
        } else {
            overlayMode.style.display = 'none';
            overlayHelpText.style.display = 'none';
        }
        pdfBtn.style.display = 'flex';
        resultPlaceholder.style.display = 'none';
        
        if (window.logToTerminal) {
            window.logToTerminal('SUCCESS', `Inference finished. Spot Area: ${data.area_percentage}, Diameter: ${data.diameter || '0.0 mm'}, Certainty: ${data.confidence}`);
        }

    } catch (error) {
        console.error(error);
        reportBox.className = 'report-box danger';
        reportBox.innerHTML = 'An error occurred during analysis.';
        if (window.logToTerminal) {
            window.logToTerminal('ERROR', `Inference aborted. Reason: Server connection failed or invalid image file.`);
        }
    } finally {
        analyzeBtn.textContent = 'Execute AI Analysis';
        analyzeBtn.disabled = false;
    }
});

// PDF Generation
pdfBtn.addEventListener('click', () => {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    doc.setFont("Helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(13, 148, 136); // Medical Green color matching theme
    doc.text("Derma-AI Clinical Diagnostic Report", 20, 28);

    doc.setFont("Helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(100, 116, 139);
    doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, 36);
    doc.line(20, 40, 190, 40);

    // Patient EHR Registry Meta section
    const patientIdVal = document.getElementById('patient-id').value || "N/A";
    const scanDateVal = document.getElementById('scan-date').value || "N/A";
    doc.setFont("Helvetica", "bold");
    doc.setTextColor(71, 85, 105);
    doc.text("PATIENT REGISTRY PROFILE", 20, 48);
    doc.setFont("Helvetica", "normal");
    doc.setTextColor(15, 23, 42);
    doc.text(`EHR Patient ID: ${patientIdVal}`, 20, 55);
    doc.text(`Scan Capture Date: ${scanDateVal}`, 110, 55);
    doc.line(20, 60, 190, 60);

    doc.setFont("Helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(15, 23, 42);
    doc.text("Clinical & Diagnostic Metrics", 20, 72);

    doc.setFont("Helvetica", "normal");
    doc.setFontSize(11);
    doc.text(`Estimated Spot Size: ${document.getElementById('area-val').textContent}`, 20, 82);
    doc.text(`AI Certainty Rating: ${document.getElementById('conf-val').textContent}`, 20, 90);
    doc.text(`Spot Border Shape (Irregularity): ${document.getElementById('irregularity-val').textContent}`, 20, 98);
    doc.text(`Physical Spot Diameter: ${document.getElementById('diameter-val').textContent}`, 20, 106);
    doc.text(`Asymmetry Overlap Index: ${document.getElementById('asymmetry-val').textContent}`, 20, 114);

    doc.setFont("Helvetica", "bold");
    doc.text("Diagnostic Analysis Summary:", 20, 126);
    doc.setFont("Helvetica", "normal");
    const splitText = doc.splitTextToSize(
        reportBox.innerText
            .replace("⚠️ Skin Spot Detected\n", "")
            .replace("✅ No Spots Detected\n", "")
            .replace("⚠️ Scan Quality Warning\n", "")
            .replace("❌ Unrecognized Image\n", ""),
        170
    );
    doc.text(splitText, 20, 134);

    try {
        doc.addImage(previewImage.src, 'JPEG', 20, 150, 75, 75);
        doc.text("Source Dermoscopy Scan", 20, 232);

        const activeImg = overlayMode.value === 'heatmap' ? heatmapImage.src : resultImage.src;
        doc.addImage(activeImg, 'JPEG', 105, 150, 75, 75);
        doc.text("AI Segmentation Overlay", 105, 232);
    } catch (e) {
        doc.setFontSize(9);
        doc.setTextColor(150, 150, 150);
        doc.text("[Image render skipped in PDF due to base64 dimensions]", 20, 150);
    }

    doc.save(`DermaAI_Clinical_Report_${patientIdVal}_${Date.now()}.pdf`);
});



// Evaluator logic
const runEvalBtn = document.getElementById('run-eval-btn');
const evalBatches = document.getElementById('eval-batches');
const evalSamples = document.getElementById('eval-samples');
const galleryPlaceholder = document.getElementById('gallery-placeholder');
const evalRoc = document.getElementById('eval-roc');

evalBatches.addEventListener('input', e => document.getElementById('batch-val').textContent = e.target.value);
evalSamples.addEventListener('input', e => document.getElementById('sample-val').textContent = e.target.value);

runEvalBtn.addEventListener('click', async () => {
    if (window.logToTerminal) {
        window.logToTerminal('API', `POST /api/evaluate?batches=${evalBatches.value}&samples=${evalSamples.value} -> Spawned dataset loader thread`);
    }

    runEvalBtn.disabled = true;
    runEvalBtn.textContent = 'Running Validation...';
    document.getElementById('eval-log').innerHTML = '⏳ <strong>Running evaluation sequence...</strong> Please wait, this may take some time for processing due to the computational load.';
    document.getElementById('eval-gallery').style.display = 'none';
    evalRoc.style.display = 'none';
    galleryPlaceholder.style.display = 'flex';
    galleryPlaceholder.innerHTML = `
        <div style="text-align: center; padding: 20px;">
            <svg class="animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="width: 28px; height: 28px; color: var(--primary); margin: 0 auto 12px auto; animation: spin 1s linear infinite;">
                <circle cx="12" cy="12" r="10" stroke-dasharray="40 20"/>
            </svg>
            <p style="font-weight: 600; font-size: 1.05em; color: var(--text-primary); margin-bottom: 6px;">Processing Deep Learning Validation...</p>
            <p style="font-size: 0.85em; color: var(--text-secondary); max-width: 420px; margin: 0 auto; line-height: 1.45;">
                Analyzing validation skin scans and ground truths. This task calculates spatial boundaries and overlap statistics across all batches. Please stand by.
            </p>
        </div>
    `;

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                batches: parseInt(evalBatches.value),
                samples: parseInt(evalSamples.value)
            })
        });

        const data = await response.json();

        if (data.error) {
            document.getElementById('eval-log').innerHTML = `<span style="color:#ef4444;">[ERROR] ${data.error}</span>`;
            return;
        }

        document.getElementById('eval-dice').textContent = data.mean_dice;
        document.getElementById('eval-dice').style.color = parseFloat(data.mean_dice) > 80 ? '#10b981' : '#f59e0b';

        document.getElementById('eval-iou').textContent = data.mean_iou;
        document.getElementById('eval-iou').style.color = parseFloat(data.mean_iou) > 70 ? '#10b981' : '#f59e0b';

        document.getElementById('eval-vol').textContent = data.total_eval;

        document.getElementById('eval-log').innerHTML = `<span style="color:#10b981;">[SUCCESS] Validation complete. Processed ${data.total_eval} images.</span>`;

        document.getElementById('eval-gallery').src = data.gallery_image;
        document.getElementById('eval-gallery').style.display = 'block';
        galleryPlaceholder.style.display = 'none';

        if (data.roc_image) {
            evalRoc.src = data.roc_image;
            evalRoc.style.display = 'block';
        }
        
        if (window.logToTerminal) {
            window.logToTerminal('SUCCESS', `Validation complete. Evaluated ${data.total_eval} images. Mean Dice: ${data.mean_dice}%, Jaccard IoU: ${data.mean_iou}%`);
        }

    } catch (err) {
        document.getElementById('eval-log').innerHTML = '<span style="color:#ef4444;">[ERROR] Server connection failed.</span>';
    } finally {
        runEvalBtn.disabled = false;
        runEvalBtn.textContent = 'Execute Validation Run';
    }
});

// Workstation Tab Switcher Controller
window.switchClinicalTab = function (tabName) {
    const tabs = ['visual', 'pathology', 'advisor'];
    tabs.forEach(t => {
        const pane = document.getElementById(`tab-${t}`);
        const btn = document.getElementById(`tab-btn-${t}`);
        if (pane && btn) {
            if (t === tabName) {
                pane.style.display = 'block';
                btn.classList.add('active');
            } else {
                pane.style.display = 'none';
                btn.classList.remove('active');
            }
        }
    });
};

// Telemetry Analytics Tab Switcher Controller
window.switchTelemTab = function (tabName) {
    const tabs = ['engine', 'abcd'];
    tabs.forEach(t => {
        const pane = document.getElementById(`telem-${t}`);
        const btn = document.getElementById(`telem-btn-${t}`);
        if (pane && btn) {
            if (t === tabName) {
                pane.style.display = 'block';
                btn.classList.add('active');
            } else {
                pane.style.display = 'none';
                btn.classList.remove('active');
            }
        }
    });
    if (tabName === 'abcd' && window.calculateABCDRisk) {
        window.calculateABCDRisk();
    }
};

// ==========================================
// Interactive ABCD Risk Simulator Sandbox Controller
// ==========================================
const simAsymmetry = document.getElementById('sim-asymmetry');
const simBorder = document.getElementById('sim-border');
const simColor = document.getElementById('sim-color');
const simDiameter = document.getElementById('sim-diameter');

const simAsymmetryLabel = document.getElementById('sim-asymmetry-label');
const simBorderLabel = document.getElementById('sim-border-label');
const simColorLabel = document.getElementById('sim-color-label');
const simDiameterLabel = document.getElementById('sim-diameter-label');

const simBadge = document.getElementById('sim-badge');
const simScore = document.getElementById('sim-score');
const simProgressBar = document.getElementById('sim-progress-bar');
const simVerdict = document.getElementById('sim-verdict');

function calculateABCDRisk() {
    if (!simAsymmetry) return;
    
    // 1. Get raw slider values
    const A_pct = parseInt(simAsymmetry.value);
    const B_raw = parseInt(simBorder.value) / 100.0;
    const C_val = parseInt(simColor.value);
    const D_mm = parseInt(simDiameter.value) / 10.0;

    // 2. Update descriptive slider labels in real-time
    simAsymmetryLabel.textContent = `${A_pct}%`;
    simBorderLabel.textContent = B_raw.toFixed(2);
    simColorLabel.textContent = C_val === 1 ? '1 color' : `${C_val} colors`;
    simDiameterLabel.textContent = `${D_mm.toFixed(1)} mm`;

    // 3. Clinical scoring coefficients (ABCD rule formula)
    // Asymmetry score: maps 0-100% to 0-2 points. Coeff: 1.3
    const A_points = A_pct > 35 ? 2 : (A_pct > 15 ? 1 : 0);
    // Border score: maps 1.00-3.00 to 0-8 points. Coeff: 0.1
    const B_points = Math.min(8, Math.max(0, Math.floor((B_raw - 1.0) * 4)));
    // Color score: number of colors. Coeff: 0.5
    const C_points = C_val;
    // Diameter score: maps mm size to points. Coeff: 0.5
    const D_points = D_mm > 6.0 ? 3 : (D_mm > 3.0 ? 2 : 1);

    // Standard ABCD equation: Total Score = (A*1.3) + (B*0.1) + (C*0.5) + (D*0.5)
    const score = (A_points * 1.3) + (B_points * 0.1) + (C_points * 0.5) + (D_points * 0.5);
    simScore.textContent = score.toFixed(2);

    // 4. Determine diagnostic triage bands and visually style the components
    // Scale total score (max possible ~ 8.0) to a percentage for progress bar
    const progressPct = Math.min(100, Math.max(10, (score / 8.0) * 100));
    simProgressBar.style.width = `${progressPct}%`;

    if (score < 4.75) {
        // Benign Band
        simBadge.textContent = 'Benign Features';
        simBadge.style.background = '#e6fbf3';
        simBadge.style.color = '#059669';
        simScore.style.color = '#059669';
        simProgressBar.style.backgroundColor = '#10b981';
        simVerdict.innerHTML = 'This simulated spot presents <strong>benign</strong> features. Recommend standard annual skin exams and monitoring for changes.';
    } else if (score >= 4.75 && score <= 5.45) {
        // Suspicious / Atypical Band
        simBadge.textContent = 'Suspicious / Atypical';
        simBadge.style.background = '#fffbeb';
        simBadge.style.color = '#d97706';
        simScore.style.color = '#d97706';
        simProgressBar.style.backgroundColor = '#f59e0b';
        simVerdict.innerHTML = 'This simulated spot presents <strong>atypical / borderline</strong> features. Recommend a standard clinical review or short-term follow-up within 3 months.';
    } else {
        // Highly Atypical / Excision Warranted Band
        simBadge.textContent = 'Highly Atypical';
        simBadge.style.background = '#fff1f2';
        simBadge.style.color = '#e11d48';
        simScore.style.color = '#e11d48';
        simProgressBar.style.backgroundColor = '#f43f5e';
        simVerdict.innerHTML = 'This simulated spot presents <strong>highly atypical</strong> features. Recommended action: refer to a dermatologist for dermoscopy evaluation and potential diagnostic excision.';
    }
}

// Bind slider change listeners
if (simAsymmetry) {
    [simAsymmetry, simBorder, simColor, simDiameter].forEach(input => {
        input.addEventListener('input', calculateABCDRisk);
    });
    // Trigger initial calculation
    calculateABCDRisk();
    window.calculateABCDRisk = calculateABCDRisk;
}

// Real-time Console Log Terminal Logger Utility
function logToTerminal(level, message) {
    const terminal = document.getElementById('terminal-log-box');
    if (!terminal) return;
    
    const timestamp = new Date().toTimeString().split(' ')[0];
    const colorMap = {
        'INFO': '#e2e8f0',
        'SUCCESS': '#34d399',
        'WARNING': '#fbbf24',
        'ERROR': '#f87171',
        'API': '#38bdf8'
    };
    const color = colorMap[level] || '#ffffff';
    const logLine = document.createElement('div');
    logLine.innerHTML = `<span style="color: #475569;">[${timestamp}]</span> <span style="color: ${color};">[${level}] ${message}</span>`;
    terminal.appendChild(logLine);
    terminal.scrollTop = terminal.scrollHeight;
}
window.logToTerminal = logToTerminal;
