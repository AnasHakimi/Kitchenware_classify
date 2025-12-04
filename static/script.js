const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const chartContainer = document.getElementById('chartContainer');

let selectedFile = null;

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Click to Browse
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
        uploadArea.style.display = 'none'; // Hide upload area to clean up UI
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        displayResults(data.predictions);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis.');
    } finally {
        analyzeBtn.textContent = 'Analyze Image';
        analyzeBtn.disabled = false;
    }
});

function displayResults(predictions) {
    resultsSection.style.display = 'block';
    chartContainer.innerHTML = '';

    predictions.forEach((pred, index) => {
        const percentage = (pred.probability * 100).toFixed(1);
        const isTop = index === 0;
        
        const row = document.createElement('div');
        row.className = `chart-row ${isTop ? 'top-result' : ''}`;
        
        row.innerHTML = `
            <div class="label-row">
                <span class="class-name">${pred.class}</span>
                <span class="percentage">${percentage}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: 0%"></div>
            </div>
        `;
        
        chartContainer.appendChild(row);
        
        // Animate bar
        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = `${percentage}%`;
        }, 50);
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}
