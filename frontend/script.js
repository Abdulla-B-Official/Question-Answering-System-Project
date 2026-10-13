async function loadRandomPassage() {
    const loading = document.getElementById('loading');
    loading.style.display = 'block';
    loading.textContent = 'Fetching random passage and 5 questions from validation pool...';

    try {
        const response = await fetch('/api/random-samples');
        const data = await response.json();

        if (data.success) {
            document.getElementById('contextInput').value = data.context;
            
            // Populate the question dropdown with up to 5 questions
            const select = document.getElementById('questionSelect');
            select.innerHTML = '<option value="">-- Choose one of the 5 generated questions --</option>';
            
            data.questions.forEach((q, index) => {
                const opt = document.createElement('option');
                opt.value = q;
                opt.textContent = `Question ${index + 1}: ${q}`;
                select.appendChild(opt);
            });

            document.getElementById('questionInput').value = '';
            document.getElementById('resultContainer').style.display = 'none';
        } else {
            alert('Error loading random sample.');
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect to backend.');
    } finally {
        loading.style.display = 'none';
    }
}

function selectQuestion(selectElement) {
    if (selectElement.value) {
        document.getElementById('questionInput').value = selectElement.value;
    }
}

async function handlePredict() {
    const context = document.getElementById('contextInput').value.trim();
    const question = document.getElementById('questionInput').value.trim();
    const resultContainer = document.getElementById('resultContainer');
    const answerOutput = document.getElementById('answerOutput');
    const confidenceOutput = document.getElementById('confidenceOutput');
    const loading = document.getElementById('loading');

    if (!context || !question) {
        alert('Please provide both a context passage and a question.');
        return;
    }

    loading.style.display = 'block';
    loading.textContent = 'Extracting answer and computing confidence...';
    resultContainer.style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ context, question })
        });

        const data = await response.json();

        if (data.success) {
            answerOutput.textContent = data.answer;
            confidenceOutput.textContent = data.confidence;
            resultContainer.style.display = 'block';
        } else {
            alert('Error: ' + (data.error || 'Prediction failed.'));
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect to the backend server.');
    } finally {
        loading.style.display = 'none';
    }
}