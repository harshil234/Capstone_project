
document.addEventListener('DOMContentLoaded', function () {
    // Chatbot Submit Handler
    document.getElementById('chat-form')?.addEventListener('submit', async function (e) {
        e.preventDefault();
        const messageInput = document.getElementById('chat-input');
        const responseBox = document.getElementById('chat-response');
        const message = messageInput.value.trim();
        if (!message) return;

        responseBox.innerHTML = '⏳ Thinking...';

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            const data = await res.json();
            responseBox.innerHTML = data.response || data.error || '⚠️ Unexpected error';
        } catch (err) {
            responseBox.innerHTML = '❌ Error contacting chatbot.';
        }
    });

    // CTR Prediction Handler
    document.getElementById('ctr-form')?.addEventListener('submit', async function (e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const input = Object.fromEntries(formData.entries());
        const output = document.getElementById('ctr-result');
        output.textContent = '⏳ Predicting...';

        try {
            const res = await fetch('/api/predict-ctr', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(input)
            });
            const data = await res.json();
            output.textContent = `CTR: ${data.ctr?.toFixed(3) || 'N/A'}`;
        } catch {
            output.textContent = '❌ Error predicting CTR';
        }
    });

    // ROI Prediction Handler
    document.getElementById('roi-form')?.addEventListener('submit', async function (e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const input = Object.fromEntries(formData.entries());
        const output = document.getElementById('roi-result');
        output.textContent = '⏳ Predicting...';

        try {
            const res = await fetch('/api/predict-roi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(input)
            });
            const data = await res.json();
            output.textContent = `ROI: ${data.roi?.toFixed(2) || 'N/A'}`;
        } catch {
            output.textContent = '❌ Error predicting ROI';
        }
    });

    // Thumbnail Upload Handler
    document.getElementById('thumbnail-form')?.addEventListener('submit', async function (e) {
        e.preventDefault();
        const input = document.getElementById('thumbnail-input');
        const file = input.files[0];
        const output = document.getElementById('thumbnail-analysis');
        const chart = document.getElementById('thumbnail-chart');

        if (!file) {
            output.textContent = '❌ Please select a file';
            return;
        }

        const formData = new FormData();
        formData.append('thumbnail', file);

        output.textContent = '⏳ Analyzing...';

        try {
            const res = await fetch('/api/analyze-thumbnail', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            output.innerHTML = `<strong>${data.performance}</strong><br>${data.performance_factors.join('<br>')}`;
            chart.src = data.analysis_chart;
        } catch (err) {
            output.textContent = '❌ Failed to analyze thumbnail';
        }
    });
});
