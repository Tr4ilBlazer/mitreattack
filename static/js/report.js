document.addEventListener('DOMContentLoaded', function() {
    // Create Tactic Distribution Chart
    const tacticCtx = document.getElementById('tacticChart').getContext('2d');
    const tacticChart = new Chart(tacticCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(tacticDistribution),
            datasets: [{
                label: 'Tactic Distribution',
                data: Object.values(tacticDistribution),
                backgroundColor: 'rgba(231, 76, 60, 0.7)',
                borderColor: 'rgba(231, 76, 60, 1)',
                borderWidth: 1
            }, {
                label: 'Confidence',
                data: Object.keys(tacticDistribution).map(tactic => tacticConfidences[tactic]),
                type: 'line',
                fill: false,
                borderColor: 'rgba(52, 152, 219, 1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Create Technique Distribution Chart
    const techniqueCtx = document.getElementById('techniqueChart').getContext('2d');
    const techniqueChart = new Chart(techniqueCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(techniqueDistribution),
            datasets: [{
                label: 'Technique Distribution',
                data: Object.values(techniqueDistribution),
                backgroundColor: 'rgba(41, 128, 185, 0.7)',
                borderColor: 'rgba(41, 128, 185, 1)',
                borderWidth: 1
            }, {
                label: 'Confidence',
                data: Object.keys(techniqueDistribution).map(technique => techniqueConfidences[technique]),
                type: 'line',
                fill: false,
                borderColor: 'rgba(230, 126, 34, 1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    document.getElementById('save-pdf').addEventListener('click', function() {
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.innerHTML = '<div class="spinner"></div><p>Generating PDF...</p>';
        document.body.appendChild(loadingOverlay);

        const element = document.getElementById('report-content');
        const opt = {
            margin: [10, 10],
            filename: 'threat_analysis_report.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2, logging: true, dpi: 192, letterRendering: true },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };

        // Convert charts to images before PDF generation
        Promise.all([
            convertChartToImage(tacticChart),
            convertChartToImage(techniqueChart)
        ]).then(([tacticImage, techniqueImage]) => {
            document.getElementById('tacticChart').replaceWith(tacticImage);
            document.getElementById('techniqueChart').replaceWith(techniqueImage);

            html2pdf().set(opt).from(element).save().then(() => {
                document.body.removeChild(loadingOverlay);
                // Restore original charts
                tacticImage.replaceWith(document.getElementById('tacticChart'));
                techniqueImage.replaceWith(document.getElementById('techniqueChart'));
            }).catch(error => {
                console.error('PDF generation failed:', error);
                document.body.removeChild(loadingOverlay);
                alert('PDF generation failed. Please try again.');
            });
        });
    });

    function convertChartToImage(chart) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = chart.toBase64Image();
        });
    }

    // Function to sanitize and format description text
    function sanitizeDescription(description) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(description, 'text/html');
        return doc.body.textContent || "";
    }

    // Format descriptions
    document.querySelectorAll('.description-text').forEach(element => {
        element.textContent = sanitizeDescription(element.textContent);
    });
});