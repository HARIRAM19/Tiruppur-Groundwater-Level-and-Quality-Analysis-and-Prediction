// Global utility functions
function downloadPlot(divId, filename = 'plot') {
    Plotly.downloadImage(divId, {
        format: 'png',
        width: 800,
        height: 600,
        filename: filename
    });
}

// Format numbers
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}