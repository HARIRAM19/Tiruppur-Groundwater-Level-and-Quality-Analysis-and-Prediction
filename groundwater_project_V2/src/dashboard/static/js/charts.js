// Additional chart helpers (e.g., for correlation matrix)
function createCorrelationMatrix(containerId, data) {
    // data should be a 2D array of values
    Plotly.newPlot(containerId, [{
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis'
    }]);
}