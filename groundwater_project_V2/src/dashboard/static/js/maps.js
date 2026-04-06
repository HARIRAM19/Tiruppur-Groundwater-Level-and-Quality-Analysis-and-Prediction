// maps.js
function initializeMap(months, wells) {
    if (!months || months.length === 0) {
        document.getElementById('interpolationMap').innerHTML = '<p class="text-center mt-5">No data available for mapping.</p>';
        return;
    }

    var map = L.map('interpolationMap').setView([11.0, 77.5], 9);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap'
    }).addTo(map);

    var heatLayer = L.heatLayer([], { radius: 25, blur: 15, maxZoom: 10 }).addTo(map);
    var markerLayer = L.layerGroup().addTo(map);
    var boundaryLayer = L.layerGroup();

    // Add well markers
    wells.forEach(function (w) {
        var marker = L.marker([w.lat, w.lon]).bindPopup(
            `<b>Well ${w.well_id}</b><br>${w.village}<br><a href="/well/${w.well_id}">Details</a>`
        );
        markerLayer.addLayer(marker);
    });

    // Time slider
    var slider = document.getElementById('monthSlider');
    var monthLabel = document.getElementById('selectedMonth');

    function updateHeatmap(index) {
        var month = months[index];
        monthLabel.innerText = month;
        fetch(`/api/interpolated/${month}`)
            .then(response => response.json())
            .then(data => {
                if (data && data.length > 0) {
                    heatLayer.setLatLngs(data.map(p => [p[0], p[1], p[2]]));
                } else {
                    heatLayer.setLatLngs([]);
                    console.log('No heat data for', month);
                }
            })
            .catch(err => console.error('Error fetching heat data:', err));
    }

    slider.addEventListener('input', function (e) {
        updateHeatmap(e.target.value);
    });

    // Initial load
    if (months.length > 0) {
        updateHeatmap(0);
    }

    // Layer toggles
    document.getElementById('toggleHeatmap').addEventListener('change', function (e) {
        if (e.target.checked) map.addLayer(heatLayer);
        else map.removeLayer(heatLayer);
    });
    document.getElementById('toggleWells').addEventListener('change', function (e) {
        if (e.target.checked) map.addLayer(markerLayer);
        else map.removeLayer(markerLayer);
    });
    document.getElementById('toggleBoundaries').addEventListener('change', function (e) {
        if (e.target.checked) map.addLayer(boundaryLayer);
        else map.removeLayer(boundaryLayer);
    });

    // Click-to-well: find nearest well
    map.on('click', function (e) {
        var latlng = e.latlng;
        var minDist = Infinity;
        var nearest = null;
        wells.forEach(function (w) {
            var d = map.distance(latlng, L.latLng(w.lat, w.lon));
            if (d < minDist) {
                minDist = d;
                nearest = w;
            }
        });
        if (nearest) {
            document.getElementById('clickInfo').innerHTML =
                `Nearest: Well ${nearest.well_id} (${nearest.village})<br>Distance: ${(minDist / 1000).toFixed(2)} km`;
        }
    });
}