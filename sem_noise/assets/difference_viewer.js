// Native differences are decoded once. Changing range only recolours them.
(async () => {
    const element = id => document.getElementById(id);
    const error = message => { element('error').textContent = message; element('error').hidden = !message; };
    try {
        const payload = JSON.parse(element('difference-data').textContent);
        const packed = Uint8Array.from(atob(payload.data), c => c.charCodeAt(0));
        payload.data = '';
        element('difference-data').remove();
        const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
        const buffer = await new Response(stream).arrayBuffer();
        const bytes = new DataView(buffer);
        const [count, height, width] = payload.shape;
        const size = width * height;
        const precision = payload.bytes_per_value;
        if (buffer.byteLength !== count * size * precision) throw Error('Difference data length is incorrect.');
        const values = precision === 8 ? new Float64Array(count * size) : new Float32Array(count * size);
        for (let i = 0; i < values.length; i++) values[i] = precision === 8 ? bytes.getFloat64(i * 8, true) : bytes.getFloat32(i * 4, true);
        const canvases = [], cells = [], totals = [];
        const number = value => Number.isFinite(value) ? Number(value.toPrecision(7)).toString() : 'n/a';
        for (let panel = 0; panel < count; panel++) {
            const figure = document.createElement('figure');
            const caption = document.createElement('figcaption');
            caption.textContent = payload.labels[panel];
            const canvas = document.createElement('canvas');
            canvas.width = width; canvas.height = height;
            canvas.setAttribute('aria-label', payload.labels[panel] + ' difference map');
            figure.append(caption, canvas); element('panels').append(figure); canvases.push(canvas);
            let minimum = Infinity, maximum = -Infinity, sum = 0, total = 0;
            for (let p = 0; p < size; p++) {
                const v = values[panel * size + p];
                if (!Number.isFinite(v)) continue;
                minimum = Math.min(minimum, v); maximum = Math.max(maximum, v); sum += v * v; total++;
            }
            totals.push(total);
            const row = document.createElement('tr');
            for (const value of [payload.labels[panel], number(minimum), number(maximum), number(total ? Math.sqrt(sum / total) : NaN), '']) {
                const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
            }
            element('statistics').append(row); cells.push(row.lastChild);
            canvas.addEventListener('mousemove', event => {
                const rect = canvas.getBoundingClientRect();
                const x = Math.min(width - 1, Math.max(0, Math.floor((event.clientX - rect.left) * width / rect.width)));
                const y = Math.min(height - 1, Math.max(0, Math.floor((event.clientY - rect.top) * height / rect.height)));
                const value = values[panel * size + y * width + x];
                element('readout').textContent = `${payload.labels[panel]}: x=${x}, y=${y}, ${Number.isFinite(value) ? number(value) + ' DN' : 'invalid / failed stage'}`;
            });
        }
        const layout = () => { element('panels').style.gridTemplateColumns = `repeat(${count}, ${element('native').checked ? width + 'px' : 'minmax(0, 1fr)'})`; };
        element('native').addEventListener('change', layout); layout();
        let current = payload.maximum > 0 ? payload.maximum : 1;
        element('range').max = current;
        let scheduled = false;
        function render() {
            scheduled = false;
            const limit = current;
            for (let panel = 0; panel < count; panel++) {
                const context = canvases[panel].getContext('2d');
                const image = context.createImageData(width, height);
                let saturated = 0;
                for (let p = 0; p < size; p++) {
                    const value = values[panel * size + p];
                    const finite = Number.isFinite(value);
                    if (finite && Math.abs(value) > limit) saturated++;
                    const index = finite ? Math.max(0, Math.min(254, Math.round(127 + 127 * value / limit))) : 255;
                    image.data[p * 4] = payload.palette[index * 3];
                    image.data[p * 4 + 1] = payload.palette[index * 3 + 1];
                    image.data[p * 4 + 2] = payload.palette[index * 3 + 2];
                    image.data[p * 4 + 3] = 255;
                }
                context.putImageData(image, 0, 0);
                cells[panel].textContent = totals[panel] ? number(100 * saturated / totals[panel]) : 'n/a';
            }
            element('negative').textContent = '≤ −' + number(limit);
            element('positive').textContent = '≥ +' + number(limit);
            document.body.dataset.limit = limit;
        }
        function setLimit(value) {
            if (!(value > 0) || !Number.isFinite(value)) { error('Enter a finite colour limit greater than zero.'); return; }
            error(''); current = value;
            if (value > Number(element('range').max)) element('range').max = value;
            element('limit').value = value; element('range').value = value;
            if (!scheduled) { scheduled = true; setTimeout(render, 0); }
        }
        element('limit').addEventListener('input', event => setLimit(Number(event.target.value)));
        element('range').addEventListener('input', event => setLimit(Math.max(Number(event.target.value), Number.EPSILON * Number(event.target.max))));
        element('full').addEventListener('click', () => setLimit(payload.maximum > 0 ? payload.maximum : 1));
        element('download').addEventListener('click', () => {
            render();
            const exportedLimit = current;
            const joined = document.createElement('canvas'); joined.width = count * width + (count - 1) * 8; joined.height = height;
            const context = joined.getContext('2d'); context.fillStyle = '#808080'; context.fillRect(0, 0, joined.width, height);
            canvases.forEach((canvas, i) => context.drawImage(canvas, i * (width + 8), 0));
            joined.toBlob(blob => {
                if (!blob) { error('The browser could not export this image size.'); return; }
                const url = URL.createObjectURL(blob), link = document.createElement('a');
                link.href = url; link.download = `differences-limit-${exportedLimit}-DN.png`; link.click();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
            });
        });
        for (const id of ['limit', 'range', 'full', 'download']) element(id).disabled = false;
        element('loading').hidden = true;
        element('limit').value = current; element('range').value = current; render();
        document.body.dataset.ready = 'true';
    } catch (failure) {
        element('loading').hidden = true;
        error('Could not load interactive differences: ' + failure.message);
        document.body.dataset.ready = 'failed';
    }
})();
