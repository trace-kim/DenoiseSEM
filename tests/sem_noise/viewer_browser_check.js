window.addEventListener('load', async () => {
    const status = document.createElement('p'); status.id = 'browser-check'; document.body.append(status);
    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
    const assert = (ok, message) => { if (!ok) throw Error(message); };
    const wait = async condition => {
        for (let i = 0; i < 200; i++) { if (condition()) return; await sleep(20); }
        throw Error('Timed out waiting for viewer');
    };
    let stage = 'initialization';
    try {
        await wait(() => document.body.dataset.ready);
        assert(document.body.dataset.ready === 'true', 'Viewer failed to initialize');
        const field = document.getElementById('limit'), slider = document.getElementById('range');
        const canvases = [...document.querySelectorAll('canvas')];
        assert(canvases.length === 5 && canvases.every(c => c.width === 8 && c.height === 8), 'Native dimensions lost');
        const metrics = () => [...document.querySelectorAll('#statistics tr')].map(row => [...row.children].slice(0, 4).map(cell => cell.textContent));
        const before = JSON.stringify(metrics());
        const rgba = (panel, x, y) => [...canvases[panel].getContext('2d').getImageData(x, y, 1, 1).data].join(',');
        const set = async value => { stage = 'DN entry ' + value; field.value = value; field.dispatchEvent(new Event('input')); await wait(() => Number(document.body.dataset.limit) === value); };
        for (const value of [3.7, 0.03125, 1e-6, 70000, 6]) {
            await set(value);
            assert(Number(field.value) === value && Number(slider.value) === value, 'Exact entry altered');
            assert(JSON.stringify(metrics()) === before, 'Range changed numerical metrics');
            assert(rgba(0, 0, 0) === '128,128,128,255', 'Invalid pixel lost');
            assert(rgba(3, 3, 3) === '128,128,128,255', 'Failed stage not grey');
            assert(rgba(0, 3, 1) === '255,255,255,255', 'Zero is not white');
        }
        // A previously saturated pixel must recover its actual value when widened.
        await set(3.7); const clipped = rgba(0, 5, 1);
        await set(6); const widened = rgba(0, 5, 1);
        assert(clipped === '217,51,38,255' && widened !== clipped, 'Viewer recolours a clipped PNG instead of numerical differences');
        stage = 'slider'; slider.value = 12.345; slider.dispatchEvent(new Event('input'));
        await wait(() => Number(document.body.dataset.limit) === 12.345);
        const previous = Number(document.body.dataset.limit);
        for (const invalid of ['0', '-1', '']) {
            field.value = invalid; field.dispatchEvent(new Event('input'));
            assert(!document.getElementById('error').hidden, 'Invalid range silently accepted');
            assert(Number(document.body.dataset.limit) === previous, 'Invalid range changed image');
        }
        stage = 'full range'; document.getElementById('full').click();
        await wait(() => Number(document.body.dataset.limit) === 65535);
        assert([...document.querySelectorAll('#statistics tr')].filter((_, i) => i !== 3).every(row => row.lastChild.textContent === '0'), 'Full range still clips');
        const native = document.getElementById('native'); native.checked = true; native.dispatchEvent(new Event('change'));
        assert(document.getElementById('panels').style.gridTemplateColumns.includes('8px'), 'Native view does not use native dimensions');
        const rect = canvases[0].getBoundingClientRect();
        canvases[0].dispatchEvent(new MouseEvent('mousemove', {clientX: rect.left + 1.5, clientY: rect.top + 1.5}));
        assert(document.getElementById('readout').textContent.includes('-5 DN'), 'Pixel readout is incorrect');
        let exported;
        URL.createObjectURL = blob => { exported = blob; return 'blob:test'; };
        HTMLAnchorElement.prototype.click = function () {};
        stage = 'PNG export'; document.getElementById('download').click();
        await wait(() => exported);
        const png = new DataView(await exported.arrayBuffer());
        assert(png.getUint32(16) === 72 && png.getUint32(20) === 8, 'Export dimensions incorrect');
        status.textContent = 'PASS: arbitrary DN, slider, reset, clipping recovery, masks, metrics, native pixels, readout, PNG export';
    } catch (error) { status.textContent = 'FAIL at ' + stage + ': ' + error.message; }
});
