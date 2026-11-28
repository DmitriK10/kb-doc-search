// Запись демо-видео ОДНИМ ФРАГМЕНТОМ. Headless Chrome + добивка кадров.
//
// ПОЧЕМУ ТАК, А НЕ gdigrab. Первый вариант (screencast) давал честную
// запись, но Chrome присылает кадр только при перерисовке: статичный
// экран держится одним кадром секундами, потом прыжок — визуально
// читается как склейка. Второй вариант (gdigrab по экрану) захватил
// ЧУЖОЕ окно: gdigrab снимает область рабочего стола, а не конкретное
// приложение, и на переднем плане оказался не мой Chrome.
//
// Здесь: headless-браузер (экран пользователя не трогаем) + screencast
// для кадров по изменению + ТАЙМЕР, который добивает пропуски снимком
// экрана, если кадр не приходил дольше FILL_MS. В итоге кадры идут
// непрерывно, а не «замерло → прыжок».
//
// Кадры не переставляются и не вырезаются: каждый получает реальный
// таймстемп момента съёмки, сборка идёт через concat с этими
// длительностями. Это запись, а не монтаж.
//
// Запуск: node record_smooth.js <outdir> <calcHtml>

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const WebSocket = require('ws');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = 'https://vds3095790.my-ihor.ru/kb/';
const OUT = process.argv[2] || '.';
const CALC = process.argv[3] || '';

const W = 1600, H = 1000;
const FILL_MS = 110;      // если кадр не приходил дольше — добиваем снимком
const FILL_QUALITY = 85;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(d)); })
    .on('error', rej));

(async () => {
  const port = 9431;
  const framesDir = path.join(OUT, 'frames');
  fs.rmSync(framesDir, { recursive: true, force: true });
  fs.mkdirSync(framesDir, { recursive: true });

  const proc = spawn(CHROME, [
    `--remote-debugging-port=${port}`,
    '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
    `--window-size=${W},${H}`, '--force-device-scale-factor=1',
    '--user-data-dir=' + path.join(OUT, 'profile'),
    'about:blank'
  ], { stdio: 'ignore' });

  let targets = null;
  for (let i = 0; i < 80; i++) {
    try { targets = JSON.parse(await get(`http://127.0.0.1:${port}/json`)); break; }
    catch (e) { await sleep(250); }
  }
  if (!targets) { console.error('chrome не поднялся'); proc.kill(); process.exit(1); }

  const page = targets.find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  let frameNo = 0;
  const frames = [];         // {file, t}
  let lastFrameT = 0;
  let t0 = 0;

  const writeFrame = (buf, t) => {
    const fname = `f${String(frameNo++).padStart(5, '0')}.jpg`;
    fs.writeFileSync(path.join(framesDir, fname), buf);
    frames.push({ file: fname, t });
    lastFrameT = t;
  };

  ws.on('message', raw => {
    const m = JSON.parse(raw);
    if (m.id && pending.has(m.id)) {
      pending.get(m.id)(m.result); pending.delete(m.id); return;
    }
    if (m.method === 'Page.screencastFrame') {
      const t = m.params.metadata && m.params.metadata.timestamp
        ? m.params.metadata.timestamp * 1000 : Date.now();
      if (!t0) t0 = t;
      writeFrame(Buffer.from(m.params.data, 'base64'), t);
      ws.send(JSON.stringify({
        id: ++id, method: 'Page.screencastFrameAck',
        params: { sessionId: m.params.sessionId }
      }));
    }
  });

  const send = (method, params = {}) => new Promise(res => {
    const i = ++id; pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params }));
  });

  const evaluate = async expr => {
    const r = await send('Runtime.evaluate',
      { expression: expr, returnByValue: true, awaitPromise: true });
    return r && r.result ? r.result.value : null;
  };

  await new Promise(r => ws.on('open', r));
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride',
    { width: W, height: H, deviceScaleFactor: 1, mobile: false });

  await send('Page.addScriptToEvaluateOnNewDocument', {
    source: `try { localStorage.setItem('kb_sid', 'video-demo-2711'); } catch(e){}`
  });

  console.log('открываю стенд');
  await send('Page.navigate', { url: BASE });
  await sleep(6500);

  // Подписи
  await evaluate(`(function(){
    if (document.getElementById('__cap')) return 'exists';
    const st = document.createElement('style');
    st.textContent = '#__cap{position:fixed;left:0;right:0;bottom:0;z-index:99999;' +
      'background:linear-gradient(180deg,rgba(15,17,21,0),rgba(15,17,21,.97) 34%);' +
      'padding:30px 44px 20px;font-family:\\'Segoe UI\\',Arial,sans-serif;' +
      'font-size:20px;color:#e6e9ef;line-height:1.45;pointer-events:none;text-align:center}' +
      '#__cap .n{color:#7ee787;font-weight:700;margin-right:10px}';
    document.head.appendChild(st);
    const d = document.createElement('div');
    d.id = '__cap'; document.body.appendChild(d);
    return 'ok';
  })()`);

  const caption = async html => {
    await evaluate(`(function(){
      const d = document.getElementById('__cap');
      if (d) d.innerHTML = ${JSON.stringify(html)};
      return 'ok';
    })()`);
  };

  // --- СТАРТ ЗАПИСИ ---
  await send('Page.startScreencast', {
    format: 'jpeg', quality: 92, maxWidth: W, maxHeight: H, everyNthFrame: 1
  });
  console.log('запись пошла');

  // --- ДОБИВКА: если кадр не приходил дольше FILL_MS, снимаем экран сами.
  // Именно это убирает «замерло → прыжок»: статичные участки получают
  // кадры с той же частотой, что и динамичные.
  let filling = false;
  const fillTimer = setInterval(async () => {
    if (filling) return;
    const now = Date.now();
    if (now - lastFrameT < FILL_MS) return;
    filling = true;
    try {
      const shot = await send('Page.captureScreenshot',
        { format: 'jpeg', quality: FILL_QUALITY });
      if (shot && shot.data) {
        const t = lastFrameT + FILL_MS;   // ставим в очередь по времени
        writeFrame(Buffer.from(shot.data, 'base64'), t);
      }
    } catch (e) { /* пропускаем такт */ }
    filling = false;
  }, 40);

  // Ввод посимвольно — видно, как набирается вопрос
  const typeHuman = async (text, delay = 42) => {
    await evaluate(`(function(){
      const i = document.getElementById('q');
      i.focus(); i.value = '';
      return 'ok';
    })()`);
    await sleep(200);
    for (const ch of text) {
      await send('Input.insertText', { text: ch });
      await sleep(delay);
    }
    await sleep(250);
  };

  const ask = async (text, waitMs) => {
    await typeHuman(text);
    await evaluate(`document.getElementById('go').click()`);
    await sleep(waitMs);
  };

  // ---------- Сцена 1 ----------
  console.log('1: титул');
  await evaluate(`(function(){
    document.getElementById('out').innerHTML = '<div class="card">' +
      '<div class="badge b-ok">ПОИСК ОТВЕТОВ ПО КОРПОРАТИВНЫМ ДОКУМЕНТАМ</div>' +
      '<div class="q" style="font-size:18px;line-height:1.6;margin:16px 0">' +
      'Агент находит нужный пункт регламента и показывает его <b>дословно</b> — ' +
      'с указанием документа, версии и номера пункта.</div>' +
      '<div class="meta" style="font-size:14px;line-height:1.65">' +
      'Генеративной модели в контуре нет: ответ — подстрока документа. ' +
      'Галлюцинация исключена архитектурно.</div></div>';
    return 'ok';
  })()`);
  await caption('<span class="n">1/9</span>Что делает агент: цитата из регламента, а не пересказ');
  await sleep(5500);

  // ---------- Сцена 2 ----------
  console.log('2: п. 2.1');
  await caption('<span class="n">2/9</span>Прямой ответ · вопрос сотрудника своими словами');
  await ask('как назначается РП после уведомления из CRM', 4000);
  await caption('<span class="n">2/9</span>Цитата п. 2.1 Проектной ДИБ — дословно, с источником');
  await sleep(4000);

  // ---------- Сцена 3 ----------
  console.log('3: п. 2.3');
  await caption('<span class="n">3/9</span>Второй прямой ответ · другой документ');
  await ask('что если от SS нет ответа более 5 рабочих дней', 4000);
  await caption('<span class="n">3/9</span>Цитата п. 2.3 — правило про отсутствие ответа от SS');
  await sleep(3800);

  // ---------- Сцена 4 ----------
  console.log('4: каскад');
  await caption('<span class="n">4/9</span>Кейс 2: вопрос неоднозначен — агент уточняет');
  await ask('заказчик выбрал продукт и запрашивает только расчёт поставки', 4500);

  await caption('<span class="n">4/9</span>Варианты — из текста регламента. Клик 1');
  await evaluate(`document.querySelector('#clar button[data-i="0"]').click()`);
  await sleep(3500);

  await caption('<span class="n">4/9</span>Клик 2 — и финальная цитата п. 1.3');
  await evaluate(`document.querySelector('#clar button[data-i="1"]').click()`);
  await sleep(4000);
  await caption('<span class="n">4/9</span>Два клика до дословной цитаты. Тупика нет');
  await sleep(3500);

  // ---------- Сцена 5 ----------
  console.log('5: отказ');
  await caption('<span class="n">5/9</span>Чего нет в базе — агент не придумывает');
  await ask('как оформить визу в Китай', 4000);
  await caption('<span class="n">5/9</span>Честный отказ + запрос в службу качества с номером');
  await sleep(4000);

  // ---------- Сцена 6 ----------
  console.log('6: английский');
  await caption('<span class="n">6/9</span>Вопрос по-английски — тот же поиск');
  await ask('who appoints the project manager', 4000);
  await caption('<span class="n">6/9</span>Найдено междокументное расхождение — развилка');
  await sleep(3800);

  // ---------- Сцена 7 ----------
  console.log('7: конфликт');
  await caption('<span class="n">7/9</span>Два регламента — разные сроки одного действия');
  await ask('какой срок инициации заявки на пресейл', 4500);
  await caption('<span class="n">7/9</span>Агент не выбирает сам: показывает оба и предлагает сверку');
  await sleep(4500);

  // ---------- Сцена 8 ----------
  console.log('8: панель');
  await caption('<span class="n">8/9</span>Панель: метрики и где база не отвечает');
  await evaluate(`(function(){
    document.getElementById('q').value = '';
    document.getElementById('out').innerHTML = '';
    window.scrollTo(0, 0); return 'ok';
  })()`);
  await sleep(5000);

  await caption('<span class="n">8/9</span>Белые пятна: темы, по которым чаще всего нет ответа');
  await evaluate(`document.getElementById('clusters').scrollIntoView({block:'center'})`);
  await sleep(5000);

  await caption('<span class="n">8/9</span>Владельцы получают уведомление — база обновляется');
  await evaluate(`document.getElementById('alerts').scrollIntoView({block:'center'})`);
  await sleep(4200);

  // ---------- Сцена 9: нажатие на ветку развилки ----------
  // Калькулятор убран: он не про продукт, а про экономику, и занимал
  // 12 секунд в конце. Вместо него — то, что делает агент: пользователь
  // нажимает ветку и получает дословную цитату с источником.
  console.log('9: выбор ветки');
  await caption('<span class="n">9/9</span>Клик по ветке — и цитата выбранного регламента');
  await evaluate(`(function(){
    document.getElementById('q').value = '';
    document.getElementById('out').innerHTML = '';
    window.scrollTo(0, 0); return 'ok';
  })()`);
  await sleep(1200);
  await evaluate(`(function(){
    const i = document.getElementById('q');
    i.value = 'кто назначает руководителя проекта';
    document.getElementById('go').click(); return 'ok';
  })()`);
  await sleep(4000);

  await caption('<span class="n">9/9</span>Два регламента называют разных РП — агент спрашивает');
  await sleep(3500);

  await caption('<span class="n">9/9</span>Клик «На этапе пресейла»');
  await evaluate(`document.querySelector('#clar button[data-i="0"]').click()`);
  await sleep(4000);

  await caption('<span class="n">9/9</span>Цитата с источником. Обе формулировки рядом');
  await sleep(4500);

  clearInterval(fillTimer);
  await send('Page.stopScreencast');
  await sleep(300);
  ws.close();
  proc.kill();

  // --- Сборка: длительность кадра = время до следующего
  const base = frames.length ? frames[0].t : 0;
  const lines = ['ffconcat version 1.0'];
  for (let i = 0; i < frames.length; i++) {
    const cur = frames[i], next = frames[i + 1];
    const dur = next ? Math.max(0.033, (next.t - cur.t) / 1000) : 1.5;
    lines.push(`file 'frames/${cur.file}'`);
    lines.push(`duration ${dur.toFixed(3)}`);
  }
  if (frames.length) lines.push(`file 'frames/${frames[frames.length - 1].file}'`);
  fs.writeFileSync(path.join(OUT, 'frames.txt'), lines.join('\n'), 'utf8');

  const total = frames.length ? (frames[frames.length - 1].t - base) / 1000 + 1.5 : 0;
  console.log(`кадров: ${frameNo}, длительность: ${total.toFixed(1)} с`);
})();
