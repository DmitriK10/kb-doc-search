// Запись демо-видео ОДНИМ ФРАГМЕНТОМ с реального экрана (gdigrab).
//
// ПОЧЕМУ ПЕРЕПИСАНО. Первая версия шла через Page.startScreencast в
// headless-браузере. Chrome присылает кадр только при ИЗМЕНЕНИИ экрана,
// поэтому статичный участок держится одним кадром, а потом контент
// прыгает мгновенно. Технически это одна запись (кадры не переставлялись
// и не вырезались), но визуально читается как склейка: жюри видит
// «замерло → прыжок».
//
// Здесь пишем ЖИВОЙ экран через gdigrab: 30 кадров в секунду всегда,
// видно курсор, видно как текст набирается посимвольно, видно как
// страница отвечает. Никакой интерполяции и монтажа — честный захват.
//
// Сценарий тот же, что в отчёте: девять сцен, ~2:10.
// Запуск: node record_screen.js <outdir> <calcHtml>

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn, execSync } = require('child_process');
const WebSocket = require('ws');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const FFMPEG = 'C:\\Users\\ADATA\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-9.1-full_build\\bin\\ffmpeg.exe';
const BASE = 'https://vds3095790.my-ihor.ru/kb/';
const OUT = process.argv[2] || '.';
const CALC = process.argv[3] || '';

// Окно 1600x1000 в левом верхнем углу: ровно тот регион и захватываем.
const W = 1600, H = 1000;
const X = 0, Y = 0;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(d)); })
    .on('error', rej));

(async () => {
  const port = 9421;
  const videoFile = path.join(OUT, 'demo_screen.mp4');

  // --- 1. Chrome с РЕАЛЬНЫМ окном ---
  const proc = spawn(CHROME, [
    `--remote-debugging-port=${port}`,
    `--window-size=${W},${H}`,
    `--window-position=${X},${Y}`,
    '--no-first-run', '--no-default-browser-check',
    '--disable-infobars', '--hide-crash-restore-bubble',
    '--user-data-dir=' + path.join(OUT, 'profile'),
    'about:blank'
  ], { stdio: 'ignore', detached: false });

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

  ws.on('message', raw => {
    const m = JSON.parse(raw);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
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

  // Фиксированный session-id: триггер пометит служебным ДО вставки.
  await send('Page.addScriptToEvaluateOnNewDocument', {
    source: `try { localStorage.setItem('kb_sid', 'video-demo-2711'); } catch(e){}`
  });

  // Окно на передний план
  try {
    execSync(`powershell -Command "$w = New-Object -ComObject WScript.Shell; $w.AppActivate('Chrome')"`,
      { stdio: 'ignore' });
  } catch (e) { /* не критично */ }

  console.log('открываю стенд');
  await send('Page.navigate', { url: BASE });
  await sleep(7000);

  // --- 2. Стартуем захват экрана ---
  const ff = spawn(FFMPEG, [
    '-y',
    '-f', 'gdigrab',
    '-framerate', '30',
    '-offset_x', String(X), '-offset_y', String(Y),
    '-video_size', `${W}x${H}`,
    '-i', 'desktop',
    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
    '-pix_fmt', 'yuv420p',
    videoFile
  ], { stdio: ['pipe', 'ignore', 'ignore'] });

  await sleep(1500);   // дать ffmpeg выйти на стабильный захват
  console.log('запись пошла (gdigrab 30 fps)');

  // --- 3. Подписи сцен ---
  await evaluate(`(function(){
    if (document.getElementById('__cap')) return 'exists';
    const st = document.createElement('style');
    st.textContent = '#__cap{position:fixed;left:0;right:0;bottom:0;z-index:99999;' +
      'background:linear-gradient(180deg,rgba(15,17,21,0),rgba(15,17,21,.97) 34%);' +
      'padding:30px 44px 20px;font-family:\\'Segoe UI\\',Arial,sans-serif;' +
      'font-size:20px;color:#e6e9ef;line-height:1.45;pointer-events:none;text-align:center}' +
      '#__cap b{color:#8ab4f4}#__cap .n{color:#7ee787;font-weight:700;margin-right:10px}';
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

  // Ввод как у человека: посимвольно через Input.insertText
  const typeHuman = async (text, delay = 45) => {
    await evaluate(`(function(){
      const i = document.getElementById('q');
      i.focus(); i.value = '';
      return 'ok';
    })()`);
    await sleep(250);
    for (const ch of text) {
      await send('Input.insertText', { text: ch });
      await sleep(delay);
    }
    await sleep(300);
  };

  // Клик мышью по координатам элемента (видно курсор)
  const clickAt = async selector => {
    const box = await evaluate(`(function(){
      const e = document.querySelector(${JSON.stringify(selector)});
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return JSON.stringify({x: Math.round(r.left + r.width/2), y: Math.round(r.top + r.height/2)});
    })()`);
    if (!box) return false;
    const { x, y } = JSON.parse(box);
    // Двигаем мышь, потом жмём — видно движение курсора
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
    await sleep(120);
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
    await sleep(80);
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
    return true;
  };

  const ask = async (text, waitMs) => {
    await typeHuman(text);
    await clickAt('#go');
    await sleep(waitMs);
  };

  // ---------- Сцена 1: титул ----------
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
  await sleep(6500);

  // ---------- Сцена 2 ----------
  console.log('2: п. 2.1');
  await caption('<span class="n">2/9</span>Прямой ответ · вопрос сотрудника своими словами');
  await ask('как назначается РП после уведомления из CRM', 4500);
  await caption('<span class="n">2/9</span>Цитата п. 2.1 Проектной ДИБ — дословно, с источником');
  await sleep(5000);

  // ---------- Сцена 3 ----------
  console.log('3: п. 2.3');
  await caption('<span class="n">3/9</span>Второй прямой ответ · другой документ');
  await ask('что если от SS нет ответа более 5 рабочих дней', 4500);
  await caption('<span class="n">3/9</span>Цитата п. 2.3 — правило про отсутствие ответа от SS');
  await sleep(4500);

  // ---------- Сцена 4: каскад ----------
  console.log('4: каскад');
  await caption('<span class="n">4/9</span>Кейс 2: вопрос неоднозначен — агент уточняет');
  await ask('заказчик выбрал продукт и запрашивает только расчёт поставки', 5000);

  await caption('<span class="n">4/9</span>Варианты — из текста регламента. Клик 1');
  await clickAt('#clar button[data-i="0"]');
  await sleep(4000);

  await caption('<span class="n">4/9</span>Клик 2 — и финальная цитата п. 1.3');
  await clickAt('#clar button[data-i="1"]');
  await sleep(4500);
  await caption('<span class="n">4/9</span>Два клика до дословной цитаты. Тупика нет');
  await sleep(4500);

  // ---------- Сцена 5 ----------
  console.log('5: отказ');
  await caption('<span class="n">5/9</span>Чего нет в базе — агент не придумывает');
  await ask('как оформить визу в Китай', 4500);
  await caption('<span class="n">5/9</span>Честный отказ + запрос в службу качества с номером');
  await sleep(5000);

  // ---------- Сцена 6 ----------
  console.log('6: английский');
  await caption('<span class="n">6/9</span>Вопрос по-английски — тот же поиск');
  await ask('who appoints the project manager', 4500);
  await caption('<span class="n">6/9</span>Найдено междокументное расхождение — развилка');
  await sleep(4500);

  // ---------- Сцена 7 ----------
  console.log('7: конфликт');
  await caption('<span class="n">7/9</span>Два регламента — разные сроки одного действия');
  await ask('какой срок инициации заявки на пресейл', 5000);
  await caption('<span class="n">7/9</span>Агент не выбирает сам: показывает оба и предлагает сверку');
  await sleep(5500);

  // ---------- Сцена 8: панель ----------
  console.log('8: панель');
  await caption('<span class="n">8/9</span>Панель: метрики и где база не отвечает');
  await evaluate(`(function(){
    document.getElementById('q').value = '';
    document.getElementById('out').innerHTML = '';
    window.scrollTo(0, 0); return 'ok';
  })()`);
  await sleep(6000);

  await caption('<span class="n">8/9</span>Белые пятна: темы, по которым чаще всего нет ответа');
  await evaluate(`(function(){
    const el = document.getElementById('clusters');
    if (el) el.scrollIntoView({block:'center'}); return 'ok';
  })()`);
  await sleep(6000);

  await caption('<span class="n">8/9</span>Владельцы получают уведомление — база обновляется');
  await evaluate(`(function(){
    const el = document.getElementById('alerts');
    if (el) el.scrollIntoView({block:'center'}); return 'ok';
  })()`);
  await sleep(5000);

  // ---------- Сцена 9: калькулятор ----------
  if (CALC) {
    console.log('9: калькулятор');
    await send('Page.navigate', { url: 'file:///' + CALC.replace(/\\/g, '/') });
    await sleep(4000);
    await caption('<span class="n">9/9</span>Экономика: 0 ₽ за токены — энкодер локальный');
    await sleep(6000);
    await caption('<span class="n">9/9</span>Облачная LLM ≈ 0.029 ₽/запрос ≈ 1400×. Главное — риск выхода текста за контур');
    await sleep(5500);
  }

  // --- 4. Останавливаем захват ---
  ff.stdin.write('q');
  await new Promise(r => ff.on('close', r));
  ws.close();
  proc.kill();

  const sz = fs.existsSync(videoFile) ? fs.statSync(videoFile).size : 0;
  console.log(`готово: ${videoFile} (${(sz / 1024 / 1024).toFixed(1)} MB)`);
})();
