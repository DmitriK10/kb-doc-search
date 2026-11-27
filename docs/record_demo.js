// Запись демо-видео ОДНИМ ФРАГМЕНТОМ через CDP Page.startScreencast.
//
// Регламент запрещает монтаж и склейку. CDP отдаёт кадры по мере отрисовки
// в реальном времени — это честная запись экрана, не сборка из скриншотов.
//
// Частота кадров: Chrome присылает кадр только при ИЗМЕНЕНИИ экрана.
// За ~3 минуты набегает 300-600 кадров вместо 5400 при 30 fps. Поэтому
// пишем ТАЙМСТЕМП каждого кадра и собираем видео с реальной длительностью
// через concat-демультиплексор ffmpeg. Кадры не переставляются и не
// вырезаются — это запись, а не монтаж.
//
// Сессия записи: kb_sid подставляется ДО загрузки страницы, чтобы после
// записи помечать её служебной и не засорять панель.
//
// Сценарий (9 сцен, ~3 минуты).
// Запуск: node record_demo.js <outdir> <calcHtml>

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const WebSocket = require('ws');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const BASE = 'https://vds3095790.my-ihor.ru/kb/';
const OUT = process.argv[2] || '.';
const CALC = process.argv[3] || '';
const SID = 'video-demo-2711';
const W = 1600, H = 1000;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(d)); })
    .on('error', rej));

(async () => {
  const port = 9411;
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
  for (let i = 0; i < 60; i++) {
    try { targets = JSON.parse(await get(`http://127.0.0.1:${port}/json`)); break; }
    catch (e) { await sleep(250); }
  }
  if (!targets) { console.error('chrome не поднялся'); proc.kill(); process.exit(1); }

  const page = targets.find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  let frameNo = 0;
  const frames = [];

  ws.on('message', raw => {
    const m = JSON.parse(raw);
    if (m.id && pending.has(m.id)) {
      pending.get(m.id)(m.result); pending.delete(m.id); return;
    }
    if (m.method === 'Page.screencastFrame') {
      const t = m.params.metadata && m.params.metadata.timestamp
        ? m.params.metadata.timestamp * 1000 : Date.now();
      const fname = `f${String(frameNo++).padStart(5, '0')}.jpg`;
      fs.writeFileSync(path.join(framesDir, fname), Buffer.from(m.params.data, 'base64'));
      frames.push({ file: fname, t });
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

  // Что реально на экране: нужно, чтобы лог подтверждал сцену, а не
  // только факт отправки запроса.
  const screen = async () => {
    const t = await evaluate(
      `((document.getElementById('out')||{}).innerText||'').replace(/\\s+/g,' ').slice(0,120)`);
    return t || '(пусто)';
  };

  await new Promise(r => ws.on('open', r));
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride',
    { width: W, height: H, deviceScaleFactor: 1, mobile: false });

  // Фиксированный session-id до загрузки страницы.
  await send('Page.addScriptToEvaluateOnNewDocument', {
    source: `try { localStorage.setItem('kb_sid', ${JSON.stringify(SID)}); } catch(e){}`
  });

  const injectCaption = async () => {
    await evaluate(`(function(){
      if (document.getElementById('__cap')) return 'exists';
      const st = document.createElement('style');
      st.textContent = '#__cap{position:fixed;left:0;right:0;bottom:0;z-index:99999;' +
        'background:linear-gradient(180deg,rgba(15,17,21,0),rgba(15,17,21,.97) 34%);' +
        'padding:30px 44px 20px;font-family:\\'Segoe UI\\',Arial,sans-serif;' +
        'font-size:20px;color:#e6e9ef;line-height:1.45;pointer-events:none;text-align:center}' +
        '#__cap b{color:#8ab4f8}#__cap .n{color:#7ee787;font-weight:700;margin-right:10px}';
      document.head.appendChild(st);
      const d = document.createElement('div');
      d.id = '__cap'; d.innerHTML = '';
      document.body.appendChild(d);
      return 'ok';
    })()`);
  };
  const caption = async html => {
    await evaluate(`(function(){
      const d = document.getElementById('__cap');
      if (d) d.innerHTML = ${JSON.stringify(html)};
      return 'ok';
    })()`);
  };

  console.log('открываю стенд');
  await send('Page.navigate', { url: BASE });
  await sleep(6500);
  await injectCaption();

  // ЗАПИСЬ НАЧИНАЕТСЯ ЗДЕСЬ. До этой строки кадры не пишутся — иначе
  // в кадр попадёт загрузка и пустой экран.
  await send('Page.startScreencast', {
    format: 'jpeg', quality: 92, maxWidth: W, maxHeight: H, everyNthFrame: 1
  });
  console.log('запись пошла');

  // Снимок панели на момент загрузки: что реально в кадре.
  const statsText = await evaluate(
    `(document.getElementById('stats')||{}).innerText||''`);
  console.log('--- ПАНЕЛЬ В КАДРЕ ---');
  console.log(statsText.replace(/\n/g, ' | ').slice(0, 260));

  // ---------- Сцена 1: титул ----------
  console.log('1: титул');
  await evaluate(`(function(){
    const out = document.getElementById('out');
    out.innerHTML = '<div class="card">' +
      '<div class="badge b-ok">ПОИСК ОТВЕТОВ ПО КОРПОРАТИВНЫМ ДОКУМЕНТАМ</div>' +
      '<div class="q" style="font-size:18px;line-height:1.6;margin:16px 0">' +
      'Агент находит нужный пункт регламента и показывает его <b>дословно</b> — ' +
      'с указанием документа, версии и номера пункта.</div>' +
      '<div class="meta" style="font-size:14px;line-height:1.65">' +
      'Генеративной модели в контуре нет: ответ — подстрока документа. ' +
      'Галлюцинация исключена архитектурно. Если фрагмент не найден, ' +
      'агент честно отказывает и создаёт запрос владельцу базы знаний.' +
      '</div></div>';
    return 'ok';
  })()`);
  await caption('<span class="n">1/9</span>Что делает агент: цитата из регламента, а не пересказ');
  await sleep(11000);

  // ---------- Сцена 2 ----------
  console.log('2: п. 2.1');
  await caption('<span class="n">2/9</span>Прямой ответ · вопрос сотрудника своими словами');
  await evaluate(`(function(){
    const i = document.getElementById('q');
    i.value = 'как назначается РП после уведомления из CRM';
    document.getElementById('go').click(); return 'ok';
  })()`);
  await sleep(5500);
  console.log('   экран:', await screen());
  await caption('<span class="n">2/9</span>Цитата п. 2.1 Проектной ДИБ — дословно, с источником');
  await sleep(7500);

  // ---------- Сцена 3 ----------
  console.log('3: п. 2.3');
  await caption('<span class="n">3/9</span>Второй прямой ответ · другой документ');
  await evaluate(`(function(){
    const i = document.getElementById('q');
    i.value = 'что если от SS нет ответа более 5 рабочих дней';
    document.getElementById('go').click(); return 'ok';
  })()`);
  await sleep(5500);
  console.log('   экран:', await screen());
  await caption('<span class="n">3/9</span>Цитата п. 2.3 — правило про отсутствие ответа от SS');
  await sleep(7500);

  // ---------- Сцена 4: каскад ----------
  console.log('4: каскад');
  await caption('<span class="n">4/9</span>Кейс 2: вопрос неоднозначен — агент уточняет, а не угадывает');
  await evaluate(`(function(){
    const i = document.getElementById('q');
    i.value = 'заказчик выбрал продукт и запрашивает только расчёт поставки';
    document.getElementById('go').click(); return 'ok';
  })()`);
  await sleep(6500);

  await caption('<span class="n">4/9</span>Варианты — из текста регламента. Клик 1');
  await evaluate(`(function(){
    const b = document.querySelector('#clar button[data-i="0"]');
    if (!b) return 'нет кнопки'; b.click(); return 'clicked';
  })()`);
  await sleep(5500);

  await caption('<span class="n">4/9</span>Клик 2 — и финальная цитата п. 1.3');
  await evaluate(`(function(){
    const b = document.querySelector('#clar button[data-i="1"]');
    if (!b) return 'нет кнопки'; b.click(); return 'clicked';
  })()`);
  await sleep(5500);
  console.log('   экран:', await screen());
  await caption('<span class="n">4/9</span>Два клика до дословной цитаты. Тупика нет');
  await sleep(7000);

  // ---------- Сцена 5 ----------
  console.log('5: отказ');
  await caption('<span class="n">5/9</span>Чего нет в базе — агент не придумывает');
  await evaluate(`(function(){
    const i = document.getElementById('q');
    i.value = 'как оформить визу в Китай';
    document.getElementById('go').click(); return 'ok';
  })()`);
  await sleep(5500);
  console.log('   экран:', await screen());
  await caption('<span class="n">5/9</span>Честный отказ + запрос в службу качества с номером');
  await sleep(7500);

  // ---------- Сцена 6 ----------
  console.log('6: английский');
  await caption('<span class="n">6/9</span>Вопрос по-английски — тот же поиск, без перевода модели');
  await evaluate(`(function(){
    const i = document.getElementById('q');
    i.value = 'who appoints the project manager';
    document.getElementById('go').click(); return 'ok';
  })()`);
  await sleep(5500);
  console.log('   экран:', await screen());
  await caption('<span class="n">6/9</span>Найдено междокументное расхождение — развилка');
  await sleep(7000);

  // ---------- Сцена 7 ----------
  console.log('7: конфликт');
  await caption('<span class="n">7/9</span>Два регламента — разные сроки одного действия');
  await evaluate(`(function(){
    const i = document.getElementById('q');
    i.value = 'какой срок инициации заявки на пресейл';
    document.getElementById('go').click(); return 'ok';
  })()`);
  await sleep(6000);
  console.log('   экран:', await screen());
  await caption('<span class="n">7/9</span>Агент не выбирает сам: показывает оба и предлагает сверку');
  await sleep(8000);

  // ---------- Сцена 8: панель ----------
  console.log('8: панель');
  await caption('<span class="n">8/9</span>Панель: метрики и где база не отвечает');
  await evaluate(`(function(){
    document.getElementById('q').value = '';
    document.getElementById('out').innerHTML = '';
    window.scrollTo(0, 0); return 'ok';
  })()`);
  await sleep(8000);

  console.log('8: кластеры');
  await caption('<span class="n">8/9</span>Белые пятна: темы, по которым чаще всего нет ответа');
  await evaluate(`(function(){
    const el = document.getElementById('clusters');
    if (el) el.scrollIntoView({block:'center'}); return 'ok';
  })()`);
  await sleep(8000);

  console.log('8: алерты');
  await caption('<span class="n">8/9</span>Владельцы получают уведомление — база обновляется');
  await evaluate(`(function(){
    const el = document.getElementById('alerts');
    if (el) el.scrollIntoView({block:'center'}); return 'ok';
  })()`);
  await sleep(7000);

  // ---------- Сцена 9: калькулятор ----------
  if (CALC) {
    console.log('9: калькулятор');
    await send('Page.navigate', { url: 'file:///' + CALC.replace(/\\/g, '/') });
    await sleep(3500);
    // Зум НЕ уменьшаем: кадр 1484x1052 целиком вписывается в 1600x1000
    // по ширине и по высоте (max-height в CSS). Проверено по пикселям:
    // контент доходит до 99% ширины исходного рендера, обрезки нет.
    await evaluate(`(function(){
      const img = document.querySelector('img');
      if (img) { img.style.maxHeight = 'calc(100vh - 66px)'; }
      return 'fit';
    })()`);
    await sleep(800);
    await caption('<span class="n">9/9</span>Экономика: 0 ₽ за токены — энкодер локальный, генерации нет');
    await sleep(8000);
    await caption('<span class="n">9/9</span>Облачная LLM ≈ 0.029 ₽/запрос ≈ 1400×. Главное — риск выхода текста за контур');
    await sleep(8500);
  }

  await send('Page.stopScreencast');
  await sleep(300);
  ws.close();
  proc.kill();

  const t0 = frames.length ? frames[0].t : 0;
  const lines = ['ffconcat version 1.0'];
  for (let i = 0; i < frames.length; i++) {
    const cur = frames[i], next = frames[i + 1];
    const dur = next ? Math.max(0.03, (next.t - cur.t) / 1000) : 1.5;
    lines.push(`file 'frames/${cur.file}'`);
    lines.push(`duration ${dur.toFixed(3)}`);
  }
  if (frames.length) lines.push(`file 'frames/${frames[frames.length - 1].file}'`);
  fs.writeFileSync(path.join(OUT, 'frames.txt'), lines.join('\n'), 'utf8');

  const total = frames.length ? (frames[frames.length - 1].t - t0) / 1000 + 1.5 : 0;
  console.log(`кадров: ${frameNo}, длительность: ${total.toFixed(1)} с`);
})();
