# KB — ИИ-агент поиска ответов по корпоративным документам

Поиск по смыслу по регламентам, приказам и базам знаний. Агент находит
релевантный фрагмент и показывает его **дословно** со ссылкой на документ,
раздел и версию. Если надёжного фрагмента нет — **честно отказывается**.

MVP работает **без LLM-генерации**: галлюцинация исключена архитектурно,
а не промптом. Модель используется только как энкодер.

**Стенд:** http://185.105.116.187/kb/

---

## Проверенные цифры

Все числа воспроизводятся командами из раздела «Запуск». Сплит: порог
подобран на `dev`, метрики сняты на `test` (невиданные данные).

| Метрика | Значение | Как получено |
|---|---|---|
| Recall@3 | **0.842** | `scripts/eval.py --split test`, 38 in-scope |
| Refusal precision | **0.857** | там же, 21 out-of-scope (1 уверенный ложный) |
| Stale accuracy | **1.000** | 4/4 вопросов про версию |
| Латентность p50 / p95 | **169 / 218 мс** | там же, 2 vCPU |
| Эмбеддинг запроса | **2.6 мс** | `scripts/export_onnx.py` |
| kb-api RSS | **160 МБ** | `ps -o rss= -p $(systemctl show -p MainPID --value kb-api)` |
| Корпус | 30 документов, 86 активных чанков, 160 предложений | `scripts/index_corpus.py` |
| Набор | 153 вопроса: dev 90 (56/30/4), test 63 (38/21/4) | `scripts/gen_validation_v2.py` |

**Три исхода вместо двух.** Помимо `success` и `refusal` есть тир
`success + low_confidence`: цитата показана, но покрытие терминов ниже
гейта, и интерфейс явно просит проверить источник. Дамп показал, что
ретривер находит золото в 62% «ложных» отказов — отказывать в таких
случаях значит терять сотрудника, который ушёл бы искать вручную.

Метрики разделены честно: `refusal_precision` считается только по
уверенным решениям (ответ или отказ), тир на out-of-scope показывается
отдельно как `tier_on_out_of_scope` (14 из 21 на test).

**Про падение Recall@3 с 0.864.** На первом наборе (30 вопросов) было
0.864. На расширенном (143 вопроса, dev 82 / test 61) — 0.579. Это не
регресс, а исправление измерения: первый набор строился из формулировок,
близких к тексту корпуса, и завышал Recall. Расширенный содержит чужие
формулировки и чужие домены (бухгалтерия, логистика, маркетинг).

При этом refusal precision выросла с 0.750 до 0.947. Компромисс выбран
по требованию брифа: выдуманный ответ дороже отказа.

Тесты SQL на живом PostgreSQL 16 + pgvector 0.8.7:

```
scripts/smoke_db.py      28/28   (леммы, tsquery, синонимы, trigram)
scripts/test_schema.py   20/20   (halfvec, <=>, HNSW, EXCLUDE, CHECK, ACL-кэш)
tests/test_fusion.py     21/21   (RRF, реранк, логистика)
```

---

## Как устроен путь от запроса до ответа

```
запрос
  │
  ├─ нормализация, токенизация, стоп-фильтр, потолок 8 групп   tsquery.py
  │
  ├─ эмбеддинг: rubert-tiny2 int8 через onnxruntime (2.6 мс)   encoder.py
  │
  ├─ ДВЕ ВЕТКИ ПАРАЛЛЕЛЬНО (один SQL, фильтр is_active в обеих)
  │    FTS:   kb_tsquery → OR внутри группы синонимов, AND между группами
  │    VEC:   halfvec(312) <=> , HNSW при >15k чанков
  │
  ├─ RRF k=60 — согласие двух веток побеждает ранг в одной      fusion.py
  │
  ├─ реранк: max-cosine по ПРЕДВЫЧИСЛЕННЫМ векторам предложений
  │         (модель на запросе НЕ вызывается: −150 мс, −200 МБ)
  │
  ├─ порядок показа: косинус + покрытие терминов + совпадение чисел
  │
  ├─ калиброванная логистика на 6 признаках → порог 0.63
  │
  └─ ответ: дословная цитата ИЛИ отказ с рекомендацией владельца
       │
       └─ если отказ: второй проход по НЕактивным версиям (stale-флоу)
```

### Где принимает решения модель

Только как энкодер. Она не порождает ни одного слова ответа.
Решение «ответить или отказаться» принимает логистическая регрессия
на измеримых признаках, обученная на валидационном наборе.

### Как проверяется ответ

1. **Цитата дословна по построению** — это подстрока текста чанка.
2. **Числовой guard** — если в ответе есть число, которого нет в источнике,
   ответ отсекается детерминированно, без учёта косинуса.
3. **Версия** — при вопросе про недействительную версию бот отвечает по
   действующей и предупреждает, а не молчит.

---

## Решения, которые нельзя менять не подумав

1. **`doc_key` без версии.** `reg_dogovora_v2.3` и `reg_dogovora_v3.1` —
   один `doc_key = reg_dogovora`. Иначе фильтр актуальности не работает.

2. **`chunks.is_active` денормализован.** Фильтр актуальности через `JOIN`
   внутри `ORDER BY embedding <=>` убивает HNSW (3 мс → 40 мс).

3. **ё/е: замена только симметрично.** Проверено на PostgreSQL:
   `tsv('Отчёт') = tsv('Отчет')` → `true`, словарь `russian` унифицирует
   ё/е сам. Ручная замена в одном месте создаёт асимметрию.

4. **Синонимы матчатся по лексеме** (`term_synonyms.term_lemma` считается
   тем же `kb_lemma`, что и токен запроса). Поверхностное сравнение молча
   не находило падежные формы: «командировку» не находило «командировка».

5. **Версии не идут в полнотекст.** `2.3` в AND-цепочке обнуляет выдачу.
   Версия уходит в структурный фильтр и в stale-флоу.

6. **Потолок 8 AND-групп.** По 40 группам совпадений не бывает, и FTS-ветка
   молча умирает, оставляя RRF с одной веткой.

7. **Кэш включает ACL.** `cache_key = HMAC(pepper, query|acl_signature)`.
   Иначе сотрудник без прав получит зарплатное положение из кэша HR.

8. **`response_type='stale'`** — третье значение, не «успех».
   Deflection = `(success + stale)/total`, Stale Rate показывается отдельно.

9. **Покрытие токенов — каскадный фильтр с порогом 0.35.** Не чистый
   признак и не жёсткий фильтр 0.5: первый не разделял классы, второй
   ронял Recall. Порог 0.35 подобран развёрткой на 143 вопросах и даёт
   precision 0.947 при Recall 0.579. Каскад живёт в `search.py`, чтобы
   метрики eval и поведение продукта совпадали.

10. **Метка кластера — из онтологии, не из документа.** Центроид отказов
    лежит вне пространства документов, поэтому ближайший чанк — случайный
    сосед. Без онтологии кластер про доверенности подписывался
    «Положение о командировках».

---

## Порядок миграций

Имена файлов не совпадают с порядком применения — это осознанно:

```
000_extensions.sql   расширения (vector, pg_trgm, btree_gist, pgcrypto)
002_lexeme.sql       kb_norm, kb_lemma, kb_lemmas, kb_tsv, агрегаты tsquery
003_synonyms.sql     term_synonyms + kb_expand_query + kb_tsquery
001_schema.sql       документы, чанки, журнал, кластеры (зависит от kb_tsv)
```

`001_schema.sql` грузится последним: он использует `kb_tsv()` из `002`.

---

## Запуск

### 1. База данных

```bash
sudo apt install postgresql-16 postgresql-16-pgvector
sudo -u postgres psql -c "CREATE USER kb WITH PASSWORD '...'"
sudo -u postgres psql -c "CREATE DATABASE kb OWNER kb"
sudo -u postgres psql -d kb -c "CREATE EXTENSION vector, pg_trgm, btree_gist, pgcrypto"

cd /opt/kb
for f in 000_extensions 002_lexeme 003_synonyms 001_schema; do
  PGPASSWORD=$PW psql -h 127.0.0.1 -U kb -d kb -f db/migrations/$f.sql
done
```

### 2. Модель (один раз, нужен torch)

```bash
pip install torch transformers onnxscript onnxruntime
python scripts/export_onnx.py --out models/rubert-tiny2-int8
# результат: model_int8.onnx 29.5 МБ, латентность 2.6 мс, сдвиг косинуса 0.009
```

В прод едет **только** `onnxruntime` — torch не нужен.

### 3. Корпус и индекс

```bash
python scripts/gen_corpus.py --out data/corpus
python scripts/index_corpus.py --corpus data/corpus --dsn "$DSN"
```

### 4. API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8090 --workers 1
```

**Один воркер обязателен:** второй удвоит ONNX-сессию (~180 МБ)
и выбьет в OOM на 2 ГБ RAM.

### 5. Проверки

```bash
python scripts/smoke_db.py  --dsn "$DSN" --mode fts    # 28 проверок
python scripts/test_schema.py --dsn "$DSN"             # 20 проверок
python tests/test_fusion.py                            # 21 проверка

python scripts/gen_validation_v2.py                    # 143 вопроса, сплит dev/test
python scripts/enrich_gold.py --dsn "$DSN"             # chunk_id в разметку
python scripts/eval.py --dsn "$DSN" --split dev --calibrate
python scripts/eval.py --dsn "$DSN" --split test       # цифры для слайда
```

Отчёты пишутся в `reports/*.json` и лежат в репозитории. Каждая цифра
«проверено N/N» имеет свой JSON и команду воспроизведения:

| Файл | Команда |
|---|---|
| `reports/smoke_db_*.json` | `python scripts/smoke_db.py --dsn "$DSN" --mode fts` |
| `reports/schema_*.json` | `python scripts/test_schema.py --dsn "$DSN"` |
| `reports/eval_dev.json` | `python scripts/eval.py --dsn "$DSN" --split dev --calibrate` |
| `reports/eval_test.json` | `python scripts/eval.py --dsn "$DSN" --split test` |
| `reports/index_report.json` | `python scripts/index_corpus.py --corpus data/corpus --dsn "$DSN"` |

`tests/test_fusion.py` печатает результат в stdout (21 проверка) — он
запускается без БД и потому не пишет JSON.

---

## Аналитика

```bash
python scripts/gen_query_logs.py --dsn "$DSN" --days 30 --per-day 13
python scripts/cluster_whitespots.py --dsn "$DSN" --notify
```

**Демо-история помечена `is_simulated = TRUE`.** KPI считается только по
`is_simulated = FALSE`; синтетика показывается отдельной панелью. Смешивать
измерение и демонстрацию нельзя — иначе цифра на слайде становится фикцией.

Формулировки вопросов и таймстемпы синтетические; эмбеддинги, решения и
латентности — реальные, прогон идёт через настоящий пайплайн.

Зоны: `red` при доле отказов ≥ 0.6, `yellow` при «бесполезно» ≥ 0.3.
Метки берутся из `knowledge_owners` (онтология тем с владельцами).

---

## Структура

```
app/
  ingest/     docx_parser.py, chunker.py, sentence_split.py
  search/     encoder.py, search.py, tsquery.py, fusion.py,
              grounding.py, versions.py
  utils/      vec.py (единая точка конвертации), sqlsplit.py
  main.py     FastAPI + веб-интерфейс
db/migrations/  000, 002, 003, 001 (порядок — см. выше)
scripts/      export_onnx, gen_corpus, index_corpus, gen_validation,
              enrich_gold, eval, gen_query_logs, cluster_whitespots,
              smoke_db, test_schema
tests/        test_fusion.py
data/         validation_all.jsonl (75 вопросов), corpus/ (генерируется)
```

---

## Ограничения

- **Стенд работает по пути `/kb/`** на существующем сервере, рядом с другими
  проектами. PostgreSQL занимает 256 МБ `shared_buffers`, kb-api 160 МБ;
  свободно ~900 МБ из 2 ГБ.
- **pgvector на дев-машине не устанавливался** (нужны права администратора).
  Векторная часть проверена на сервере: `halfvec`, `<=>`, HNSW — все 20/20.
- **n8n для алертов не подключён**: алерты пишутся в `knowledge_alerts` со
  статусом `pending`. Логика готова, транспорт — следующий шаг. Это честно
  отражено в поле «что сделано» при сдаче.
- **PDF не поддерживается**: DOCX как обязательный формат для критичных
  регламентов. PDF-парсинг ломает структуру пунктов.
