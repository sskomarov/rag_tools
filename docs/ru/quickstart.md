# Quickstart
Эта секция поможет быстро начать работу с RAG Tools и протестировать метрики LLM и RAG.

## Требования:
* Python >= 3.10

## Установка
Сначала клонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/sskomarov/rag_tools/
cd rag_tools
pip install -e ".[dev]"
```

## Быстрая проверка работы API

* Запустите API сервер:
```bash
uvicorn rag_tools.main:app --reload
```
* Проверьте доступность API с помощью запроса к эндпоинту `/health`:
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
Ожидаемый ответ:
```json
{
    "status": "ok"
}
```