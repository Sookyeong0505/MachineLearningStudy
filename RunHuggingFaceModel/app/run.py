import uvicorn


if __name__ == "__main__":
   uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True) # http://127.0.0.1:8000/docs