from fastapi import FastAPI

app=FastAPI()

@app.get('/')
def home():
    return 'home'
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app",port=8000,host="127.0.0.1")