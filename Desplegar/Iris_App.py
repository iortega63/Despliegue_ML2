import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
from typing import List
from pydantic import BaseModel

class ItemList(BaseModel):
    data_model: List[float]

loaded_model = pickle.load(open('./Desplegar/LRClassifier.pkl', 'rb'))

app = FastAPI()

@app.get('/')
async def index():
    return {"result": "ivonne"}

@app.get('/items/{variables}')
async def get_item(variables):
    return {"result": variables}

@app.post('/predict/')
async def predict(data: ItemList):
    result = loaded_model.predict(np.array(data.data_model).reshape(1,4))
    return result[0]    

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)