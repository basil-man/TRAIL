import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from llmlingua import PromptCompressor
from typing import List, Union, Tuple, Dict
import math
import torch
from torch.utils.data import DataLoader, Dataset
import argparse

from analyzer_utils import AnalyzerSingleton

# 配置日志
logging.basicConfig(level=logging.INFO)

# 解析命令行参数
parser = argparse.ArgumentParser(description='启动分析器服务')
parser.add_argument('--beta', type=float, default=1.02, help='重要性计算中的beta参数')
args = parser.parse_args()
print(f"Debug: beta 参数值为 {args.beta}")  # 添加此行用于调试

# 创建应用
app = FastAPI()

# 初始化分析器
analyzer = AnalyzerSingleton.get_instance()
device = next(analyzer.model.parameters()).device
logging.info(f"Analyzer model loaded on: {device}")
logging.info(f"Using beta={args.beta} for importance calculation")

# 请求模型
class SolutionRequest(BaseModel):
    solution_str: str

@app.post("/get_token_importance")
async def get_token_importance(req: SolutionRequest):
    try:
        probs,token_length = analyzer.get_token_importance(req.solution_str)
        # 使用beta计算长度感知的重要性
        length_factor = 1 / (token_length ** (args.beta-1))
        # importance = (sum(probs)/len(probs) )* length_factor
        importance = (1)* length_factor
        
        logging.info(f"Calculated importance: {importance}")
        return {"importance": float(importance) if isinstance(importance, (int, float)) else importance}
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload_model")
async def unload_model():
    try:
        # 将模型移到CPU，释放GPU内存
        logging.info("Unloading model from GPU memory...")
        device = next(analyzer.model.parameters()).device
        logging.info(f"Model currently on device: {device}")
        analyzer.model.to("cpu")
        torch.cuda.empty_cache()  # 清理GPU缓存
        return {"status": "success", "message": "Model unloaded from GPU"}
    except Exception as e:
        logging.error(f"Error unloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_model")
async def load_model():
    try:
        # 将模型加载回GPU
        logging.info("Loading model back to GPU...")
        analyzer.model.to("cuda")
        device = next(analyzer.model.parameters()).device
        logging.info(f"Model loaded to device: {device}")
        return {"status": "success", "message": "Model loaded to GPU"}
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)