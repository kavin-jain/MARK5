import logging
import asyncio
import grpc
import numpy as np
import time
from concurrent import futures
import sys
import os

# Context path handling
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.serving import inference_pb2, inference_pb2_grpc
from core.prediction_engine import MARK5PredictionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Mark5.InferenceServer")

class OptimizedInferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        # The engine must be thread-safe or stateless
        self.predictor = MARK5PredictionEngine()
        logger.info("Mark V Prediction Engine Online - GIL Aware Mode")

    async def Predict(self, request, context):
        """
        Handled Asynchronously. 
        """
        start_ns = time.time_ns()
        try:
            # Direct feature access, minimizing dict creation overhead
            # Assumes request.features is a repeated float field
            features = np.array(request.features, dtype=np.float32)
            
            # Run inference (If this is heavy, it blocks the loop. 
            # Ideally, self.predictor.predict_single runs in a ThreadPool or ProcessPool)
            result = await asyncio.to_thread(self.predictor.predict_single, features)
            
            return inference_pb2.PredictionResponse(
                model_name=request.model_name,
                prediction=result.get('prediction', 0.0),
                confidence=result.get('confidence', 0.0),
                timestamp=str(time.time_ns()) # Use Nanoseconds
            )
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def BatchPredict(self, request, context):
        try:
            if not request.requests:
                return inference_pb2.BatchPredictionResponse(model_name=request.model_name)

            # CRITICAL OPTIMIZATION: Vectorization
            # Convert list of protos to numpy matrix in one go if possible, 
            # otherwise efficient list comp.
            # Using 'len' on repeated field is fast.
            rows = len(request.requests)
            cols = len(request.requests[0].features)
            
            # Pre-allocate numpy array to avoid memory fragmentation
            batch_matrix = np.zeros((rows, cols), dtype=np.float32)
            
            for i, req in enumerate(request.requests):
                batch_matrix[i, :] = req.features

            # Offload heavy calculation to thread to keep the event loop responsive for heartbeats
            result = await asyncio.to_thread(
                self.predictor.predict_batch_fast, 
                request.model_name, 
                batch_matrix
            )
            
            return inference_pb2.BatchPredictionResponse(
                model_name=request.model_name,
                predictions=result['predictions'],
                confidences=result['confidences'],
                timestamps=[time.time_ns()] * rows
            )
        except Exception as e:
            logger.error(f"Batch Error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

async def serve():
    # Use an AsyncIO server. This is vastly superior for throughput than ThreadPool servers in Python
    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(OptimizedInferenceServicer(), server)
    
    port = os.environ.get("GRPC_PORT", "50051")
    server.add_insecure_port(f'[::]:{port}')
    logger.info(f"Mark V Inference Node listening on {port}")
    
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
