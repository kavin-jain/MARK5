import logging
import grpc
import numpy as np
import asyncio
from core.serving import inference_pb2, inference_pb2_grpc
from core.utils.fast_io import pack_features

logger = logging.getLogger("Mark5.InferenceClient")

class AsyncInferenceClient:
    """
    Non-blocking gRPC client optimized for event loops (FastAPI/main strategies).
    Maintains a persistent channel to avoid TCP handshake penalties.
    """
    def __init__(self, host="localhost", port=50051):
        self.target = f"{host}:{port}"
        self.channel = None
        self.stub = None
        # Optimization: Keepalive pings to prevent firewalls from killing idle connections
        self.options = [
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.min_reconnect_backoff_ms', 1000),
            ('grpc.max_reconnect_backoff_ms', 10000)
        ]

    async def connect(self):
        if self.channel:
            return
        
        logger.info(f"Establishing high-speed channel to {self.target}...")
        self.channel = grpc.aio.insecure_channel(self.target, options=self.options)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        
        # Warmup call
        try:
            await self.channel.channel_ready()
            logger.info("Channel Ready.")
        except grpc.aio.AioRpcError as e:
            logger.critical(f"Inference Server unreachable: {e}")

    async def predict_batch(self, features_batch: np.ndarray, model_name="default"):
        """
        Asynchronous batch prediction. 
        Expects features_batch as a numpy array for maximum speed.
        """
        if not self.stub:
            await self.connect()
            
        try:
            # OPTIMIZATION: If possible, we should modify the .proto to accept 'bytes' 
            # instead of repeated float. Assuming standard proto for now, but 
            # pre-converting to list is faster in C++ land of numpy than Python loops.
            
            # fast conversion
            requests = [
                inference_pb2.PredictionRequest(
                    model_name=model_name,
                    features=row.tolist() # Numpy's tolist() is highly optimized in C
                ) for row in features_batch
            ]
            
            request = inference_pb2.BatchPredictionRequest(
                model_name=model_name,
                requests=requests
            )
            
            # This await yields control, allowing the event loop to process market ticks
            response = await self.stub.BatchPredict(request)
            
            return {
                'predictions': np.array(response.predictions),
                'confidences': np.array(response.confidences),
                'timestamps': response.timestamps
            }
        except grpc.aio.AioRpcError as e:
            logger.error(f"Inference Failed: {e.code()} - {e.details()}")
            return None

    async def close(self):
        if self.channel:
            await self.channel.close()
            self.channel = None
