# inference.py - Custom inference script for embedding operations
import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingInferenceHandler:
    """
    Custom inference handler that supports both encoding and similarity operations.
    """
    
    def __init__(self):
        self.model = None
        self.device = None
    
    def model_fn(self, model_dir):
        """
        Load the fine-tuned model from the model directory.
        """
        logger.info(f"Loading model from: {model_dir}")
        
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_dir, device=self.device)
        
        return self.model
    
    def input_fn(self, request_body, content_type='application/json'):
        """
        Parse input data for inference.
        """
        if content_type == 'application/json':
            logger.info(f"Inference request: {request_body}")
            data = json.loads(request_body)
        else:
            logger.warn(f"Wrong content type: {content_type}")
            raise ValueError(f"Unsupported content type: {content_type}")
        
        return data
    
    def predict_fn(self, data, model):
        """
        Perform inference based on the operation type.
        """
        operation = data.get('operation', 'encode')
        logger.info(f"Prediction input: {data}")
        if operation == 'encode':
            return self._encode_operation(data, model)
        elif operation == 'similarity':
            return self._similarity_operation(data, model)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _encode_operation(self, data, model):
        """
        Encode text inputs to embeddings.
        """
        inputs = data.get('inputs', [])
        if not inputs:
            raise ValueError("No inputs provided for encoding")
        
        # Get target dimension (default to 512 for Matryoshka)
        target_dim = data.get('dimension', 512)
        
        # Encode inputs
        embeddings = model.encode(
            inputs,
            batch_size=data.get('batch_size', 32),
            show_progress_bar=False,
            normalize_embeddings=data.get('normalize', True)
        )
        
        # Truncate to target dimension if specified
        if target_dim and target_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :target_dim]
        
        return {
            'embeddings': embeddings.tolist(),
            'dimension': embeddings.shape[1],
            'num_texts': len(inputs)
        }
    
    def _similarity_operation(self, data, model):
        """
        Calculate similarity between text pairs.
        """
        text1 = data.get('text1')
        text2 = data.get('text2')
        
        if not text1 or not text2:
            raise ValueError("Both text1 and text2 required for similarity")
        
        # Get target dimension
        target_dim = data.get('dimension', 512)
        
        # Encode both texts
        embeddings = model.encode([text1, text2], normalize_embeddings=True)
        
        # Truncate if needed
        if target_dim and target_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :target_dim]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        return {
            'similarity': float(similarity),
            'dimension': embeddings.shape[1],
            'text1_embedding': embeddings[0].tolist(),
            'text2_embedding': embeddings[1].tolist()
        }
    
    def output_fn(self, prediction, accept='application/json'):
        """
        Format the prediction output.
        """
        if accept == 'application/json':
            return json.dumps(prediction), 'application/json'
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

# Global handler instance
handler = EmbeddingInferenceHandler()

# SageMaker inference functions
def model_fn(model_dir):
    return handler.model_fn(model_dir)

def input_fn(request_body, content_type):
    return handler.input_fn(request_body, content_type)

def predict_fn(data, model):
    logger.info(f"predict_fn data: {data}")
    return handler.predict_fn(data, model)

def output_fn(prediction, accept):
    return handler.output_fn(prediction, accept)
