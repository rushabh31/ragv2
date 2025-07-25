import asyncio
import base64
import logging
import os
import requests
from typing import Dict, Any, List, Optional, Union
from vertexai.generative_models import GenerativeModel, Content, Part, Image
from vertexai.language_models import TextEmbeddingModel
import vertexai
from credentials import Credentials

logger = logging.getLogger(__name__)

class TokenCredentials(Credentials):
    def __init__(self, token: str):
        super().__init__()
        self.token = token
        
    def refresh(self, request):
        pass
        
    def before_request(self, request, method, url, headers):
        headers["Authorization"] = f"Bearer {self.token}"

class VertexGenAI:
    """Wrapper for Google Cloud VertexAI generative AI services."""
    
    def __init__(self, 
                 model_name: str = "gemini-1.5-pro-002",
                 embedding_model: str = "text-embedding-004"):
        """Initialize the VertexAI client.
        
        Args:
            model_name: Name of the Vertex generative model to use
            embedding_model: Name of the text embedding model to use
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self._gen_model = None
        self._embedding_model = None
        self.provider = "vertex"
        self.vertex_config = None
    
    def get_coin_token(self):
        """Get authentication token for Vertex AI directly from environment variables."""
        url = os.environ.get("COIN_CONSUMER_ENDPOINT_URL") # URL from .env
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        body = {
            "grant_type": "client_credentials",
            "scope": os.environ.get("COIN_CONSUMER_SCOPE"), # Scope from .env
            "client_id": os.environ.get("COIN_CONSUMER_CLIENT_ID"), # Client ID from .env
            "client_secret": os.environ.get("COIN_CONSUMER_CLIENT_SECRET"), # Client Secret from .env
        }
        
        print("Requesting API token") # Debug step 1
        
        response = requests.post(url, headers=headers, data=body, verify=False, timeout=10)
        
        if response.status_code == 200:
            print("Token received successfully") # Debug step 2
            return response.json().get('access_token', None)
        else:
            print(f"Failed to get token! status code: {response.status_code}, Response: {response.text}")
            return None
    
    async def _init_gen_model(self):
        """Initialize generative model lazily."""
        if self._gen_model is None:
            try:
                # Use API key as OAuth token for credentials
                token = self.get_coin_token()
                if not token:
                    raise ValueError("Failed to retrieve token for Vertex AI client.")
                credentials = TokenCredentials(token)
                
                # Initialize Vertex AI with provided credentials and project details from environment variables
                project_id = os.environ.get("PROJECT_ID")
                api_endpoint = os.environ.get("VERTEXAI_API_ENDPOINT")
                api_transport = os.environ.get("VERTEXAI_API_TRANSPORT", "grpc")
                
                vertexai.init(
                    project=project_id,
                    api_transport=api_transport,
                    api_endpoint=api_endpoint,  # uses R2D2 UAT
                    credentials=credentials,
                )
                
                # Initialize the generator for Vertex AI
                self._gen_model = GenerativeModel(self.model_name)
                logger.info(f"Initialized generative model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize generative model: {str(e)}")
                raise
    
    async def _init_embedding_model(self):
        """Initialize embedding model lazily."""
        if self._embedding_model is None:
            try:
                self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
                logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {str(e)}")
                raise
    
    async def generate_content(self, 
                              prompt: str, 
                              image_data: Optional[str] = None,
                              chat_history: Optional[List[Dict[str, str]]] = None,
                              generation_config: Optional[Dict[str, Any]] = None) -> Any:
        """Generate content from text and optional image input.
        
        Args:
            prompt: Text prompt
            image_data: Optional base64 encoded image data
            chat_history: Optional chat history in format [{"role": "user", "content": "..."}, ...]
            generation_config: Optional generation configuration
            
        Returns:
            Model response object
        """
        await self._init_gen_model()
        
        try:
            contents = []
            
            # Add chat history if provided
            if chat_history:
                for message in chat_history:
                    role = message["role"]
                    if role not in ["user", "assistant", "system"]:
                        role = "user"  # Default to user for unknown roles
                    
                    contents.append(
                        Content(
                            role=role,
                            parts=[Part.from_text(message["content"])]
                        )
                    )
            
            # Create content with text and optional image
            parts = []
            if prompt:
                parts.append(Part.from_text(prompt))
            
            # Add image if provided
            if image_data:
                image = Image.from_bytes(base64.b64decode(image_data))
                parts.append(Part.from_image(image))
            
            # Add the current message
            contents.append(Content(role="user", parts=parts))
            
            # Set default generation config if not provided
            if generation_config is None:
                generation_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                    "top_p": 1.0,
                    "top_k": 40
                }
            
            # Make the API call
            response = await asyncio.to_thread(
                lambda: self._gen_model.generate_content(
                    contents,
                    generation_config=generation_config
                )
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise
    
    async def get_embeddings(self, texts: Union[str, List[str]], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for one or more text strings.
        
        Args:
            texts: A single text string or a list of text strings
            batch_size: Batch size for embedding requests
            
        Returns:
            A list of embedding vectors
        """
        await self._init_embedding_model()
        
        try:
            # Handle single string case
            if isinstance(texts, str):
                texts = [texts]
            
            all_embeddings = []
            
            # Process in batches to avoid API limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Get embeddings for the batch
                embeddings = await asyncio.to_thread(
                    lambda: self._embedding_model.get_embeddings(batch)
                )
                
                # Extract the embedding vectors
                batch_embeddings = [embedding.values for embedding in embeddings]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
            
    async def parse_text_from_image(self, base64_encoded: str, prompt: str) -> str:
        """Parse text from an image using Vertex AI generative model.
        
        Args:
            base64_encoded: Base64 encoded image data
            prompt: Text prompt for the model
            
        Returns:
            Extracted text in markdown format
        """
        await self._init_gen_model()
        
        try:
            # Create text part
            text_part = Part.from_text(prompt)
            
            # Decode base64 image data and create image part
            image_data = base64.b64decode(base64_encoded)
            image_part = Part.from_image(Image.from_bytes(image_data))
            
            # Combine parts into a Content object
            content = Content(parts=[text_part, image_part], role="user")
            
            # Generate content using the model
            response = await asyncio.to_thread(
                lambda: self._gen_model.generate_content(content)
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to parse text from image: {str(e)}")
            raise
