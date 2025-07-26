import logging
import json
import os
from typing import Dict, Any, List, Optional
import jinja2

from src.rag.chatbot.generators.base_generator import BaseGenerator
from src.rag.core.interfaces.base import Document
from src.models.generation.model_factory import GenerationModelFactory
from src.rag.core.exceptions.exceptions import GenerationError
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class VertexGenerator(BaseGenerator):
    """Generator that uses Google Vertex AI for response generation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Vertex AI generator with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: Name of the Vertex AI model
                - temperature: Temperature for sampling
                - max_tokens: Maximum tokens in response
                - prompt_template: Path to Jinja2 prompt template
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "gemini-1.5-pro-002")
        self.template_path = self.config.get("prompt_template", "./templates/rag_prompt.jinja2")
        self._generation_model = None
        self._template = None
        
        # Load generation configuration from system config
        config_manager = ConfigManager()
        system_config = config_manager.get_config("generation")
        if system_config:
            provider = system_config.get("provider", "vertex")
            # Map vertex to vertex_ai for the GenerationModelFactory
            self.generation_provider = "vertex_ai" if provider == "vertex" else provider
            self.generation_config = system_config.get("config", {})
        else:
            # Default to vertex_ai if no generation config found
            self.generation_provider = "vertex_ai"
            self.generation_config = {}
        
    async def _init_components(self):
        """Initialize generation model and prompt template lazily."""
        if self._generation_model is None:
            try:
                # Create generation model using factory based on configuration
                self._generation_model = GenerationModelFactory.create_model(
                    provider=self.generation_provider,
                    model_name=self.model_name,
                    **self.generation_config
                )
                
                # Validate authentication
                is_valid = await self._generation_model.validate_authentication()
                if is_valid:
                    logger.info(f"Initialized {self.generation_provider} generation model: {self.model_name}")
                else:
                    logger.warning(f"{self.generation_provider} generation model authentication validation failed")
            except Exception as e:
                logger.error(f"Failed to initialize generation model: {str(e)}")
                raise
        
        if self._template is None:
            try:
                # Load template from file
                template_loader = jinja2.FileSystemLoader("./")
                template_env = jinja2.Environment(loader=template_loader)
                self._template = template_env.get_template(self.template_path)
                logger.info(f"Loaded prompt template from: {self.template_path}")
            except Exception as e:
                # Use a default template if loading fails
                logger.warning(f"Failed to load template from {self.template_path}: {str(e)}")
                logger.info("Using default in-memory template")
                template_str = """
                Answer the following question based on the provided context.
                
                Context:
                {% for doc in documents %}
                ---
                {{ doc.content }}
                ---
                {% endfor %}
                
                {% if conversation_history %}
                Previous conversation:
                {% for message in conversation_history %}
                {{ message.role }}: {{ message.content }}
                {% endfor %}
                {% endif %}
                
                Question: {{ query }}
                
                Instructions:
                1. Answer the question based only on the provided context.
                2. If the context doesn't contain the answer, say "I don't have enough information to answer that question."
                3. Provide specific references to the context when possible.
                4. Be concise and accurate.
                
                Answer:
                """
                self._template = jinja2.Template(template_str)
    
    async def _generate_response(self, 
                               query: str, 
                               documents: List[Document], 
                               conversation_history: Optional[List[Dict[str, str]]] = None, 
                               config: Dict[str, Any] = None) -> str:
        """Generate a response using Vertex AI based on the query and documents.
        
        Args:
            query: User query string
            documents: List of relevant documents
            conversation_history: Optional conversation history
            config: Configuration parameters for generation
            
        Returns:
            Generated response
        """
        await self._init_components()
        
        # Extract config parameters
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)
        
        try:
            # Format prompt using template
            prompt = self._template.render(
                query=query,
                documents=documents,
                conversation_history=conversation_history or []
            )
            
            # Log the prompt for debugging (but truncate for clarity)
            truncated_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.debug(f"Generated prompt: {truncated_prompt}")
            
            # Configure generation parameters
            gen_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": config.get("top_p", 0.95),
                "top_k": config.get("top_k", 40)
            }
            
            # Generate response using new VertexGenAI API
            if conversation_history:
                # Use chat completion with history
                messages = []
                for msg in conversation_history:
                    role = "user" if msg["role"].lower() == "user" else "model"
                    messages.append({"role": role, "parts": [{"text": msg["content"]}]})
                
                # Add current prompt
                messages.append({"role": "user", "parts": [{"text": prompt}]})
                
                response = await self._generation_model.chat_completion(
                    messages=messages,
                    **gen_config
                )
            else:
                # Generate without chat history
                response = await self._generation_model.generate_content(
                    prompt=prompt,
                    **gen_config
                )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            else:
                # Handle response format change if needed
                return str(response)
            
        except Exception as e:
            logger.error(f"Vertex AI response generation failed: {str(e)}", exc_info=True)
            raise GenerationError(f"Failed to generate response: {str(e)}") from e
