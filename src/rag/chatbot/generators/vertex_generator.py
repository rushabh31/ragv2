import logging
import json
import os
from typing import Dict, Any, List, Optional
import jinja2

from src.rag.chatbot.generators.base_generator import BaseGenerator
from src.rag.core.interfaces.base import Document
from src.rag.shared.utils.vertex_ai import VertexGenAI
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
        self.model_name = self.config.get("model_name", "gemini-1.5-pro")
        self.template_path = self.config.get("prompt_template", "./templates/rag_prompt.jinja2")
        self._vertex_ai = None
        self._template = None
        
        # Load vertex configuration
        config_manager = ConfigManager()
        self.vertex_config = config_manager.get_config("vertex")
        
    async def _init_components(self):
        """Initialize Vertex AI client and prompt template lazily."""
        if self._vertex_ai is None:
            # Initialize with model name
            self._vertex_ai = VertexGenAI(model_name=self.model_name)
            
            # Set the vertex configuration for token-based auth
            if self.vertex_config:
                self._vertex_ai.vertex_config = self.vertex_config
                logger.info(f"Initialized Vertex AI client with model: {self.model_name} and token-based authentication")
            else:
                logger.warning("Vertex AI configuration not found. Token-based authentication won't be available.")
        
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
            
            # Generate response
            if conversation_history:
                # Convert conversation history to Vertex AI format for the updated API
                chat_history = []
                for msg in conversation_history:
                    role = "user" if msg["role"].lower() == "user" else "assistant"
                    chat_history.append({"role": role, "content": msg["content"]})
                
                # Generate with chat history
                response = await self._vertex_ai.generate_content(
                    prompt=prompt,
                    chat_history=chat_history,
                    generation_config=gen_config
                )
            else:
                # Generate without chat history
                response = await self._vertex_ai.generate_content(
                    prompt=prompt,
                    generation_config=gen_config
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
