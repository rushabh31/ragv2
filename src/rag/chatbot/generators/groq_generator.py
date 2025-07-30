"""Groq Generator for RAG system."""

import logging
from typing import Dict, Any, List, Optional
from jinja2 import Template, FileSystemLoader, Environment

from src.rag.chatbot.generators.base_generator import BaseGenerator
from src.rag.core.interfaces.base import Document
from src.models.generation.groq_gen import GroqGenAI
from src.rag.core.exceptions.exceptions import GenerationError
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GroqGenerator(BaseGenerator):
    """Groq-based generator for RAG responses."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Groq generator.
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: Groq model name (default: llama-3.1-70b-versatile)
                - temperature: Generation temperature (default: 0.2)
                - max_tokens: Maximum tokens to generate (default: 1024)
                - top_p: Top-p sampling parameter (default: 0.95)
                - prompt_template: Path to Jinja2 prompt template
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "llama-3.1-70b-versatile")
        self.template_path = self.config.get("prompt_template", "./templates/rag_prompt.jinja2")
        self._groq_model = None
        self._template = None
        
        # Load generation configuration from system config
        config_manager = ConfigManager()
        system_config = config_manager.get_section("generation")
        if system_config and system_config.get("provider") == "groq":
            generation_config = system_config.get("config", {})
            self.model_name = generation_config.get("model_name", self.model_name)
            self.template_path = generation_config.get("prompt_template", self.template_path)
        
    async def _init_components(self):
        """Initialize Groq model and prompt template lazily."""
        if self._groq_model is None:
            try:
                # Initialize with model name using Groq API
                self._groq_model = GroqGenAI(model_name=self.model_name)
                
                # Validate authentication
                is_valid = await self._groq_model.validate_authentication()
                if is_valid:
                    logger.info(f"Initialized Groq model: {self.model_name}")
                else:
                    logger.warning("Groq model authentication validation failed")
            except Exception as e:
                logger.error(f"Failed to initialize Groq model: {str(e)}")
                raise
        
        if self._template is None:
            try:
                # Load Jinja2 template
                with open(self.template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                self._template = Template(template_content)
                logger.info(f"Loaded prompt template from: {self.template_path}")
            except FileNotFoundError:
                logger.warning(f"Template file not found: {self.template_path}, using default template")
                # Default template
                default_template = """You are a helpful AI assistant. Use the following context to answer the user's question.

Context:
{% for doc in context %}
{{ doc.content }}
{% endfor %}

Question: {{ query }}

Answer: """ 
                self._template = Template(default_template)
            except Exception as e:
                logger.error(f"Failed to load template: {str(e)}")
                raise GenerationError(f"Template loading failed: {str(e)}")
    
    async def _generate_response(self, 
                               query: str, 
                               documents: List[Document], 
                               conversation_history: Optional[List[Dict[str, str]]] = None, 
                               config: Dict[str, Any] = None) -> str:
        """Generate response using Groq model.
        
        Args:
            query: User query
            documents: List of relevant documents
            conversation_history: Optional conversation history
            config: Generation configuration
            
        Returns:
            Generated response
        """
        try:
            # Initialize components
            await self._init_components()
            
            # Render prompt template
            prompt = self._template.render(
                query=query,
                context=documents
            )
            
            # Get generation parameters from config (with defaults)
            config = config or {}
            gen_config = {
                "temperature": config.get("temperature", 0.2),
                "max_tokens": config.get("max_tokens", 1024),
                "top_p": config.get("top_p", 0.95)
            }
            
            logger.debug(f"Generating response with Groq model: {self.model_name}")
            
            # Use conversation_history parameter instead of config
            chat_history = conversation_history or []
            
            if chat_history:
                # Build messages for chat completion
                messages = []
                
                # Add chat history
                for msg in chat_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    messages.append({"role": role, "content": content})
                
                # Add current prompt
                messages.append({"role": "user", "content": prompt})
                
                response = await self._groq_model.chat_completion(
                    messages=messages,
                    **gen_config
                )
            else:
                # Generate without chat history
                response = await self._groq_model.generate_content(
                    prompt=prompt,
                    **gen_config
                )
            
            if not response:
                raise GenerationError("Empty response from Groq model")
            
            logger.debug(f"Generated response length: {len(response)}")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Groq response generation failed: {str(e)}")
            raise GenerationError(f"Generation failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Groq model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "groq",
            "model_name": self.model_name,
            "template_path": self.template_path,
            "api_based": True
        }
