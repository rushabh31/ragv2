"""
OpenAI Generator for RAG System.

This module provides a RAG-compatible generator wrapper for OpenAI models
using the universal authentication system.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
import jinja2

from src.rag.chatbot.generators.base_generator import BaseGenerator
from src.rag.core.interfaces.base import Document
from src.rag.core.exceptions.exceptions import GenerationError
from src.models.generation import OpenAIGenAI

logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseGenerator):
    """
    OpenAI generator for RAG system.
    
    This class wraps the OpenAIGenAI model to provide RAG-compatible
    generation capabilities with document context and prompt templating.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI generator.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "Meta-Llama-3-70B-Instruct")
        self.base_url = self.config.get("base_url")
        self.template_path = self.config.get("prompt_template", "./templates/rag_prompt.jinja2")
        
        self._openai_model = None
        self._template = None
        
        logger.info(f"Initialized OpenAIGenerator with model: {self.model_name}")
    
    async def _init_components(self):
        """Initialize OpenAI model and prompt template lazily."""
        if self._openai_model is None:
            self._openai_model = OpenAIGenAI(
                model_name=self.model_name,
                base_url=self.base_url
            )
            logger.info(f"Initialized OpenAI model: {self.model_name}")
        
        if self._template is None:
            try:
                with open(self.template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                self._template = jinja2.Template(template_content)
                logger.info(f"Loaded prompt template from: {self.template_path}")
            except Exception as e:
                logger.warning(f"Failed to load template from {self.template_path}: {str(e)}")
                # Use default template
                self._template = jinja2.Template("""
Based on the following context documents, please answer the user's question.

Context:
{% for doc in context_documents %}
Document {{ loop.index }}:
{{ doc.content }}

{% endfor %}

Question: {{ query }}

Please provide a comprehensive answer based on the context provided above.
                """.strip())
    
    async def _generate_response(self, query: str, context_documents: List[Document], config: Dict[str, Any]) -> str:
        """
        Generate response using OpenAI.
        
        Args:
            query: User query
            context_documents: List of relevant documents
            config: Generation configuration
            
        Returns:
            Generated response text
        """
        await self._init_components()
        
        try:
            # Render the prompt template
            user_prompt = self._template.render(
                query=query,
                context_documents=context_documents
            )
            
            # Get generation parameters
            max_tokens = config.get("max_tokens")
            temperature = config.get("temperature", 0.7)
            system_message = config.get("system_message", "You are a helpful AI assistant that answers questions based on provided context.")
            
            # Generate response
            response = await self._openai_model.chat_completion(
                prompt=user_prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.info(f"Generated response using {self.model_name}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise GenerationError(f"Response generation failed: {str(e)}") from e
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        if self._openai_model:
            return self._openai_model.get_auth_health_status()
        return {"status": "not_initialized"}
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        await self._init_components()
        return await self._openai_model.validate_authentication()
