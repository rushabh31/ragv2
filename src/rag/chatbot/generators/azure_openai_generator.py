"""
Azure OpenAI Generator for RAG System.

This module provides a RAG-compatible generator wrapper for Azure OpenAI models
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
from src.models.generation import AzureOpenAIGenAI

logger = logging.getLogger(__name__)


class AzureOpenAIGenerator(BaseGenerator):
    """
    Azure OpenAI generator for RAG system.
    
    This class wraps the AzureOpenAIGenAI model to provide RAG-compatible
    generation capabilities with document context and prompt templating.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Azure OpenAI generator.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "GPT4-o")
        self.azure_endpoint = self.config.get("azure_endpoint")
        self.api_version = self.config.get("api_version", "2023-05-15")
        self.template_path = self.config.get("prompt_template", "./templates/rag_prompt.jinja2")
        
        self._azure_model = None
        self._template = None
        
        logger.info(f"Initialized AzureOpenAIGenerator with model: {self.model_name}")
    
    async def _init_components(self):
        """Initialize Azure OpenAI model and prompt template lazily."""
        if self._azure_model is None:
            self._azure_model = AzureOpenAIGenAI(
                model_name=self.model_name,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            logger.info(f"Initialized Azure OpenAI model: {self.model_name}")
        
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
        Generate response using Azure OpenAI.
        
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
            temperature = config.get("temperature", 0.2)
            stream = config.get("stream", False)
            system_message = config.get("system_message", "You are a helpful AI assistant that answers questions based on provided context.")
            
            # Generate response
            response = await self._azure_model.chat_completion(
                prompt=user_prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            
            logger.info(f"Generated response using {self.model_name}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise GenerationError(f"Response generation failed: {str(e)}") from e
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        if self._azure_model:
            return self._azure_model.get_auth_health_status()
        return {"status": "not_initialized"}
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        await self._init_components()
        return await self._azure_model.validate_authentication()
