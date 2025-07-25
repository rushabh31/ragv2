"""
Anthropic Vertex AI Generator for RAG System.

This module provides a RAG-compatible generator wrapper for Anthropic Vertex AI models
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
from src.models.generation import AnthropicVertexGenAI

logger = logging.getLogger(__name__)


class AnthropicVertexGenerator(BaseGenerator):
    """
    Anthropic Vertex AI generator for RAG system.
    
    This class wraps the AnthropicVertexGenAI model to provide RAG-compatible
    generation capabilities with document context and prompt templating.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic Vertex AI generator.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "claude-3-5-sonnet@20240229")
        self.region = self.config.get("region", "us-east5")
        self.project_id = self.config.get("project_id")
        self.template_path = self.config.get("prompt_template", "./templates/rag_prompt.jinja2")
        
        self._anthropic_model = None
        self._template = None
        
        logger.info(f"Initialized AnthropicVertexGenerator with model: {self.model_name}")
    
    async def _init_components(self):
        """Initialize Anthropic model and prompt template lazily."""
        if self._anthropic_model is None:
            self._anthropic_model = AnthropicVertexGenAI(
                model_name=self.model_name,
                region=self.region,
                project_id=self.project_id
            )
            logger.info(f"Initialized Anthropic Vertex model: {self.model_name}")
        
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
        Generate response using Anthropic Vertex AI.
        
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
            prompt = self._template.render(
                query=query,
                context_documents=context_documents
            )
            
            # Get generation parameters
            max_tokens = config.get("max_tokens", 4000)
            temperature = config.get("temperature", 0.7)
            
            # Generate response
            response = await self._anthropic_model.chat_completion(
                prompt=prompt,
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
        if self._anthropic_model:
            return self._anthropic_model.get_auth_health_status()
        return {"status": "not_initialized"}
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        await self._init_components()
        return await self._anthropic_model.validate_authentication()
