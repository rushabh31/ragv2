"""Groq-based generator for RAG system."""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Union
import time
import asyncio
from groq import AsyncGroq
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape

from src.rag.core.interfaces.base import Document
from src.rag.chatbot.generators.base_generator import BaseGenerator
from src.rag.core.exceptions.exceptions import GenerationError
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GroqGenerator(BaseGenerator):
    """Generator using the Groq API."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Groq generator.
        
        Args:
            config: Configuration for the generator
        """
        super().__init__(config)
        self._client = None
        self._env = None
        self._template = None
        
    async def _init_client(self):
        """Initialize the Groq client."""
        if self._client is None:
            try:
                # Get API key from config or environment
                api_key = self.config.get("api_key")
                logger.info(f"API key from config: {'Found' if api_key else 'Not found'} (value might be a template)")
                
                # Check for environment variable placeholder
                if api_key and api_key.startswith("${"): 
                    env_var_name = api_key.strip("${}")
                    logger.info(f"Found environment variable placeholder: {env_var_name}")
                    env_value = os.environ.get(env_var_name)
                    if env_value:
                        api_key = env_value
                        logger.info(f"Using value from environment variable {env_var_name}")
                    else:
                        logger.warning(f"Environment variable {env_var_name} not found or empty")
                        api_key = None
                
                # Fall back to direct environment variable
                if not api_key:
                    api_key = os.environ.get("GROQ_API_KEY")
                    logger.info(f"API key from GROQ_API_KEY environment variable: {'Found' if api_key else 'Not found'}")
                
                if not api_key:
                    raise GenerationError("Groq API key not provided in config or environment variables")
                    
                # Debug masked key
                if len(api_key) > 8:
                    masked_key = api_key[:4] + '***' + api_key[-4:]
                    logger.info(f"Using API key: {masked_key}")
                else:
                    logger.warning("API key seems too short, might be invalid")
                
                # Create async client
                self._client = AsyncGroq(api_key=api_key)
                logger.info("Groq client initialized successfully")
                
                # Initialize template environment
                templates_dir = self.config.get("templates_dir", "templates")
                
                # Use absolute path for templates
                import os
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                absolute_templates_dir = os.path.join(base_dir, templates_dir)
                
                logger.info(f"Looking for templates in: {absolute_templates_dir}")
                
                self._env = Environment(
                    loader=FileSystemLoader(absolute_templates_dir),
                    autoescape=select_autoescape(['html', 'xml'])
                )
                
                # Load the template
                template_name = self.config.get("template", "rag_prompt.jinja2")
                try:
                    self._template = self._env.get_template(template_name)
                    logger.info(f"Loaded prompt template: {template_name}")
                except Exception as template_error:
                    # Use a default template if loading fails
                    logger.warning(f"Failed to load template from {templates_dir}/{template_name}: {str(template_error)}")
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
                    self._template = Template(template_str)
                
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}", exc_info=True)
                raise GenerationError(f"Failed to initialize Groq client: {str(e)}")
    
    async def _generate_response(self,
                               query: str,
                               documents: List[Document] = None,
                               conversation_history: List[Dict[str, str]] = None,
                               config: Dict[str, Any] = None) -> str:
        """Generate a response using Groq.
        
        Args:
            query: User query
            documents: Retrieved documents
            conversation_history: Previous conversation
            config: Additional configuration
            
        Returns:
            Generated response text
        """
        await self._init_client()
        
        try:
            # Get generation parameters
            model = config.get("model") or self.config.get("model", "llama3-70b-8192")
            temperature = config.get("temperature") or self.config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens") or self.config.get("max_tokens", 1024)
            top_p = config.get("top_p") or self.config.get("top_p", 0.9)
            
            # Prepare the prompt using the template
            prompt = self._template.render(
                query=query,
                documents=documents or [],
                conversation_history=conversation_history or []
            )
            
            # Log prompt length
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Track generation time
            start_time = time.time()
            
            # Call Groq API
            chat_completion = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            # Extract response text
            response_text = chat_completion.choices[0].message.content
            
            # Log generation statistics
            elapsed_time = time.time() - start_time
            logger.info(f"Generated response in {elapsed_time:.2f}s with {len(response_text)} characters")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Groq generation failed: {str(e)}", exc_info=True)
            raise GenerationError(f"Failed to generate response using Groq: {str(e)}")
