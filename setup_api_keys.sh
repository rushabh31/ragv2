#!/bin/bash
# Script to set up API keys for the RAG system

# Prompt for the Groq API key
echo "Please enter your Groq API key (starts with 'gsk_'):"
read groq_api_key

# Check if .env file exists, create if not
if [ ! -f .env ]; then
    touch .env
    echo "Created new .env file"
fi

# Check if GROQ_API_KEY already exists in .env
if grep -q "GROQ_API_KEY" .env; then
    # Replace existing key
    sed -i '' "s/GROQ_API_KEY=.*/GROQ_API_KEY=$groq_api_key/" .env
    echo "Updated GROQ_API_KEY in .env file"
else
    # Add new key
    echo "GROQ_API_KEY=$groq_api_key" >> .env
    echo "Added GROQ_API_KEY to .env file"
fi

# Set the key in the current environment
export GROQ_API_KEY="$groq_api_key"
echo "API key has been exported to current shell environment"

echo ""
echo "Setup complete. To use this API key in your current session, run:"
echo "source .env"
echo ""
echo "To verify, run:"
echo "echo \$GROQ_API_KEY"
