from fastapi import FastAPI
from abc.ingestion.v1.api import router as ingestion_router
# from abc.chatbot.v1.api import router as chatbot_router  # Placeholder for chatbot router

app = FastAPI()
app.include_router(ingestion_router)
# app.include_router(chatbot_router)  # Uncomment when chatbot router is implemented 