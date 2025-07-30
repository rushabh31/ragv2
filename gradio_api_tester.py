#!/usr/bin/env python3
"""
Professional Gradio UI for Testing RAG System API Endpoints
"""

import gradio as gr
import requests
import json
from typing import Tuple

# Configuration
INGESTION_BASE_URL = "http://localhost:8000"
CHATBOT_BASE_URL = "http://localhost:8001"
DEFAULT_SOEID = "test-user-123"
DEFAULT_API_KEY = "test-api-key"

# Custom CSS for professional styling
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.header-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.header-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.gr-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.json-output {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-family: 'Monaco', 'Menlo', monospace !important;
}
"""

class APITester:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': DEFAULT_API_KEY,
            'soeid': DEFAULT_SOEID
        })
    
    def format_response(self, response: requests.Response) -> Tuple[str, str]:
        try:
            if response.status_code == 200:
                status = f"✅ SUCCESS ({response.status_code})"
                content = json.dumps(response.json(), indent=2)
            else:
                status = f"❌ ERROR ({response.status_code})"
                try:
                    content = json.dumps(response.json(), indent=2)
                except:
                    content = response.text
        except Exception as e:
            status = f"❌ EXCEPTION"
            content = str(e)
        return status, content
    
    def upload_document(self, file, soeid: str, options: str = "") -> Tuple[str, str]:
        if not file:
            return "❌ ERROR", "No file selected"
        
        try:
            headers = {'soeid': soeid, 'X-API-Key': DEFAULT_API_KEY}
            files = {'file': (file.name, open(file.name, 'rb'))}
            data = {'options': options} if options else {}
            
            response = requests.post(
                f"{INGESTION_BASE_URL}/ingest/upload",
                files=files,
                data=data,
                headers=headers
            )
            return self.format_response(response)
        except Exception as e:
            return "❌ EXCEPTION", str(e)
    
    def check_job_status(self, job_id: str, soeid: str) -> Tuple[str, str]:
        try:
            headers = {'soeid': soeid, 'X-API-Key': DEFAULT_API_KEY}
            response = requests.get(
                f"{INGESTION_BASE_URL}/ingest/status/{job_id}",
                headers=headers
            )
            return self.format_response(response)
        except Exception as e:
            return "❌ EXCEPTION", str(e)
    
    def list_documents(self, soeid: str, page: int = 1, page_size: int = 20) -> Tuple[str, str]:
        try:
            headers = {'soeid': soeid, 'X-API-Key': DEFAULT_API_KEY}
            params = {'page': page, 'page_size': page_size}
            response = requests.get(
                f"{INGESTION_BASE_URL}/ingest/documents",
                headers=headers,
                params=params
            )
            return self.format_response(response)
        except Exception as e:
            return "❌ EXCEPTION", str(e)
    
    def send_chat_message(self, query: str, soeid: str, session_id: str = None, 
                         use_retrieval: bool = True, use_history: bool = True,
                         use_chat_history: bool = False, chat_history_days: int = 7,
                         metadata: str = "") -> Tuple[str, str]:
        try:
            headers = {'soeid': soeid, 'X-API-Key': DEFAULT_API_KEY}
            data = {
                'query': query,
                'use_retrieval': use_retrieval,
                'use_history': use_history,
                'use_chat_history': use_chat_history,
                'chat_history_days': chat_history_days
            }
            
            if session_id:
                data['session_id'] = session_id
            if metadata:
                data['metadata_json'] = metadata
            
            response = requests.post(
                f"{CHATBOT_BASE_URL}/chat/message",
                data=data,
                headers=headers
            )
            return self.format_response(response)
        except Exception as e:
            return "❌ EXCEPTION", str(e)
    
    def get_chat_history(self, soeid: str) -> Tuple[str, str]:
        try:
            headers = {'soeid': soeid, 'X-API-Key': DEFAULT_API_KEY}
            response = requests.get(
                f"{CHATBOT_BASE_URL}/chat/history/{soeid}",
                headers=headers
            )
            return self.format_response(response)
        except Exception as e:
            return "❌ EXCEPTION", str(e)
    
    def get_memory_stats(self, soeid: str) -> Tuple[str, str]:
        try:
            headers = {'soeid': soeid, 'X-API-Key': DEFAULT_API_KEY}
            response = requests.get(
                f"{CHATBOT_BASE_URL}/chat/memory/stats",
                headers=headers
            )
            return self.format_response(response)
        except Exception as e:
            return "❌ EXCEPTION", str(e)
    
    def delete_user_history(self, soeid: str) -> Tuple[str, str]:
        try:
            headers = {'soeid': soeid, 'X-API-Key': DEFAULT_API_KEY}
            response = requests.delete(
                f"{CHATBOT_BASE_URL}/chat/history/{soeid}",
                headers=headers
            )
            return self.format_response(response)
        except Exception as e:
            return "❌ EXCEPTION", str(e)

api_tester = APITester()

def create_app():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="RAG System API Tester",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")
    ) as app:
        
        # Header
        gr.HTML(f"""
        <div class="header-container">
            <h1 class="header-title">🚀 RAG System API Tester</h1>
            <p style="text-align: center; color: #64748b; font-size: 1.1rem;">
                Professional testing interface for ingestion and chatbot endpoints
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # Ingestion Tab
            with gr.Tab("📄 Document Ingestion"):
                with gr.Column(elem_classes=["card"]):
                    gr.HTML("<h2 style='color: #667eea;'>📄 Document Ingestion</h2>")
                    
                    soeid_input = gr.Textbox(
                        label="SOEID (User ID)",
                        value=DEFAULT_SOEID,
                        placeholder="Enter user ID"
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("📤 Upload Document"):
                            with gr.Row():
                                with gr.Column():
                                    file_input = gr.File(
                                        label="Select Document",
                                        file_types=[".pdf", ".docx", ".txt", ".md"]
                                    )
                                    options_input = gr.Textbox(
                                        label="Options (JSON or key=value)",
                                        placeholder='{"metadata": {"category": "report"}}',
                                        lines=2
                                    )
                                    upload_btn = gr.Button("🚀 Upload Document", variant="primary")
                                
                                with gr.Column():
                                    upload_status = gr.Textbox(label="Status", interactive=False)
                                    upload_response = gr.Code(
                                        label="Response",
                                        language="json",
                                        elem_classes=["json-output"]
                                    )
                        
                        with gr.Tab("📊 Job Status"):
                            with gr.Row():
                                with gr.Column():
                                    job_id_input = gr.Textbox(
                                        label="Job ID",
                                        placeholder="Enter job ID from upload"
                                    )
                                    status_btn = gr.Button("🔍 Check Status")
                                
                                with gr.Column():
                                    job_status = gr.Textbox(label="Status", interactive=False)
                                    job_response = gr.Code(
                                        label="Response",
                                        language="json",
                                        elem_classes=["json-output"]
                                    )
                        
                        with gr.Tab("📋 List Documents"):
                            with gr.Row():
                                with gr.Column():
                                    page_input = gr.Number(label="Page", value=1, minimum=1)
                                    page_size_input = gr.Number(label="Page Size", value=20)
                                    list_btn = gr.Button("📋 List Documents")
                                
                                with gr.Column():
                                    list_status = gr.Textbox(label="Status", interactive=False)
                                    list_response = gr.Code(
                                        label="Response",
                                        language="json",
                                        elem_classes=["json-output"]
                                    )
            
            # Chatbot Tab
            with gr.Tab("💬 Chatbot Testing"):
                with gr.Column(elem_classes=["card"]):
                    gr.HTML("<h2 style='color: #764ba2;'>💬 Chatbot Interaction</h2>")
                    
                    chat_soeid_input = gr.Textbox(
                        label="SOEID (User ID)",
                        value=DEFAULT_SOEID,
                        placeholder="Enter user ID"
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("💬 Send Message"):
                            with gr.Row():
                                with gr.Column():
                                    query_input = gr.Textbox(
                                        label="Message",
                                        placeholder="Enter your question",
                                        lines=3
                                    )
                                    session_id_input = gr.Textbox(
                                        label="Session ID (optional)",
                                        placeholder="Leave empty for new session"
                                    )
                                    
                                    with gr.Row():
                                        use_retrieval = gr.Checkbox(label="Use Retrieval", value=True)
                                        use_history = gr.Checkbox(label="Use History", value=True)
                                        use_chat_history = gr.Checkbox(label="Use Chat History", value=False)
                                    
                                    chat_history_days = gr.Number(
                                        label="Chat History Days",
                                        value=7,
                                        minimum=1
                                    )
                                    
                                    metadata_input = gr.Textbox(
                                        label="Metadata (JSON)",
                                        placeholder='{"key": "value"}',
                                        lines=2
                                    )
                                    
                                    send_btn = gr.Button("💬 Send Message", variant="primary")
                                
                                with gr.Column():
                                    chat_status = gr.Textbox(label="Status", interactive=False)
                                    chat_response = gr.Code(
                                        label="Response",
                                        language="json",
                                        elem_classes=["json-output"]
                                    )
                        
                        with gr.Tab("📜 Chat History"):
                            with gr.Row():
                                with gr.Column():
                                    history_btn = gr.Button("📜 Get Chat History")
                                
                                with gr.Column():
                                    history_status = gr.Textbox(label="Status", interactive=False)
                                    history_response = gr.Code(
                                        label="Response",
                                        language="json",
                                        elem_classes=["json-output"]
                                    )
                        
                        with gr.Tab("🧠 Memory Management"):
                            with gr.Row():
                                with gr.Column():
                                    stats_btn = gr.Button("📊 Get Memory Stats")
                                    delete_btn = gr.Button("🗑️ Delete User History", variant="stop")
                                    gr.HTML("<small style='color: #dc2626;'>⚠️ Deletes all chat history</small>")
                                
                                with gr.Column():
                                    memory_status = gr.Textbox(label="Status", interactive=False)
                                    memory_response = gr.Code(
                                        label="Response",
                                        language="json",
                                        elem_classes=["json-output"]
                                    )
            
            # API Info Tab
            with gr.Tab("ℹ️ API Information"):
                with gr.Column(elem_classes=["card"]):
                    gr.HTML(f"""
                    <h2 style='color: #667eea;'>📚 API Endpoints Reference</h2>
                    
                    <h3 style='color: #667eea;'>📄 Ingestion API</h3>
                    <ul>
                        <li><strong>POST /ingest/upload</strong> - Upload document</li>
                        <li><strong>GET /ingest/status/{{job_id}}</strong> - Check job status</li>
                        <li><strong>GET /ingest/documents</strong> - List documents</li>
                    </ul>
                    
                    <h3 style='color: #764ba2;'>💬 Chatbot API</h3>
                    <ul>
                        <li><strong>POST /chat/message</strong> - Send message</li>
                        <li><strong>GET /chat/history/{{soeid}}</strong> - Get chat history</li>
                        <li><strong>GET /chat/memory/stats</strong> - Memory statistics</li>
                        <li><strong>DELETE /chat/history/{{soeid}}</strong> - Delete history</li>
                    </ul>
                    
                    <div style='background: #f8fafc; padding: 20px; border-radius: 12px; margin-top: 20px;'>
                        <h4>🔧 Configuration</h4>
                        <p><strong>Ingestion API:</strong> {INGESTION_BASE_URL}</p>
                        <p><strong>Chatbot API:</strong> {CHATBOT_BASE_URL}</p>
                        <p><strong>Default SOEID:</strong> {DEFAULT_SOEID}</p>
                    </div>
                    """)
        
        # Event handlers
        upload_btn.click(
            fn=lambda file, soeid, options: api_tester.upload_document(file, soeid, options),
            inputs=[file_input, soeid_input, options_input],
            outputs=[upload_status, upload_response]
        )
        
        status_btn.click(
            fn=lambda job_id, soeid: api_tester.check_job_status(job_id, soeid),
            inputs=[job_id_input, soeid_input],
            outputs=[job_status, job_response]
        )
        
        list_btn.click(
            fn=lambda soeid, page, page_size: api_tester.list_documents(soeid, page, page_size),
            inputs=[soeid_input, page_input, page_size_input],
            outputs=[list_status, list_response]
        )
        
        send_btn.click(
            fn=lambda query, soeid, session_id, use_ret, use_hist, use_chat_hist, chat_days, metadata: 
                api_tester.send_chat_message(query, soeid, session_id, use_ret, use_hist, use_chat_hist, chat_days, metadata),
            inputs=[query_input, chat_soeid_input, session_id_input, use_retrieval, use_history, 
                   use_chat_history, chat_history_days, metadata_input],
            outputs=[chat_status, chat_response]
        )
        
        history_btn.click(
            fn=lambda soeid: api_tester.get_chat_history(soeid),
            inputs=[chat_soeid_input],
            outputs=[history_status, history_response]
        )
        
        stats_btn.click(
            fn=lambda soeid: api_tester.get_memory_stats(soeid),
            inputs=[chat_soeid_input],
            outputs=[memory_status, memory_response]
        )
        
        delete_btn.click(
            fn=lambda soeid: api_tester.delete_user_history(soeid),
            inputs=[chat_soeid_input],
            outputs=[memory_status, memory_response]
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )
